"""
ProctorGuard v6 — Gaze + Call-Pose Accuracy Fix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT CAUSES fixed vs v5:
  ✓ Looking DOWN not detected  — v5 only checked LEFT/RIGHT via profile
                                  cascade. v6 uses eye Y-position inside
                                  face ROI to detect downward gaze.
  ✓ Phone-to-ear not detected  — skin-mask & face-overlap exclusions were
                                  blocking the call-pose candidate because
                                  the phone sits BESIDE the face and shares
                                  its bounding box. v6 uses a dedicated
                                  "face-perimeter scan" for call pose that
                                  bypasses those exclusions.
  ✓ Multiple-face FP           — requires eye detection in EACH candidate
                                  face before counting it as real. Also
                                  raises confirm threshold to 5 frames.

Install:
  pip install flask flask-cors opencv-python numpy
  pip install ultralytics   ← optional
"""

import cv2
import numpy as np
import base64
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

EVIDENCE_DIR = Path("cheating_evidence")
EVIDENCE_DIR.mkdir(exist_ok=True)
sessions: dict = {}

# ══════════════════════════════════════════════════════════════════
#  THRESHOLDS  — edit here to tune
# ══════════════════════════════════════════════════════════════════
SNAP_COOLDOWN_S           = 8.0
PERIODIC_INTERVAL_S       = 5.0
LOOK_AWAY_VIOLATION_S     = 12.0   # seconds continuous → violation
LOOK_AWAY_DECAY           = 2.5    # streak shrinks per sec when looking forward
NO_FACE_WARNING_S         = 4.0
NO_FACE_VIOLATION_S       = 10.0
NO_FACE_DEDUCT_EVERY_S    = 30.0
FACE_OFFCENTRE_THR        = 0.44
FACE_OFFCENTRE_STREAK_S   = 5.0
MULTI_FACE_CONFIRM_FRAMES = 5      # ← raised from 4
PHONE_CONFIRM_FRAMES      = 2
PHONE_SCORE_THR           = 0.50
PHONE_CALL_SCORE_THR      = 0.46   # lower thr since call-pose is harder
POSE_WINDOW               = 6      # rolling vote window
POSE_AWAY_MIN             = 4      # votes to declare LOOKING_AWAY

# eye Y-position thresholds (normalised inside face ROI)
# 0 = top of face bbox, 1 = bottom
EYE_DOWN_THR   = 0.32   # eyes above this line → head tilted down
EYE_NORMAL_MAX = 0.62   # eyes below this → could still be forward

WEIGHTS = {
    "PHONE_DETECTED":    30,
    "MULTIPLE_FACES":    25,
    "TAB_SWITCH":        15,
    "NO_FACE_SUSTAINED": 10,
    "LOOKING_AWAY":       5,
    "FACE_OUT_OF_FRAME":  5,
    "COPY_PASTE":         8,
}

# ─── snapshot cooldown ────────────────────────────────────────────
_snap_ts: dict = {}
def cooldown_ok(sid, key):
    k = (sid, key); now = time.time()
    if now - _snap_ts.get(k, 0) >= SNAP_COOLDOWN_S:
        _snap_ts[k] = now; return True
    return False

# ─── Haar cascades ────────────────────────────────────────────────
_HD          = cv2.data.haarcascades
face_front   = cv2.CascadeClassifier(_HD + "haarcascade_frontalface_default.xml")
face_alt2    = cv2.CascadeClassifier(_HD + "haarcascade_frontalface_alt2.xml")
face_profile = cv2.CascadeClassifier(_HD + "haarcascade_profileface.xml")
eye_casc     = cv2.CascadeClassifier(_HD + "haarcascade_eye.xml")
eye_glass    = cv2.CascadeClassifier(_HD + "haarcascade_eye_tree_eyeglasses.xml")

# ─── YOLO (optional) ──────────────────────────────────────────────
yolo_model = None; yolo_available = False
try:
    from ultralytics import YOLO
    for pt in ("yolov8s.pt", "yolov8n.pt"):
        try:
            yolo_model = YOLO(pt); yolo_available = True
            print(f"[INFO] ✓ YOLO: {pt}"); break
        except Exception: pass
    if not yolo_available: print("[WARN] YOLO weights missing — ensemble only")
except ImportError: print("[INFO] ultralytics absent — ensemble only")

DEVICE_CLS = {67: "cell phone", 63: "laptop", 65: "remote"}

# ══════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════
def decode_frame(b64):
    raw = base64.b64decode(b64.split(",")[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

def encode_frame(frame, q=82):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def save_evidence(frame, sid, tag):
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fname = f"{sid}_{ts}_{tag[:30]}.jpg"
    cv2.imwrite(str(EVIDENCE_DIR / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return fname

def nms(boxes, thr=0.40):
    if not boxes: return []
    b  = np.array(boxes, dtype=float)
    x1,y1 = b[:,0],b[:,1]; x2,y2 = b[:,0]+b[:,2],b[:,1]+b[:,3]
    areas  = (x2-x1)*(y2-y1); order = np.argsort(areas)[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i); rest=order[1:]
        ix1=np.maximum(x1[i],x1[rest]); iy1=np.maximum(y1[i],y1[rest])
        ix2=np.minimum(x2[i],x2[rest]); iy2=np.minimum(y2[i],y2[rest])
        iw=np.maximum(0.,ix2-ix1);      ih=np.maximum(0.,iy2-iy1)
        order=rest[(iw*ih)/(areas[rest]+1e-6)<thr]
    return [tuple(b[i].astype(int)) for i in keep]

def rect_iou(a, b):
    ax2,ay2=a[0]+a[2],a[1]+a[3]; bx2,by2=b[0]+b[2],b[1]+b[3]
    ix1=max(a[0],b[0]);iy1=max(a[1],b[1]);ix2=min(ax2,bx2);iy2=min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.
    inter=(ix2-ix1)*(iy2-iy1)
    return inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-6)

def _clahe_gray(frame):
    lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB); l,a,b_ch=cv2.split(lab)
    l=cv2.createCLAHE(2.0,(8,8)).apply(l)
    return cv2.cvtColor(
        cv2.cvtColor(cv2.merge([l,a,b_ch]),cv2.COLOR_LAB2BGR),
        cv2.COLOR_BGR2GRAY)

# ══════════════════════════════════════════════════════════════════
#  EYE DETECTION HELPER
#  Returns list of eye rects WITHIN the face ROI coordinate system,
#  using both plain eye cascade and glasses-aware cascade.
# ══════════════════════════════════════════════════════════════════
def detect_eyes_in_face(gray_full, fx, fy, fw, fh):
    """
    Returns eyes found inside face ROI.
    Uses BOTH eye cascades (handles glasses like in the sample image).
    """
    roi = gray_full[fy:fy+fh, fx:fx+fw]
    eyes = []
    min_eye = (fw//7, fh//9)

    for casc, sc, nb in [(eye_casc, 1.1, 4), (eye_glass, 1.1, 3)]:
        det = casc.detectMultiScale(roi, scaleFactor=sc, minNeighbors=nb,
                                    minSize=min_eye)
        if len(det):
            eyes.extend(det.tolist())

    if not eyes:
        return []

    # NMS & keep only top-half of face (eyes are in top 60%)
    eyes = nms(eyes, thr=0.40)
    eyes = [e for e in eyes if (e[1]+e[3]/2) / fh < 0.65]
    return eyes

# ══════════════════════════════════════════════════════════════════
#  FACE DETECTION  — eyes required to confirm each face
# ══════════════════════════════════════════════════════════════════
def detect_faces(frame):
    gray  = _clahe_gray(frame)
    h, w  = frame.shape[:2]
    minsz = max(70, min(h,w)//7)
    raw   = []

    for casc,sc,nb in [(face_front,1.07,8),(face_alt2,1.07,7)]:
        det = casc.detectMultiScale(gray, sc, nb,
                                    minSize=(minsz,minsz),
                                    maxSize=(int(w*.88),int(h*.88)))
        if len(det): raw.extend(det.tolist())

    faces = nms(raw, thr=0.35)
    faces = [f for f in faces if 0.55 < f[2]/max(f[3],1) < 1.90]

    # ── Eye-presence filter to eliminate false "second face" ──────
    # For each candidate, require at least 1 eye IF the box is large
    # enough to reliably run the eye cascade (width > 80 px).
    confirmed = []
    for f in faces:
        fx,fy,fw,fh = f
        if fw < 80:
            confirmed.append(f)   # too small to check eyes — keep
            continue
        eyes = detect_eyes_in_face(gray, fx, fy, fw, fh)
        if eyes:
            confirmed.append(f)
        # else: no eyes found → likely a shadow/poster, discard

    # Additional size sanity: secondary face ≥ 45 % area of primary
    if len(confirmed) > 1:
        areas = [f[2]*f[3] for f in confirmed]
        mx    = max(areas)
        confirmed = [f for f in confirmed if f[2]*f[3] >= mx*0.45]

    return confirmed

# ══════════════════════════════════════════════════════════════════
#  HEAD POSE  — detects SIDEWAYS and DOWNWARD gaze
#
#  Three signals combined via rolling vote:
#    1. Profile cascade          → left/right turn
#    2. Eye Y-position in ROI    → head tilted DOWN
#    3. Eye absence (large face) → turned enough to hide eyes
# ══════════════════════════════════════════════════════════════════
def _raw_pose(gray, fx, fy, fw, fh, img_w):
    """
    Returns: "FORWARD" | "SIDEWAYS" | "DOWN"
    """
    roi_gray = gray[fy:fy+fh, fx:fx+fw]

    # ── Signal 1: Profile cascade (left/right) ──────────────────
    mp = max(40,fw//2), max(40,fh//2)
    for g in [gray, cv2.flip(gray,1)]:
        dets = face_profile.detectMultiScale(g,1.1,3,minSize=mp)
        for (px,py,pw,ph) in (dets if len(dets) else []):
            ix1=max(fx,px);iy1=max(fy,py)
            ix2=min(fx+fw,px+pw);iy2=min(fy+fh,py+ph)
            if ix2>ix1 and iy2>iy1:
                inter=(ix2-ix1)*(iy2-iy1)
                if inter/(fw*fh+pw*ph-inter+1e-6)>0.18:
                    return "SIDEWAYS"

    # Aspect ratio narrowing (head turned sideways)
    if fw/max(fh,1) < 0.56: return "SIDEWAYS"

    # Heavy lateral drift
    cx_n = abs(fx+fw/2 - img_w/2)/(img_w/2+1e-6)
    if cx_n > 0.54: return "SIDEWAYS"

    # ── Signal 2: Eye position → DOWN detection ──────────────────
    eyes = []
    min_e = (fw//7, fh//9)
    for casc,sc,nb in [(eye_casc,1.1,4),(eye_glass,1.1,3)]:
        det = casc.detectMultiScale(roi_gray,sc,nb,minSize=min_e)
        if len(det): eyes.extend(det.tolist())
    eyes = nms(eyes, thr=0.40)
    eyes = [e for e in eyes if (e[1]+e[3]/2)/fh < 0.65]

    if eyes:
        # Normalised Y of eye centres (0=top, 1=bottom of face box)
        eye_y_norms = [(e[1]+e[3]/2)/fh for e in eyes]
        avg_eye_y   = np.mean(eye_y_norms)

        # Eyes shifted toward top → chin raised, head DOWN
        if avg_eye_y < EYE_DOWN_THR:
            return "DOWN"

        # Eyes in normal range → forward
        if avg_eye_y <= EYE_NORMAL_MAX:
            return "FORWARD"

        # Eyes very low → extreme tilt up (rare in exam, treat as forward)
        return "FORWARD"

    # ── Signal 3: No eyes on large face → away ───────────────────
    if fw > 80:
        return "SIDEWAYS"   # eyes hidden = turned or covered

    return "FORWARD"

def head_pose_voted(gray, face, img_w, vote_buf):
    fx,fy,fw,fh = face
    raw = _raw_pose(gray, fx,fy,fw,fh, img_w)
    vote_buf.append(raw)
    away_count = sum(1 for v in vote_buf if v in ("SIDEWAYS","DOWN"))
    if away_count >= POSE_AWAY_MIN:
        # Return the most common non-FORWARD label
        from collections import Counter
        c = Counter(v for v in vote_buf if v!="FORWARD")
        return c.most_common(1)[0][0] if c else "FORWARD"
    return "FORWARD"

# ══════════════════════════════════════════════════════════════════
#  PHONE CALL-POSE — face-perimeter scan
#
#  Instead of scanning the whole frame and excluding face-overlapping
#  boxes (which threw away the phone-to-ear candidate), we scan a
#  STRIP around the face bounding box specifically looking for a
#  tall narrow rectangle sitting beside/on the face.
# ══════════════════════════════════════════════════════════════════
def detect_call_pose_phone(frame, gray, faces):
    """
    Dedicated detector for phone-held-to-ear.
    Scans a horizontal strip from (face_left - 1.2×face_w) to
    (face_right + 1.2×face_w) around each face, looks for tall
    dark/bright rectangles with phone-like aspect ratio that touch
    or overlap the face bbox.
    Returns list of device dicts (may be empty).
    """
    h, w  = frame.shape[:2]
    found = []

    for (fx,fy,fw,fh) in faces:
        # Expanded search region around face
        margin_x = int(fw * 1.3)
        margin_y = int(fh * 0.5)
        rx1 = max(0,  fx - margin_x)
        ry1 = max(0,  fy - margin_y)
        rx2 = min(w,  fx + fw + margin_x)
        ry2 = min(h,  fy + fh + margin_y)
        if rx2 <= rx1 or ry2 <= ry1: continue

        strip_gray  = gray [ry1:ry2, rx1:rx2]
        strip_frame = frame[ry1:ry2, rx1:rx2]

        blur  = cv2.GaussianBlur(strip_gray,(5,5),0)
        edges = cv2.dilate(cv2.Canny(blur,18,70),
                           np.ones((3,3),np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            sw   = rx2-rx1; sh = ry2-ry1
            if area < 1500 or area > sw*sh*0.70: continue

            bx,by,bw,bh = cv2.boundingRect(cnt)
            long_s  = max(bw,bh); short_s = min(bw,bh)
            if short_s < 1: continue
            aspect = long_s/short_s

            # Phone portrait: 1.5–3.5; call pose can be up to 4
            if not (1.50 <= aspect <= 4.50): continue

            # Must actually overlap or touch the face box
            # (in strip-local coords: face is at fx-rx1, fy-ry1)
            lfx = fx-rx1; lfy = fy-ry1
            box_iou = rect_iou((bx,by,bw,bh),(lfx,lfy,fw,fh))
            # Also accept if the box is directly adjacent (gap < 20 px)
            gap_x = max(0, max(bx,lfx) - min(bx+bw, lfx+fw))
            gap_y = max(0, max(by,lfy) - min(by+bh, lfy+fh))
            adjacent = (gap_x < 20 and gap_y < fh*0.9)
            if box_iou < 0.05 and not adjacent: continue

            # Score the candidate
            roi = strip_gray[by:by+bh, bx:bx+bw]
            if roi.size==0: continue

            score = 0.
            e_dens = np.count_nonzero(cv2.Canny(roi,35,110))/roi.size
            if 0.03 < e_dens < 0.42: score += 0.28
            mean_b = np.mean(roi); std_b = np.std(roi)
            if mean_b < 90:  score += 0.24   # dark phone body
            elif mean_b > 180: score += 0.20  # lit screen
            if std_b < 60:   score += 0.18
            # Bonus: box is on the SIDE of the face (not in front)
            cx_phone = bx + bw/2; cx_face = lfx + fw/2
            if abs(cx_phone - cx_face) > fw*0.3: score += 0.20
            # Vertical alignment with face
            if abs((by+bh/2) - (lfy+fh/2)) < fh*0.7: score += 0.12

            if score >= PHONE_CALL_SCORE_THR:
                # Convert back to full-frame coords
                found.append({
                    "label": "cell phone",
                    "conf":  round(score,2),
                    "bbox":  [rx1+bx, ry1+by, bw, bh],
                    "pose":  "call",
                })

    if not found: return []
    # NMS and return best
    best = nms([f["bbox"] for f in found], thr=0.45)
    if best:
        top = next((f for f in found if f["bbox"]==list(best[0])), found[0])
        return [top]
    return []

# ══════════════════════════════════════════════════════════════════
#  PHONE DETECTION — ENSEMBLE  (hand-held, whole-frame scan)
# ══════════════════════════════════════════════════════════════════
def _skin_mask(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb,(0,133,77),(255,173,127))

def _skin_frac(mask,bx,by,bw,bh):
    roi=mask[by:by+bh,bx:bx+bw]
    return np.count_nonzero(roi)/max(roi.size,1)

def detect_phone_handheld(frame, gray, faces, skin):
    h,w = frame.shape[:2]
    blur  = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.dilate(cv2.Canny(blur,22,85),np.ones((3,3),np.uint8),iterations=1)
    cnts,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cands  = []

    for cnt in cnts:
        area=cv2.contourArea(cnt)
        if area<2200 or area>w*h*0.42: continue
        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.024*peri,True)
        if not(4<=len(approx)<=8): continue
        bx,by,bw,bh=cv2.boundingRect(cnt)
        if bw<24 or bh<34: continue
        long_s=max(bw,bh); short_s=min(bw,bh)
        if short_s<1: continue
        aspect=long_s/short_s
        if not(1.60<=aspect<=2.65): continue   # only normal hold here

        hull_a=cv2.contourArea(cv2.convexHull(cnt))
        if area/max(hull_a,1)<0.55: continue
        if _skin_frac(skin,bx,by,bw,bh)>0.52: continue
        if any(rect_iou((bx,by,bw,bh),tuple(f))>0.32 for f in faces): continue

        roi=gray[by:by+bh,bx:bx+bw]
        if roi.size==0: continue
        e_dens=np.count_nonzero(cv2.Canny(roi,40,120))/roi.size
        mean_b=np.mean(roi); std_b=np.std(roi)
        score=0.
        if 0.04<e_dens<0.40: score+=0.26
        if mean_b<105: score+=0.22
        elif mean_b>195: score+=0.18
        if std_b<62: score+=0.18
        if by>h*0.18: score+=0.16
        if area/(bw*bh)>0.72: score+=0.10
        if score>=PHONE_SCORE_THR:
            cands.append({"label":"cell phone","conf":round(score,2),
                          "bbox":[bx,by,bw,bh],"pose":"hand"})
    if not cands: return []
    best=nms([c["bbox"] for c in cands],thr=0.50)
    if best:
        top=next((c for c in cands if c["bbox"]==list(best[0])),cands[0])
        return [top]
    return []

def detect_devices(frame, faces):
    gray = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    skin = _skin_mask(frame)
    found = []

    # 1. YOLO (optional)
    if yolo_available and yolo_model:
        ih,iw=frame.shape[:2]; sc=960/iw if iw<960 else 1.
        rf=cv2.resize(frame,(int(iw*sc),int(ih*sc))) if sc!=1. else frame
        res=yolo_model(rf,verbose=False,conf=0.30,iou=0.45)[0]
        for box in res.boxes:
            cid=int(box.cls[0])
            if cid not in DEVICE_CLS: continue
            conf=float(box.conf[0])
            x1,y1,x2,y2=map(int,box.xyxy[0])
            bbox=[int(x1/sc),int(y1/sc),int((x2-x1)/sc),int((y2-y1)/sc)]
            bx2,by2,bw2,bh2=bbox
            if _skin_frac(skin,bx2,by2,bw2,bh2)>0.52: continue
            if any(rect_iou(tuple(bbox),tuple(f))>0.32 for f in faces): continue
            found.append({"label":DEVICE_CLS[cid],"conf":round(conf,2),
                          "bbox":bbox,"pose":"yolo"})

    # 2. Call-pose scan (face-perimeter, bypasses skin/overlap exclusions)
    for e in detect_call_pose_phone(frame, gray, faces):
        if not any(rect_iou(tuple(e["bbox"]),tuple(f["bbox"]))>0.45 for f in found):
            found.append(e)

    # 3. Hand-held ensemble (whole frame, strict exclusions)
    for e in detect_phone_handheld(frame, gray, faces, skin):
        if not any(rect_iou(tuple(e["bbox"]),tuple(f["bbox"]))>0.45 for f in found):
            found.append(e)

    return found

# ══════════════════════════════════════════════════════════════════
#  FACE OFF-CENTRE
# ══════════════════════════════════════════════════════════════════
def face_offcentre(shape, face):
    fh,fw=shape[:2]; fx,fy,bw,bh=face
    nx=abs(fx+bw/2-fw/2)/(fw/2+1e-6)
    ny=abs(fy+bh/2-fh/2)/(fh/2+1e-6)
    return nx>FACE_OFFCENTRE_THR or ny>FACE_OFFCENTRE_THR

# ══════════════════════════════════════════════════════════════════
#  ANNOTATE
# ══════════════════════════════════════════════════════════════════
def annotate(frame, faces, devices, violations, warnings, trust, sess):
    out=frame.copy(); fh,fw=out.shape[:2]

    for (x,y,w,h) in faces:
        col=(30,210,30) if len(faces)==1 else (0,40,230)
        cv2.rectangle(out,(x,y),(x+w,y+h),col,2)
        cv2.putText(out,"FACE" if len(faces)==1 else f"FACE×{len(faces)}",
                    (x,max(y-6,14)),cv2.FONT_HERSHEY_SIMPLEX,.55,col,2)

    for d in devices:
        bx,by,bw,bh=d["bbox"]
        cv2.rectangle(out,(bx,by),(bx+bw,by+bh),(0,40,230),2)
        cv2.putText(out,f"{d['label']} [{d['pose']}] {d['conf']:.0%}",
                    (bx,max(by-6,14)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,40,230),2)

    # Look-away progress bar
    la=sess["look_away_streak"]
    if la>0:
        bar_w=int(fw*0.26); prog=min(int(bar_w*la/LOOK_AWAY_VIOLATION_S),bar_w)
        cv2.rectangle(out,(fw-bar_w-10,8),(fw-10,24),(40,40,40),-1)
        cv2.rectangle(out,(fw-bar_w-10,8),(fw-bar_w-10+prog,24),(0,200,255),-1)
        cv2.putText(out,f"Away {la:.0f}s / {LOOK_AWAY_VIOLATION_S:.0f}s",
                    (fw-bar_w-10,38),cv2.FONT_HERSHEY_SIMPLEX,.40,(220,220,220),1)

    # Trust bar
    bar=int(fw*0.22)
    col=(30,200,30) if trust>=60 else (0,165,255) if trust>=40 else (0,40,230)
    cv2.rectangle(out,(10,8),(10+bar,26),(30,30,30),-1)
    cv2.rectangle(out,(10,8),(10+int(bar*max(0,trust)/100),26),col,-1)
    cv2.putText(out,f"Trust {trust:.0f}",(14,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)

    # Status
    if violations:
        msg="⚡ VIOLATION: "+violations[0].replace("_"," "); col=(0,40,230)
    elif warnings:
        msg="⚡ WARNING: "+warnings[0].replace("_"," ");     col=(0,140,255)
    else:
        msg="✓ CLEAR"; col=(30,200,30)
    cv2.putText(out,msg,(10,fh-10),cv2.FONT_HERSHEY_SIMPLEX,.62,col,2)
    return out

# ══════════════════════════════════════════════════════════════════
#  SESSION
# ══════════════════════════════════════════════════════════════════
def get_session(sid):
    if sid not in sessions:
        sessions[sid]={
            "trust":100.,
            "no_face_streak":0.,"no_face_accum":0.,
            "look_away_streak":0.,
            "off_centre_streak":0.,
            "phone_streak":0,"multi_streak":0,
            "pose_votes":deque(maxlen=POSE_WINDOW),
            "last_ts":time.time(),"last_periodic_ts":time.time(),
            "violation_count":0,"tab_switches":0,
            "log":[],"screenshots":[],
        }
    return sessions[sid]

def deduct(sess,key,reason):
    pts=WEIGHTS.get(key,0); sess["trust"]=max(0.,sess["trust"]-pts)
    sess["violation_count"]+=1
    sess["log"].append({"ts":datetime.now().isoformat(timespec="seconds"),
                        "event":key,"reason":reason,
                        "pts":-pts,"score":round(sess["trust"],1)})

# ══════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════
def analyze(b64, sid):
    sess  = get_session(sid)
    frame = decode_frame(b64)
    gray  = _clahe_gray(frame)
    ih,iw = frame.shape[:2]

    now = time.time()
    dt  = min(now-sess["last_ts"], 3.0); sess["last_ts"]=now
    periodic_due=(now-sess["last_periodic_ts"])>=PERIODIC_INTERVAL_S
    if periodic_due: sess["last_periodic_ts"]=now

    faces   = detect_faces(frame)
    fc      = len(faces)
    devices = detect_devices(frame, faces)

    violations=[]; warnings=[]

    # ── No face ──────────────────────────────────────────────────
    if fc==0:
        sess["no_face_streak"]+=dt; warnings.append("NO_FACE")
        if sess["no_face_streak"]>=NO_FACE_WARNING_S:
            warnings.append("NO_FACE_SUSTAINED")
        if sess["no_face_streak"]>=NO_FACE_VIOLATION_S:
            sess["no_face_accum"]+=dt
            if sess["no_face_accum"]>=NO_FACE_DEDUCT_EVERY_S:
                violations.append("NO_FACE_SUSTAINED")
                deduct(sess,"NO_FACE_SUSTAINED",f"No face {sess['no_face_streak']:.0f}s")
                sess["no_face_accum"]=0.
    else:
        sess["no_face_streak"]=0.; sess["no_face_accum"]=0.

    # ── Multiple faces ───────────────────────────────────────────
    if fc>1:
        sess["multi_streak"]+=1; warnings.append("MULTIPLE_FACES")
        if sess["multi_streak"]>=MULTI_FACE_CONFIRM_FRAMES:
            violations.append("MULTIPLE_FACES")
            deduct(sess,"MULTIPLE_FACES",f"{fc} faces ×{sess['multi_streak']}f")
            sess["multi_streak"]=0
    else:
        sess["multi_streak"]=0

    # ── Single face: gaze + centering ────────────────────────────
    if fc==1:
        gaze = head_pose_voted(gray, faces[0], iw, sess["pose_votes"])

        if gaze in ("SIDEWAYS","DOWN"):
            sess["look_away_streak"]+=dt
            warnings.append("LOOKING_AWAY")
            # Annotate with direction
            if gaze=="DOWN": warnings.append("LOOKING_DOWN")
            if sess["look_away_streak"]>=LOOK_AWAY_VIOLATION_S:
                violations.append("LOOKING_AWAY")
                deduct(sess,"LOOKING_AWAY",
                       f"{gaze} {sess['look_away_streak']:.1f}s")
                sess["look_away_streak"]=LOOK_AWAY_VIOLATION_S*0.4
        else:
            sess["look_away_streak"]=max(0.,
                sess["look_away_streak"]-dt*LOOK_AWAY_DECAY)

        if face_offcentre(frame.shape, faces[0]):
            sess["off_centre_streak"]+=dt; warnings.append("FACE_OUT_OF_FRAME")
            if sess["off_centre_streak"]>=FACE_OFFCENTRE_STREAK_S:
                violations.append("FACE_OUT_OF_FRAME")
                deduct(sess,"FACE_OUT_OF_FRAME",
                       f"Off-centre {sess['off_centre_streak']:.1f}s")
                sess["off_centre_streak"]=0.
        else:
            sess["off_centre_streak"]=max(0.,sess["off_centre_streak"]-dt*1.5)

    # ── Phone ────────────────────────────────────────────────────
    if devices:
        sess["phone_streak"]+=1; warnings.append("PHONE_DETECTED")
        if sess["phone_streak"]>=PHONE_CONFIRM_FRAMES:
            violations.append("PHONE_DETECTED")
            tags="+".join(set(d.get("pose","?") for d in devices))
            deduct(sess,"PHONE_DETECTED",
                   f"{', '.join(set(d['label'] for d in devices))} [{tags}]")
    else:
        sess["phone_streak"]=0

    # ── Annotate ─────────────────────────────────────────────────
    ann = annotate(frame, faces, devices, violations, warnings,
                   sess["trust"], sess)

    # ── Save evidence for EVERY violation type ───────────────────
    saved_files=[]
    for vkey in violations:
        if cooldown_ok(sid, vkey):
            fname=save_evidence(ann,sid,vkey)
            saved_files.append(fname)
            sess["screenshots"].append({
                "file":fname,
                "ts":datetime.now().isoformat(timespec="seconds"),
                "violation":vkey,"trust":round(sess["trust"],1),
            })

    # Mid-threshold look-away snapshot (evidence before violation fires)
    if ("LOOKING_AWAY" in warnings and "LOOKING_AWAY" not in violations
            and sess["look_away_streak"]>=6.0):
        if cooldown_ok(sid,"LOOKING_AWAY_WARN"):
            fname=save_evidence(ann,sid,"LOOKING_AWAY_WARN")
            saved_files.append(fname)
            sess["screenshots"].append({
                "file":fname,"ts":datetime.now().isoformat(timespec="seconds"),
                "violation":"LOOKING_AWAY_WARN","trust":round(sess["trust"],1),
            })

    return {
        "session_id":        sid,
        "face_count":        fc,
        "devices":           devices,
        "violations":        violations,
        "warnings":          warnings,
        "trust_score":       round(sess["trust"],1),
        "violation_count":   sess["violation_count"],
        "no_face_streak":    round(sess["no_face_streak"],1),
        "look_away_streak":  round(sess["look_away_streak"],1),
        "look_away_limit":   LOOK_AWAY_VIOLATION_S,
        "off_centre_streak": round(sess["off_centre_streak"],1),
        "phone_streak":      sess["phone_streak"],
        "periodic_due":      periodic_due,
        "annotated":         encode_frame(ann),
        "saved_files":       saved_files,
        "yolo_enabled":      yolo_available,
        "timestamp":         now,
    }

# ══════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════
@app.route("/health")
def health():
    return jsonify({"status":"ok","yolo":yolo_available,
                    "look_away_limit_s":LOOK_AWAY_VIOLATION_S,
                    "periodic_interval_s":PERIODIC_INTERVAL_S})

@app.route("/analyze", methods=["POST"])
def route_analyze():
    d=request.get_json(silent=True) or {}
    if "frame" not in d: return jsonify({"error":"No frame"}),400
    try:
        return jsonify(analyze(d["frame"],d.get("session_id","default")))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error":str(e)}),500

@app.route("/event", methods=["POST"])
def route_event():
    d=request.get_json(silent=True) or {}
    sid=d.get("session_id","default"); sess=get_session(sid); ev=d.get("event","")
    if ev=="TAB_SWITCH":
        sess["tab_switches"]+=1; deduct(sess,"TAB_SWITCH",f"Switch #{sess['tab_switches']}")
    elif ev=="COPY_PASTE":
        deduct(sess,"COPY_PASTE","Clipboard intercepted")
    return jsonify({"trust_score":round(sess["trust"],1)})

@app.route("/snapshot", methods=["POST"])
def route_snapshot():
    d=request.get_json(silent=True) or {}
    if "frame" not in d: return jsonify({"error":"No frame"}),400
    frame=decode_frame(d["frame"])
    fname=save_evidence(frame,d.get("session_id","default"),d.get("label","manual"))
    return jsonify({"saved_file":fname})

@app.route("/session/<sid>")
def route_session(sid):
    s=get_session(sid)
    return jsonify({"session_id":sid,"trust_score":round(s["trust"],1),
                    "violation_count":s["violation_count"],
                    "tab_switches":s["tab_switches"],
                    "log":s["log"][-60:],"screenshots":s["screenshots"]})

@app.route("/evidence")
def route_evidence():
    files=sorted(EVIDENCE_DIR.glob("*.jpg"),key=os.path.getmtime,reverse=True)
    return jsonify([{"filename":f.name,"size_kb":round(f.stat().st_size/1024,1),
                     "modified":datetime.fromtimestamp(
                         f.stat().st_mtime).isoformat(timespec="seconds")}
                    for f in files[:100]])

@app.route("/evidence/<fname>")
def route_serve(fname):
    return send_from_directory(str(EVIDENCE_DIR.resolve()),fname)

@app.route("/reset/<sid>", methods=["POST"])
def route_reset(sid):
    sessions.pop(sid,None); return jsonify({"status":"reset"})

if __name__=="__main__":
    print("="*68)
    print("  ProctorGuard v6  →  http://localhost:5050")
    print(f"  YOLO             : {'ON ✓' if yolo_available else 'OFF'}")
    print(f"  Gaze detection   : SIDEWAYS + DOWN (eye Y-position in face ROI)")
    print(f"  Phone call-pose  : face-perimeter scan, no skin/overlap exclusion")
    print(f"  Phone hand-held  : ensemble scan, score≥{PHONE_SCORE_THR}")
    print(f"  Multi-face guard : eye-presence filter + {MULTI_FACE_CONFIRM_FRAMES} consecutive frames")
    print(f"  Look-away limit  : {LOOK_AWAY_VIOLATION_S}s (warn screenshot at 6s)")
    print(f"  Periodic monitor : every {PERIODIC_INTERVAL_S}s → periodic_due flag")
    print(f"  Evidence folder  : {EVIDENCE_DIR.resolve()}")
    print("="*68)
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
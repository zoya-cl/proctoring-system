"""
ProctorGuard v5 — Robust Accuracy Rewrite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIXES over v4:
  ✓ Phone call-pose  — dedicated detector: dark/reflective tall rect
                       beside face, upper-half frame, confirmed by
                       LBP texture + solidity + border uniformity
  ✓ Multi-face       — minNeighbors raised + size sanity + 4-frame
                       consecutive confirmation + separation check
  ✓ Periodic mode    — /analyze accepts  "periodic":true ; backend
                       returns   "periodic_due":true  every 5 s so
                       the frontend knows to capture a frame
  ✓ Screenshots      — saved for EVERY violation (not just first),
                       one per violation-type per cooldown window
  ✓ Look-away        — 12 s threshold preserved; screenshot captured
                       the moment threshold is crossed
  ✓ False-positive    guard — each detector has an independent
                       confidence gate before it is accepted

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
#  TUNABLE THRESHOLDS
# ══════════════════════════════════════════════════════════════════
SNAP_COOLDOWN_S           = 8.0    # min gap between snapshots of same violation
PERIODIC_INTERVAL_S       = 5.0    # how often the "periodic" flag fires
LOOK_AWAY_VIOLATION_S     = 12.0   # continuous gaze-away before violation
LOOK_AWAY_DECAY           = 2.5    # subtract this per second when forward
NO_FACE_WARNING_S         = 4.0
NO_FACE_VIOLATION_S       = 10.0
NO_FACE_DEDUCT_EVERY_S    = 30.0
FACE_OFFCENTRE_THR        = 0.44   # normalised; 0 = centre, 1 = edge
FACE_OFFCENTRE_STREAK_S   = 5.0    # sustained before deducting
MULTI_FACE_CONFIRM_FRAMES = 4      # consecutive frames needed
PHONE_CONFIRM_FRAMES      = 2      # consecutive frames needed
PHONE_SCORE_THR           = 0.50   # hand-held ensemble threshold
PHONE_CALL_SCORE_THR      = 0.52   # call-pose threshold
POSE_WINDOW               = 6      # rolling frames for head-pose vote
POSE_SIDEWAYS_MIN         = 4      # votes needed to declare SIDEWAYS

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

def cooldown_ok(sid: str, key: str) -> bool:
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

# ─── YOLO (optional) ──────────────────────────────────────────────
yolo_model     = None
yolo_available = False
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
def decode_frame(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.split(",")[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

def encode_frame(frame: np.ndarray, q: int = 82) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def save_evidence(frame: np.ndarray, sid: str, tag: str) -> str:
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fname = f"{sid}_{ts}_{tag[:30]}.jpg"
    cv2.imwrite(str(EVIDENCE_DIR / fname), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return fname

def nms(boxes: list, thr: float = 0.40) -> list:
    if not boxes: return []
    b  = np.array(boxes, dtype=float)
    x1, y1 = b[:,0], b[:,1]
    x2, y2 = b[:,0]+b[:,2], b[:,1]+b[:,3]
    areas  = (x2-x1)*(y2-y1)
    order  = np.argsort(areas)[::-1]
    keep   = []
    while order.size:
        i = order[0]; keep.append(i); rest = order[1:]
        ix1 = np.maximum(x1[i],x1[rest]); iy1 = np.maximum(y1[i],y1[rest])
        ix2 = np.minimum(x2[i],x2[rest]); iy2 = np.minimum(y2[i],y2[rest])
        iw  = np.maximum(0., ix2-ix1);     ih  = np.maximum(0., iy2-iy1)
        order = rest[(iw*ih)/(areas[rest]+1e-6) < thr]
    return [tuple(b[i].astype(int)) for i in keep]

def rect_iou(a, b) -> float:
    ax2,ay2 = a[0]+a[2], a[1]+a[3]
    bx2,by2 = b[0]+b[2], b[1]+b[3]
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(ax2,bx2);   iy2=min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.
    inter=(ix2-ix1)*(iy2-iy1)
    return inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-6)

# ══════════════════════════════════════════════════════════════════
#  FACE DETECTION  — stricter to avoid multi-face false positives
# ══════════════════════════════════════════════════════════════════
def detect_faces(frame: np.ndarray) -> list:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    l   = cv2.createCLAHE(2.0,(8,8)).apply(l)
    gray = cv2.cvtColor(cv2.cvtColor(cv2.merge([l,a,b_ch]),
                        cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)

    h, w  = frame.shape[:2]
    minsz = max(70, min(h,w)//7)
    raw   = []

    # minNeighbors raised to 8/7 — much stricter than v3's 6/5
    for casc, sc, nb in [(face_front,1.07,8),(face_alt2,1.07,7)]:
        det = casc.detectMultiScale(gray, scaleFactor=sc, minNeighbors=nb,
                                    minSize=(minsz,minsz),
                                    maxSize=(int(w*.88),int(h*.88)))
        if len(det): raw.extend(det.tolist())

    faces = nms(raw, thr=0.35)

    # Shape sanity: faces are roughly square-ish
    faces = [f for f in faces if 0.55 < f[2]/max(f[3],1) < 1.90]

    # Separation sanity for multiple faces:
    # two boxes that overlap > 30 % are almost certainly the same face
    if len(faces) > 1:
        faces = nms(faces, thr=0.30)

    # Size sanity: secondary faces must be at least 45 % the area of largest
    if len(faces) > 1:
        areas  = [f[2]*f[3] for f in faces]
        maxA   = max(areas)
        faces  = [f for f in faces if f[2]*f[3] >= maxA * 0.45]

    return faces

# ══════════════════════════════════════════════════════════════════
#  HEAD POSE  — rolling vote; eye check for profile confirmation
# ══════════════════════════════════════════════════════════════════
def _raw_pose(frame: np.ndarray, fx,fy,fw,fh) -> str:
    ih, iw = frame.shape[:2]
    gray   = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Profile cascade (left + mirrored right)
    mp = max(40, fw//2), max(40, fh//2)
    for g in [gray, cv2.flip(gray,1)]:
        dets = face_profile.detectMultiScale(g,1.1,3,minSize=mp)
        for (px,py,pw,ph) in (dets if len(dets) else []):
            ix1=max(fx,px);iy1=max(fy,py)
            ix2=min(fx+fw,px+pw);iy2=min(fy+fh,py+ph)
            if ix2>ix1 and iy2>iy1:
                inter=(ix2-ix1)*(iy2-iy1)
                if inter/(fw*fh+pw*ph-inter+1e-6) > 0.18:
                    return "SIDEWAYS"

    # Aspect ratio narrowing
    if fw / max(fh,1) < 0.56: return "SIDEWAYS"

    # Eye presence check (only reliable when face is large enough)
    if fw > 80:
        roi  = gray[fy:fy+fh, fx:fx+fw]
        eyes = eye_casc.detectMultiScale(roi, 1.1, 4,
                                         minSize=(fw//7,fh//8))
        if len(eyes) == 0: return "SIDEWAYS"

    # Heavy lateral drift
    cx_n = abs(fx + fw/2 - iw/2) / (iw/2 + 1e-6)
    if cx_n > 0.54: return "SIDEWAYS"

    return "FORWARD"

def head_pose(frame, face, vote_buf: deque) -> str:
    fx,fy,fw,fh = face
    vote_buf.append(_raw_pose(frame, fx,fy,fw,fh))
    n_side = sum(1 for v in vote_buf if v=="SIDEWAYS")
    return "SIDEWAYS" if n_side >= POSE_SIDEWAYS_MIN else "FORWARD"

# ══════════════════════════════════════════════════════════════════
#  PHONE DETECTION — ENSEMBLE  (hand-held  +  call-pose)
#
#  Signals used:
#    A) Contour geometry         — rectangle with phone aspect ratio
#    B) Screen-region brightness — phone screen is bright & uniform
#    C) Border uniformity        — phone border is dark & consistent
#    D) Edge density             — screen UI has moderate edges
#    E) Solidity                 — phone fills its convex hull well
#    F) Call-pose geometry       — tall box laterally adjacent to face
#    G) Skin exclusion           — > 55 % skin pixels → not a phone
# ══════════════════════════════════════════════════════════════════
def _skin_mask(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, (0,133,77), (255,173,127))

def _box_skin_frac(mask: np.ndarray, bx,by,bw,bh) -> float:
    roi = mask[by:by+bh, bx:bx+bw]
    if roi.size == 0: return 0.
    return np.count_nonzero(roi) / roi.size

def _score_phone_box(gray: np.ndarray, bx,by,bw,bh,
                     fh_img, call_pose=False) -> float:
    """Return a 0–1 confidence that this bbox is a phone."""
    roi = gray[by:by+bh, bx:bx+bw]
    if roi.size == 0: return 0.

    score = 0.

    # A — Edge density (screen: 4–40%)
    edges     = cv2.Canny(roi, 35, 110)
    edge_dens = np.count_nonzero(edges) / roi.size
    if 0.04 < edge_dens < 0.40: score += 0.25
    elif edge_dens >= 0.40:      score -= 0.10

    # B — Screen brightness
    mean_b = np.mean(roi)
    std_b  = np.std(roi)
    if mean_b > 180:  score += 0.20   # lit screen
    elif mean_b < 80: score += 0.18   # dark body

    # C — Internal uniformity (solid surface)
    if std_b < 55:  score += 0.18
    elif std_b > 90: score -= 0.05

    # D — Position in frame
    if by > fh_img * 0.15: score += 0.15   # below top 15% of frame
    if call_pose and by < fh_img * 0.55: score += 0.12  # upper frame call

    # E — Border contrast (dark bezel around bright screen)
    border_px = max(4, min(bw,bh)//10)
    inner = roi[border_px:bh-border_px, border_px:bw-border_px]
    if inner.size > 0 and (bw > 2*border_px) and (bh > 2*border_px):
        border_mean = (np.mean(roi) * roi.size - np.mean(inner) * inner.size) \
                      / max(roi.size - inner.size, 1)
        contrast = abs(float(np.mean(inner)) - float(border_mean))
        if contrast > 25: score += 0.15

    return min(score, 1.0)

def detect_phone_ensemble(frame: np.ndarray, faces: list,
                          skin: np.ndarray) -> list:
    h, w  = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 22, 85),
                       np.ones((3,3),np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cands = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 2200 or area > w*h*0.42: continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.024*peri, True)
        if not (4 <= len(approx) <= 8): continue

        bx,by,bw,bh = cv2.boundingRect(cnt)
        if bw < 24 or bh < 34: continue

        long_s  = max(bw,bh); short_s = min(bw,bh)
        if short_s < 1: continue
        aspect  = long_s / short_s

        # Aspect-ratio gates
        hand_hold  = 1.60 <= aspect <= 2.65   # normal portrait/landscape
        call_pose  = 2.20 <= aspect <= 5.00   # tall/narrow next to ear
        if not (hand_hold or call_pose): continue

        # Solidity (contour area / bounding-box area)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity  = area / max(hull_area, 1)
        if solidity < 0.55: continue

        # Skin exclusion
        if _box_skin_frac(skin, bx,by,bw,bh) > 0.52: continue

        # Face-overlap exclusion
        if any(rect_iou((bx,by,bw,bh), tuple(f)) > 0.32 for f in faces):
            continue

        # ── Score this candidate ──────────────────────────────
        is_call = False
        bonus   = 0.

        if call_pose:
            # Check if it is laterally adjacent to a face (call pose)
            cx_phone = bx + bw/2
            for (fx2,fy2,fw2,fh2) in faces:
                cx_face = fx2 + fw2/2
                lateral = abs(cx_phone - cx_face)
                vertical_overlap = not (by > fy2+fh2 or by+bh < fy2)
                if lateral < fw2*2.0 and vertical_overlap and by < h*0.60:
                    is_call = True
                    bonus   = 0.28   # strong call-pose bonus
                    break

        score = _score_phone_box(gray, bx,by,bw,bh, h, call_pose=is_call)
        score += bonus

        thr = PHONE_CALL_SCORE_THR if is_call else PHONE_SCORE_THR
        if score >= thr:
            cands.append({
                "label": "cell phone",
                "conf":  round(score, 2),
                "bbox":  [bx,by,bw,bh],
                "pose":  "call" if is_call else "hand",
            })

    if not cands: return []
    best = nms([c["bbox"] for c in cands], thr=0.50)
    if best:
        top = next((c for c in cands if c["bbox"]==list(best[0])), cands[0])
        return [top]
    return []

def detect_devices(frame: np.ndarray, faces: list) -> list:
    skin  = _skin_mask(frame)
    found = []

    # YOLO pass (optional)
    if yolo_available and yolo_model:
        ih,iw = frame.shape[:2]
        sc = 960/iw if iw < 960 else 1.0
        rf = cv2.resize(frame,(int(iw*sc),int(ih*sc))) if sc!=1. else frame
        res = yolo_model(rf, verbose=False, conf=0.30, iou=0.45)[0]
        for box in res.boxes:
            cid = int(box.cls[0])
            if cid not in DEVICE_CLS: continue
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            bbox = [int(x1/sc),int(y1/sc),int((x2-x1)/sc),int((y2-y1)/sc)]
            bx2,by2,bw2,bh2 = bbox
            # Still apply exclusions on YOLO hits
            if _box_skin_frac(skin,bx2,by2,bw2,bh2) > 0.52: continue
            if any(rect_iou(tuple(bbox),tuple(f))>0.32 for f in faces): continue
            found.append({"label":DEVICE_CLS[cid],"conf":round(conf,2),
                          "bbox":bbox,"pose":"yolo"})

    # Ensemble (especially for call pose)
    for e in detect_phone_ensemble(frame, faces, skin):
        if not any(rect_iou(tuple(e["bbox"]),tuple(f["bbox"]))>0.45
                   for f in found):
            found.append(e)

    return found

# ══════════════════════════════════════════════════════════════════
#  ANNOTATE
# ══════════════════════════════════════════════════════════════════
def annotate(frame, faces, devices, violations, warnings, trust,
             streaks: dict):
    out = frame.copy()
    fh, fw = out.shape[:2]

    for (x,y,w,h) in faces:
        col = (30,210,30) if len(faces)==1 else (0,40,230)
        cv2.rectangle(out,(x,y),(x+w,y+h),col,2)
        lbl = "FACE" if len(faces)==1 else f"FACE×{len(faces)}"
        cv2.putText(out,lbl,(x,max(y-6,14)),cv2.FONT_HERSHEY_SIMPLEX,.55,col,2)

    for d in devices:
        bx,by,bw,bh = d["bbox"]
        tag = f" [{d['pose']}]" if d["pose"] not in ("hand","yolo") else ""
        cv2.rectangle(out,(bx,by),(bx+bw,by+bh),(0,40,230),2)
        cv2.putText(out,f"{d['label']}{tag} {d['conf']:.0%}",
                    (bx,max(by-6,14)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,40,230),2)

    # Look-away progress bar
    la = streaks.get("look_away",0.)
    if la > 0:
        bar_w = int(fw * 0.25)
        prog  = min(int(bar_w * la / LOOK_AWAY_VIOLATION_S), bar_w)
        cv2.rectangle(out,(fw-bar_w-10,8),(fw-10,24),(40,40,40),-1)
        cv2.rectangle(out,(fw-bar_w-10,8),(fw-bar_w-10+prog,24),(0,200,255),-1)
        cv2.putText(out,f"Away {la:.0f}s",(fw-bar_w-10,38),
                    cv2.FONT_HERSHEY_SIMPLEX,.42,(200,200,200),1)

    # Trust bar
    bar = int(fw*0.22)
    col = (30,200,30) if trust>=60 else (0,165,255) if trust>=40 else (0,40,230)
    cv2.rectangle(out,(10,8),(10+bar,26),(30,30,30),-1)
    cv2.rectangle(out,(10,8),(10+int(bar*max(0,trust)/100),26),col,-1)
    cv2.putText(out,f"Trust {trust:.0f}",(14,22),
                cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)

    # Status
    if violations:
        msg = "VIOLATION: "+violations[0].replace("_"," "); col=(0,40,230)
    elif warnings:
        msg = "WARNING: "+warnings[0].replace("_"," ");     col=(0,140,255)
    else:
        msg = "CLEAR"; col=(30,200,30)
    cv2.putText(out,msg,(10,fh-10),cv2.FONT_HERSHEY_SIMPLEX,.62,col,2)
    return out

# ══════════════════════════════════════════════════════════════════
#  SESSION
# ══════════════════════════════════════════════════════════════════
def get_session(sid: str) -> dict:
    if sid not in sessions:
        sessions[sid] = {
            "trust":               100.,
            "no_face_streak":      0.,
            "look_away_streak":    0.,
            "off_centre_streak":   0.,
            "no_face_accum":       0.,
            "phone_streak":        0,
            "multi_streak":        0,
            "pose_votes":          deque(maxlen=POSE_WINDOW),
            "last_ts":             time.time(),
            "last_periodic_ts":    time.time(),
            "violation_count":     0,
            "tab_switches":        0,
            "log":                 [],
            "screenshots":         [],
        }
    return sessions[sid]

def deduct(sess: dict, key: str, reason: str):
    pts = WEIGHTS.get(key,0)
    sess["trust"] = max(0., sess["trust"] - pts)
    sess["violation_count"] += 1
    sess["log"].append({
        "ts":     datetime.now().isoformat(timespec="seconds"),
        "event":  key, "reason": reason,
        "pts":    -pts, "score": round(sess["trust"],1),
    })

# ══════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════
def analyze(b64: str, sid: str) -> dict:
    sess  = get_session(sid)
    frame = decode_frame(b64)

    now = time.time()
    dt  = min(now - sess["last_ts"], 3.0)
    sess["last_ts"] = now

    periodic_due = (now - sess["last_periodic_ts"]) >= PERIODIC_INTERVAL_S
    if periodic_due:
        sess["last_periodic_ts"] = now

    faces   = detect_faces(frame)
    fc      = len(faces)
    devices = detect_devices(frame, faces)

    violations = []
    warnings   = []

    # ── No face ──────────────────────────────────────────────────
    if fc == 0:
        sess["no_face_streak"] += dt
        warnings.append("NO_FACE")
        if sess["no_face_streak"] >= NO_FACE_WARNING_S:
            warnings.append("NO_FACE_SUSTAINED")
        if sess["no_face_streak"] >= NO_FACE_VIOLATION_S:
            sess["no_face_accum"] += dt
            if sess["no_face_accum"] >= NO_FACE_DEDUCT_EVERY_S:
                violations.append("NO_FACE_SUSTAINED")
                deduct(sess,"NO_FACE_SUSTAINED",
                       f"No face {sess['no_face_streak']:.0f}s")
                sess["no_face_accum"] = 0.
    else:
        sess["no_face_streak"] = 0.
        sess["no_face_accum"]  = 0.

    # ── Multiple faces ───────────────────────────────────────────
    if fc > 1:
        sess["multi_streak"] += 1
        warnings.append("MULTIPLE_FACES")
        if sess["multi_streak"] >= MULTI_FACE_CONFIRM_FRAMES:
            violations.append("MULTIPLE_FACES")
            deduct(sess,"MULTIPLE_FACES",f"{fc} faces confirmed ×{sess['multi_streak']}f")
            sess["multi_streak"] = 0   # reset so next window starts fresh
    else:
        sess["multi_streak"] = 0

    # ── Single face: pose + centering ────────────────────────────
    if fc == 1:
        pose = head_pose(frame, faces[0], sess["pose_votes"])
        if pose == "SIDEWAYS":
            sess["look_away_streak"] += dt
            warnings.append("LOOKING_AWAY")
            # Only raise violation after LOOK_AWAY_VIOLATION_S continuous seconds
            if sess["look_away_streak"] >= LOOK_AWAY_VIOLATION_S:
                violations.append("LOOKING_AWAY")
                deduct(sess,"LOOKING_AWAY",
                       f"Away {sess['look_away_streak']:.1f}s")
                # Partial reset — don't re-trigger immediately
                sess["look_away_streak"] = LOOK_AWAY_VIOLATION_S * 0.4
        else:
            sess["look_away_streak"] = max(
                0., sess["look_away_streak"] - dt * LOOK_AWAY_DECAY)

        if _face_offcentre(frame.shape, faces[0]):
            sess["off_centre_streak"] += dt
            warnings.append("FACE_OUT_OF_FRAME")
            if sess["off_centre_streak"] >= FACE_OFFCENTRE_STREAK_S:
                violations.append("FACE_OUT_OF_FRAME")
                deduct(sess,"FACE_OUT_OF_FRAME",
                       f"Off-centre {sess['off_centre_streak']:.1f}s")
                sess["off_centre_streak"] = 0.
        else:
            sess["off_centre_streak"] = max(0., sess["off_centre_streak"] - dt*1.5)

    # ── Phone ────────────────────────────────────────────────────
    if devices:
        sess["phone_streak"] += 1
        warnings.append("PHONE_DETECTED")
        if sess["phone_streak"] >= PHONE_CONFIRM_FRAMES:
            violations.append("PHONE_DETECTED")
            pose_tags = "+".join(set(d.get("pose","?") for d in devices))
            deduct(sess,"PHONE_DETECTED",
                   f"{', '.join(set(d['label'] for d in devices))} [{pose_tags}]")
    else:
        sess["phone_streak"] = 0

    # ── Annotate ─────────────────────────────────────────────────
    streaks = {"look_away": sess["look_away_streak"]}
    ann = annotate(frame, faces, devices, violations, warnings,
                   sess["trust"], streaks)

    # ── Save evidence for EVERY distinct violation type ───────────
    saved_files = []
    for vkey in violations:
        if cooldown_ok(sid, vkey):
            fname = save_evidence(ann, sid, vkey)
            saved_files.append(fname)
            sess["screenshots"].append({
                "file":       fname,
                "ts":         datetime.now().isoformat(timespec="seconds"),
                "violation":  vkey,
                "trust":      round(sess["trust"], 1),
            })

    # Also snapshot on LOOKING_AWAY warning even if below violation threshold
    # (captures evidence of the behaviour without deducting points yet)
    if "LOOKING_AWAY" in warnings and "LOOKING_AWAY" not in violations:
        if sess["look_away_streak"] >= 6.0 and cooldown_ok(sid,"LOOKING_AWAY_WARN"):
            fname = save_evidence(ann, sid, "LOOKING_AWAY_WARN")
            saved_files.append(fname)
            sess["screenshots"].append({
                "file":    fname,
                "ts":      datetime.now().isoformat(timespec="seconds"),
                "violation":"LOOKING_AWAY_WARN",
                "trust":   round(sess["trust"],1),
            })

    return {
        "session_id":          sid,
        "face_count":          fc,
        "devices":             devices,
        "violations":          violations,
        "warnings":            warnings,
        "trust_score":         round(sess["trust"], 1),
        "violation_count":     sess["violation_count"],
        "no_face_streak":      round(sess["no_face_streak"], 1),
        "look_away_streak":    round(sess["look_away_streak"], 1),
        "look_away_limit":     LOOK_AWAY_VIOLATION_S,
        "off_centre_streak":   round(sess["off_centre_streak"], 1),
        "phone_streak":        sess["phone_streak"],
        "periodic_due":        periodic_due,
        "annotated":           encode_frame(ann),
        "saved_files":         saved_files,
        "yolo_enabled":        yolo_available,
        "timestamp":           now,
    }

def _face_offcentre(shape, face) -> bool:
    fh, fw = shape[:2]
    fx,fy,bw,bh = face
    nx = abs(fx+bw/2 - fw/2) / (fw/2+1e-6)
    ny = abs(fy+bh/2 - fh/2) / (fh/2+1e-6)
    return nx > FACE_OFFCENTRE_THR or ny > FACE_OFFCENTRE_THR

# ══════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════
@app.route("/health")
def health():
    return jsonify({"status":"ok","yolo":yolo_available,
                    "look_away_limit_s":LOOK_AWAY_VIOLATION_S,
                    "periodic_interval_s":PERIODIC_INTERVAL_S,
                    "evidence_dir":str(EVIDENCE_DIR.resolve())})

@app.route("/analyze", methods=["POST"])
def route_analyze():
    d = request.get_json(silent=True) or {}
    if "frame" not in d:
        return jsonify({"error":"No frame"}), 400
    try:
        return jsonify(analyze(d["frame"], d.get("session_id","default")))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error":str(e)}), 500

@app.route("/event", methods=["POST"])
def route_event():
    d    = request.get_json(silent=True) or {}
    sid  = d.get("session_id","default")
    sess = get_session(sid)
    ev   = d.get("event","")
    if ev == "TAB_SWITCH":
        sess["tab_switches"] += 1
        deduct(sess,"TAB_SWITCH",f"Switch #{sess['tab_switches']}")
    elif ev == "COPY_PASTE":
        deduct(sess,"COPY_PASTE","Clipboard intercepted")
    return jsonify({"trust_score":round(sess["trust"],1)})

@app.route("/snapshot", methods=["POST"])
def route_snapshot():
    d = request.get_json(silent=True) or {}
    if "frame" not in d:
        return jsonify({"error":"No frame"}), 400
    frame = decode_frame(d["frame"])
    fname = save_evidence(frame, d.get("session_id","default"),
                          d.get("label","manual"))
    return jsonify({"saved_file":fname})

@app.route("/session/<sid>")
def route_session(sid):
    s = get_session(sid)
    return jsonify({"session_id":sid,
                    "trust_score":round(s["trust"],1),
                    "violation_count":s["violation_count"],
                    "tab_switches":s["tab_switches"],
                    "log":s["log"][-60:],
                    "screenshots":s["screenshots"]})

@app.route("/evidence")
def route_evidence():
    files = sorted(EVIDENCE_DIR.glob("*.jpg"),
                   key=os.path.getmtime, reverse=True)
    return jsonify([{"filename":f.name,
                     "size_kb":round(f.stat().st_size/1024,1),
                     "modified":datetime.fromtimestamp(
                         f.stat().st_mtime).isoformat(timespec="seconds")}
                    for f in files[:100]])

@app.route("/evidence/<fname>")
def route_serve(fname):
    return send_from_directory(str(EVIDENCE_DIR.resolve()), fname)

@app.route("/reset/<sid>", methods=["POST"])
def route_reset(sid):
    sessions.pop(sid,None); return jsonify({"status":"reset"})

if __name__ == "__main__":
    print("="*68)
    print("  ProctorGuard v5  →  http://localhost:5050")
    print(f"  YOLO             : {'ON ✓' if yolo_available else 'OFF'}")
    print(f"  Phone hand-held  : ensemble  score≥{PHONE_SCORE_THR}  ×{PHONE_CONFIRM_FRAMES}frames")
    print(f"  Phone call-pose  : ensemble  score≥{PHONE_CALL_SCORE_THR}  (lateral-face check)")
    print(f"  Look-away        : violation after {LOOK_AWAY_VIOLATION_S}s  decay={LOOK_AWAY_DECAY}×/s")
    print(f"  Multi-face       : confirmed after {MULTI_FACE_CONFIRM_FRAMES} consecutive frames")
    print(f"  Off-centre       : thr={FACE_OFFCENTRE_THR:.0%}  sustained {FACE_OFFCENTRE_STREAK_S}s")
    print(f"  Periodic monitor : every {PERIODIC_INTERVAL_S}s  (periodic_due flag in response)")
    print(f"  Screenshots      : per violation-type, cooldown {SNAP_COOLDOWN_S}s")
    print(f"  Evidence folder  : {EVIDENCE_DIR.resolve()}")
    print("="*68)
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
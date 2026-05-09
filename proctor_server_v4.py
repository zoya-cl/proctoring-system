"""
ProctorGuard v7 — Reliable Core Detectors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT CAUSES FIXED:
  ✓ Face not detected  — minNeighbors was 8 (too strict), eye-gate was
                         rejecting valid faces with glasses/tilt/low light.
                         v7 uses 3-cascade ensemble at moderate sensitivity,
                         profile cascade for tilted heads, NO mandatory eye
                         gate for the PRIMARY (largest) face.
  ✓ Phone-to-ear miss  — scan strip was too tight and exclusions too broad.
                         v7 scans a WIDE lateral band, uses HSV dark-object
                         + edge-gradient signals, and does NOT exclude boxes
                         that touch the face perimeter.
  ✓ DOWN gaze          — eye Y-position check retained but threshold tuned.
  ✓ Multi-face FP      — eye gate applied ONLY to secondary smaller faces,
                         never to the primary (largest) detected face.

Install:
  pip install flask flask-cors opencv-python numpy
  pip install ultralytics   ← optional
"""

import cv2
import numpy as np
import base64
import os
import time
from collections import deque, Counter
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
#  THRESHOLDS
# ══════════════════════════════════════════════════════════════════
SNAP_COOLDOWN_S           = 8.0
PERIODIC_INTERVAL_S       = 5.0
LOOK_AWAY_VIOLATION_S     = 12.0
LOOK_AWAY_DECAY           = 2.0      # streak drop per sec when looking forward
NO_FACE_WARNING_S         = 4.0
NO_FACE_DEDUCT_S          = 30.0
FACE_OFFCENTRE_THR        = 0.46
FACE_OFFCENTRE_STREAK_S   = 5.0
MULTI_FACE_CONFIRM_FRAMES = 4
PHONE_CONFIRM_FRAMES      = 2
PHONE_SCORE_THR           = 0.48
PHONE_CALL_THR            = 0.42    # lower — call pose is harder to score
POSE_WINDOW               = 7
POSE_AWAY_VOTES           = 4       # of POSE_WINDOW votes must say AWAY
EYE_DOWN_Y_MAX            = 0.35    # eye centre above this (norm) → looking down

WEIGHTS = {
    "PHONE_DETECTED":    30,
    "MULTIPLE_FACES":    25,
    "TAB_SWITCH":        15,
    "NO_FACE_SUSTAINED": 10,
    "LOOKING_AWAY":       5,
    "FACE_OUT_OF_FRAME":  5,
    "COPY_PASTE":         8,
}

# ─── cooldown ─────────────────────────────────────────────────────
_snap_ts: dict = {}
def cooldown_ok(sid, key):
    k=(sid,key); now=time.time()
    if now-_snap_ts.get(k,0)>=SNAP_COOLDOWN_S:
        _snap_ts[k]=now; return True
    return False

# ─── cascades ─────────────────────────────────────────────────────
_HD          = cv2.data.haarcascades
_face_front  = cv2.CascadeClassifier(_HD+"haarcascade_frontalface_default.xml")
_face_alt    = cv2.CascadeClassifier(_HD+"haarcascade_frontalface_alt.xml")
_face_alt2   = cv2.CascadeClassifier(_HD+"haarcascade_frontalface_alt2.xml")
_face_prof   = cv2.CascadeClassifier(_HD+"haarcascade_profileface.xml")
_eye_plain   = cv2.CascadeClassifier(_HD+"haarcascade_eye.xml")
_eye_glass   = cv2.CascadeClassifier(_HD+"haarcascade_eye_tree_eyeglasses.xml")

# ─── YOLO ─────────────────────────────────────────────────────────
yolo_model=None; yolo_available=False
try:
    from ultralytics import YOLO
    for pt in ("yolov8s.pt","yolov8n.pt"):
        try:
            yolo_model=YOLO(pt); yolo_available=True
            print(f"[INFO] YOLO loaded: {pt}"); break
        except Exception: pass
    if not yolo_available: print("[WARN] YOLO weights missing")
except ImportError: print("[INFO] ultralytics absent")

DEVICE_CLS={67:"cell phone",63:"laptop",65:"remote"}

# ══════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════
def decode_frame(b64):
    raw=base64.b64decode(b64.split(",")[-1])
    return cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)

def encode_frame(frame,q=82):
    _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,q])
    return "data:image/jpeg;base64,"+base64.b64encode(buf).decode()

def save_evidence(frame,sid,tag):
    ts=datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fname=f"{sid}_{ts}_{tag[:28]}.jpg"
    cv2.imwrite(str(EVIDENCE_DIR/fname),frame,[cv2.IMWRITE_JPEG_QUALITY,90])
    return fname

def nms(boxes, thr=0.40):
    if not boxes: return []
    b=np.array(boxes,dtype=float)
    x1,y1=b[:,0],b[:,1]; x2,y2=b[:,0]+b[:,2],b[:,1]+b[:,3]
    areas=(x2-x1)*(y2-y1); order=np.argsort(areas)[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i); rest=order[1:]
        ix1=np.maximum(x1[i],x1[rest]); iy1=np.maximum(y1[i],y1[rest])
        ix2=np.minimum(x2[i],x2[rest]); iy2=np.minimum(y2[i],y2[rest])
        iw=np.maximum(0.,ix2-ix1); ih=np.maximum(0.,iy2-iy1)
        order=rest[(iw*ih)/(areas[rest]+1e-6)<thr]
    return [tuple(b[i].astype(int)) for i in keep]

def iou(a,b):
    ax2,ay2=a[0]+a[2],a[1]+a[3]; bx2,by2=b[0]+b[2],b[1]+b[3]
    ix1=max(a[0],b[0]);iy1=max(a[1],b[1]);ix2=min(ax2,bx2);iy2=min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.
    inter=(ix2-ix1)*(iy2-iy1)
    return inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-6)

def prep_gray(frame):
    """CLAHE-enhanced grayscale — better for dark/uneven lighting."""
    lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
    l,a,b_ch=cv2.split(lab)
    l=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(l)
    bgr=cv2.cvtColor(cv2.merge([l,a,b_ch]),cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)

# ══════════════════════════════════════════════════════════════════
#  FACE DETECTION  — 4-cascade ensemble, lenient, no mandatory eyes
#
#  Key design decision:
#    • minNeighbors = 4  (was 8 in v5/v6 — caused missed detections)
#    • Profile cascade included  (catches tilted / looking-down heads)
#    • Eye gate only on SECONDARY faces (not primary/largest)
#    • Size sanity: secondary face must be ≥40% area of primary
# ══════════════════════════════════════════════════════════════════
def detect_faces(frame):
    gray=prep_gray(frame)
    h,w=frame.shape[:2]
    minsz=max(55,min(h,w)//9)   # smaller minimum → catches further-away faces
    raw=[]

    # Front-facing cascades — moderate sensitivity (minNeighbors=4 not 8)
    for casc,sc,nb in [
        (_face_front, 1.06, 4),
        (_face_alt,   1.06, 4),
        (_face_alt2,  1.06, 4),
    ]:
        det=casc.detectMultiScale(gray,sc,nb,
                                  minSize=(minsz,minsz),
                                  maxSize=(int(w*.92),int(h*.92)))
        if len(det): raw.extend(det.tolist())

    # Profile cascade — essential for looking-down / sideways faces
    det=_face_prof.detectMultiScale(gray,1.06,4,
                                    minSize=(minsz,minsz),
                                    maxSize=(int(w*.80),int(h*.80)))
    if len(det): raw.extend(det.tolist())

    if not raw: return []

    # NMS to merge duplicates
    faces=nms(raw, thr=0.35)

    # Shape filter — faces are roughly square
    faces=[f for f in faces if 0.50<f[2]/max(f[3],1)<2.00]

    if not faces: return []

    # Sort by area descending — largest = primary face
    faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)

    # ── Eye gate for SECONDARY faces only ─────────────────────────
    # Primary face (index 0) is always kept.
    # Secondary faces must have at least 1 detectable eye.
    confirmed=[faces[0]]
    for f in faces[1:]:
        fx,fy,fw,fh=f
        if fw<70:                   # too small to check
            confirmed.append(f); continue
        roi=gray[fy:fy+fh,fx:fx+fw]
        eyes=[]
        for casc in [_eye_plain,_eye_glass]:
            det=casc.detectMultiScale(roi,1.1,3,minSize=(fw//8,fh//9))
            if len(det): eyes.extend(det.tolist())
        if eyes:
            confirmed.append(f)    # secondary face has visible eyes → real

    # Size sanity: secondary face ≥ 40% area of primary
    if len(confirmed)>1:
        maxA=confirmed[0][2]*confirmed[0][3]
        confirmed=[confirmed[0]]+[
            f for f in confirmed[1:] if f[2]*f[3]>=maxA*0.40]

    return confirmed

# ══════════════════════════════════════════════════════════════════
#  HEAD POSE — SIDEWAYS + DOWN via rolling vote
#
#  DOWN detection: measure mean Y of eye centres inside face ROI.
#  When head tilts DOWN, eyes shift toward the TOP of the face bbox
#  (because the forehead rotates away from camera).
# ══════════════════════════════════════════════════════════════════
def _eyes_in_roi(gray,fx,fy,fw,fh):
    """Return list of eye rects in ROI-local coords."""
    roi=gray[fy:fy+fh,fx:fx+fw]
    eyes=[]
    min_e=(max(10,fw//8),max(8,fh//10))
    for casc in [_eye_plain,_eye_glass]:
        det=casc.detectMultiScale(roi,1.1,3,minSize=min_e)
        if len(det): eyes.extend(det.tolist())
    # Keep only top 65% of face ROI
    eyes=nms(eyes,thr=0.40)
    eyes=[e for e in eyes if (e[1]+e[3]/2)/fh<0.65]
    return eyes

def _raw_pose(gray,fx,fy,fw,fh,img_w):
    """Single-frame pose: FORWARD | SIDEWAYS | DOWN"""

    # ── Profile cascade check ─────────────────────────────────────
    mp=(max(35,fw//2),max(35,fh//2))
    roi_gray=gray[max(0,fy-10):fy+fh+10, max(0,fx-10):fx+fw+10]
    for flipped in [False,True]:
        g=cv2.flip(roi_gray,1) if flipped else roi_gray
        dets=_face_prof.detectMultiScale(g,1.1,3,minSize=mp)
        if len(dets):
            return "SIDEWAYS"

    # ── Aspect-ratio narrowing ────────────────────────────────────
    if fw/max(fh,1)<0.54:
        return "SIDEWAYS"

    # ── Strong lateral drift ──────────────────────────────────────
    if abs(fx+fw/2-img_w/2)/(img_w/2+1e-6)>0.56:
        return "SIDEWAYS"

    # ── Eye Y-position for DOWN gaze ─────────────────────────────
    if fw>70:
        eyes=_eyes_in_roi(gray,fx,fy,fw,fh)
        if eyes:
            avg_eye_y=np.mean([(e[1]+e[3]/2)/fh for e in eyes])
            if avg_eye_y<EYE_DOWN_Y_MAX:
                return "DOWN"
            return "FORWARD"   # eyes found, Y normal
        else:
            # No eyes on a large face → likely turned sideways
            return "SIDEWAYS"

    return "FORWARD"

def head_pose(gray,face,img_w,vote_buf):
    fx,fy,fw,fh=face
    raw=_raw_pose(gray,fx,fy,fw,fh,img_w)
    vote_buf.append(raw)
    away=sum(1 for v in vote_buf if v!="FORWARD")
    if away>=POSE_AWAY_VOTES:
        c=Counter(v for v in vote_buf if v!="FORWARD")
        return c.most_common(1)[0][0]
    return "FORWARD"

# ══════════════════════════════════════════════════════════════════
#  PHONE DETECTION
#
#  Two independent detectors run in parallel:
#
#  A) call_pose_detector  — searches a WIDE band beside each face.
#     Does NOT exclude face-overlapping boxes. Looks for tall dark
#     rectangle laterally adjacent to the ear region.
#
#  B) handheld_detector   — searches the whole frame for phone-shaped
#     rectangles with skin exclusion. Handles phone in hand.
#
#  Scoring uses: edge density, HSV dark-channel, border contrast,
#  solidity, position.
# ══════════════════════════════════════════════════════════════════
def _phone_score(gray_roi, bh):
    """Return 0-1 score for how phone-like a ROI is."""
    if gray_roi.size==0: return 0.
    score=0.
    edges=cv2.Canny(gray_roi,30,100)
    ed=np.count_nonzero(edges)/gray_roi.size
    if 0.03<ed<0.45: score+=0.26
    elif ed>0.45:    score-=0.08

    mb=np.mean(gray_roi); sb=np.std(gray_roi)
    if mb<100:   score+=0.24    # dark phone body
    elif mb>185: score+=0.20    # lit screen
    if sb<65:    score+=0.20    # uniform surface

    # Border contrast (dark bezel + bright screen)
    pad=max(3,min(gray_roi.shape[0],gray_roi.shape[1])//10)
    inner=gray_roi[pad:-pad,pad:-pad] if (gray_roi.shape[0]>2*pad and
                                           gray_roi.shape[1]>2*pad) else gray_roi
    if inner.size>0:
        border_px=gray_roi.size-inner.size
        bord_mean=(np.sum(gray_roi.astype(float))-np.sum(inner.astype(float)))/(border_px+1e-6)
        contrast=abs(float(np.mean(inner))-bord_mean)
        if contrast>20: score+=0.18

    return min(score,1.0)

def _skin_frac(frame,bx,by,bw,bh):
    roi=cv2.cvtColor(frame[by:by+bh,bx:bx+bw],cv2.COLOR_BGR2YCrCb)
    mask=cv2.inRange(roi,(0,133,77),(255,173,127))
    return np.count_nonzero(mask)/(bw*bh+1e-6)

# ── A) Call-pose: wide lateral scan around face ───────────────────
def detect_call_pose(frame,gray,faces):
    """
    Scans a wide band to the left and right of each face.
    Accepts boxes that TOUCH or slightly overlap the face edge
    (phone-to-ear always touches the face boundary).
    """
    h,w=frame.shape[:2]
    results=[]

    for (fx,fy,fw,fh) in faces:
        # Wide search band: 1.6× face width on each side, full face height + 30%
        pad_x=int(fw*1.6); pad_y=int(fh*0.30)
        sx1=max(0,fx-pad_x); sy1=max(0,fy-pad_y)
        sx2=min(w,fx+fw+pad_x); sy2=min(h,fy+fh+pad_y)
        if sx2<=sx1 or sy2<=sy1: continue

        strip_g=gray[sy1:sy2,sx1:sx2]
        # Edge map on strip
        blur=cv2.GaussianBlur(strip_g,(5,5),0)
        edges=cv2.dilate(cv2.Canny(blur,15,60),
                         np.ones((3,3),np.uint8),iterations=2)
        cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            area=cv2.contourArea(cnt)
            sw=sx2-sx1; sh=sy2-sy1
            if area<1200 or area>sw*sh*0.65: continue

            bx,by,bw,bh=cv2.boundingRect(cnt)
            if bw<20 or bh<30: continue

            long_s=max(bw,bh); short_s=min(bw,bh)
            if short_s<1: continue
            aspect=long_s/short_s
            # Phone portrait 1.4–4.5 (wider range for partially occluded phone)
            if not(1.40<=aspect<=4.50): continue

            # Convert to full-frame coords
            abx=sx1+bx; aby=sy1+by

            # Must be BESIDE or TOUCHING the face, not directly in front
            # Acceptable: left side or right side of face bbox
            face_cx=fx+fw/2; box_cx=abx+bw/2
            lateral_dist=abs(box_cx-face_cx)
            # Box centre must be at least 30% of face width away from face centre
            if lateral_dist<fw*0.25: continue

            # Vertical overlap with face (phone covers ear→mouth height)
            vert_top=max(aby,fy); vert_bot=min(aby+bh,fy+fh)
            vert_overlap=(vert_bot-vert_top)/fh if vert_bot>vert_top else 0
            if vert_overlap<0.20: continue   # must overlap face vertically

            # Score the candidate
            roi_g=gray[aby:aby+bh,abx:abx+bw]
            score=_phone_score(roi_g,bh)

            # Position bonus: upper half of frame (ear-level)
            if aby<h*0.60: score+=0.15
            # Lateral adjacency bonus
            gap=max(0,lateral_dist-fw/2)
            if gap<fw*0.5: score+=0.18

            if score>=PHONE_CALL_THR:
                results.append({
                    "label":"cell phone","conf":round(score,2),
                    "bbox":[abx,aby,bw,bh],"pose":"call"
                })

    if not results: return []
    best=nms([r["bbox"] for r in results],thr=0.45)
    return [next((r for r in results if r["bbox"]==list(best[0])),results[0])] if best else []

# ── B) Hand-held: whole-frame scan with skin exclusion ───────────
def detect_handheld(frame,gray,faces):
    h,w=frame.shape[:2]
    blur=cv2.GaussianBlur(gray,(5,5),0)
    edges=cv2.dilate(cv2.Canny(blur,22,85),np.ones((3,3),np.uint8),iterations=1)
    cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cands=[]

    for cnt in cnts:
        area=cv2.contourArea(cnt)
        if area<2200 or area>w*h*0.40: continue
        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.024*peri,True)
        if not(4<=len(approx)<=8): continue
        bx,by,bw,bh=cv2.boundingRect(cnt)
        if bw<24 or bh<34: continue
        long_s=max(bw,bh); short_s=min(bw,bh)
        if short_s<1: continue
        aspect=long_s/short_s
        if not(1.55<=aspect<=2.70): continue

        hull_a=cv2.contourArea(cv2.convexHull(cnt))
        if area/max(hull_a,1)<0.52: continue

        # Skin exclusion — skip if mostly skin
        if _skin_frac(frame,bx,by,bw,bh)>0.50: continue
        # Skip if heavily overlaps a face
        if any(iou((bx,by,bw,bh),tuple(f))>0.30 for f in faces): continue

        roi_g=gray[by:by+bh,bx:bx+bw]
        score=_phone_score(roi_g,bh)
        if by>h*0.15: score+=0.15
        if area/(bw*bh)>0.70: score+=0.10

        if score>=PHONE_SCORE_THR:
            cands.append({"label":"cell phone","conf":round(score,2),
                          "bbox":[bx,by,bw,bh],"pose":"hand"})

    if not cands: return []
    best=nms([c["bbox"] for c in cands],thr=0.50)
    return [next((c for c in cands if c["bbox"]==list(best[0])),cands[0])] if best else []

def detect_devices(frame,faces):
    gray=prep_gray(frame)
    found=[]

    # YOLO (optional)
    if yolo_available and yolo_model:
        ih,iw=frame.shape[:2]; sc=960/iw if iw<960 else 1.
        rf=cv2.resize(frame,(int(iw*sc),int(ih*sc))) if sc!=1. else frame
        res=yolo_model(rf,verbose=False,conf=0.28,iou=0.45)[0]
        for box in res.boxes:
            cid=int(box.cls[0])
            if cid not in DEVICE_CLS: continue
            x1,y1,x2,y2=map(int,box.xyxy[0])
            bbox=[int(x1/sc),int(y1/sc),int((x2-x1)/sc),int((y2-y1)/sc)]
            bx2,by2,bw2,bh2=bbox
            if _skin_frac(frame,bx2,by2,bw2,bh2)>0.52: continue
            found.append({"label":DEVICE_CLS[cid],
                          "conf":round(float(box.conf[0]),2),
                          "bbox":bbox,"pose":"yolo"})

    # Call-pose (highest priority, runs regardless of YOLO)
    for e in detect_call_pose(frame,gray,faces):
        if not any(iou(tuple(e["bbox"]),tuple(f["bbox"]))>0.45 for f in found):
            found.append(e)

    # Hand-held
    for e in detect_handheld(frame,gray,faces):
        if not any(iou(tuple(e["bbox"]),tuple(f["bbox"]))>0.45 for f in found):
            found.append(e)

    return found

# ══════════════════════════════════════════════════════════════════
#  ANNOTATE
# ══════════════════════════════════════════════════════════════════
def annotate(frame,faces,devices,violations,warnings,trust,look_streak):
    out=frame.copy(); fh,fw=out.shape[:2]

    for (x,y,w,h) in faces:
        col=(30,210,30) if len(faces)==1 else (0,40,230)
        cv2.rectangle(out,(x,y),(x+w,y+h),col,2)
        lbl="FACE" if len(faces)==1 else f"FACE×{len(faces)}"
        cv2.putText(out,lbl,(x,max(y-6,14)),cv2.FONT_HERSHEY_SIMPLEX,.55,col,2)

    for d in devices:
        bx,by,bw,bh=d["bbox"]
        cv2.rectangle(out,(bx,by),(bx+bw,by+bh),(0,40,230),2)
        cv2.putText(out,f"{d['label']} [{d['pose']}] {d['conf']:.0%}",
                    (bx,max(by-6,14)),cv2.FONT_HERSHEY_SIMPLEX,.50,(0,40,230),2)

    # Look-away progress bar (top-right)
    if look_streak>0:
        bw2=int(fw*0.28); prog=min(int(bw2*look_streak/LOOK_AWAY_VIOLATION_S),bw2)
        cv2.rectangle(out,(fw-bw2-8,8),(fw-8,24),(40,40,40),-1)
        col2=(0,180,255) if look_streak<LOOK_AWAY_VIOLATION_S else (0,40,230)
        cv2.rectangle(out,(fw-bw2-8,8),(fw-bw2-8+prog,24),col2,-1)
        cv2.putText(out,f"Away {look_streak:.0f}/{LOOK_AWAY_VIOLATION_S:.0f}s",
                    (fw-bw2-8,38),cv2.FONT_HERSHEY_SIMPLEX,.38,(220,220,220),1)

    # Trust bar (top-left)
    bar=int(fw*0.22)
    col=(30,200,30) if trust>=60 else (0,165,255) if trust>=40 else (0,40,230)
    cv2.rectangle(out,(10,8),(10+bar,26),(30,30,30),-1)
    cv2.rectangle(out,(10,8),(10+int(bar*max(0,trust)/100),26),col,-1)
    cv2.putText(out,f"Trust {trust:.0f}",(14,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)

    # Status (bottom)
    if violations:
        msg="⚡ VIOLATION: "+violations[0].replace("_"," "); col=(0,40,230)
    elif warnings:
        msg="⚡ WARNING: "+warnings[0].replace("_"," ");    col=(0,140,255)
    else:
        msg="✓ CLEAR"; col=(30,200,30)
    cv2.rectangle(out,(0,fh-32),(fw,fh),(0,0,0),-1)
    cv2.putText(out,msg,(8,fh-10),cv2.FONT_HERSHEY_SIMPLEX,.62,col,2)
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
def analyze(b64,sid):
    sess=get_session(sid)
    frame=decode_frame(b64)
    gray=prep_gray(frame)
    ih,iw=frame.shape[:2]

    now=time.time(); dt=min(now-sess["last_ts"],3.0); sess["last_ts"]=now
    periodic_due=(now-sess["last_periodic_ts"])>=PERIODIC_INTERVAL_S
    if periodic_due: sess["last_periodic_ts"]=now

    faces=detect_faces(frame)
    fc=len(faces)
    devices=detect_devices(frame,faces)

    violations=[]; warnings=[]

    # ── No face ──────────────────────────────────────────────────
    if fc==0:
        sess["no_face_streak"]+=dt; warnings.append("NO_FACE")
        if sess["no_face_streak"]>=NO_FACE_WARNING_S:
            warnings.append("NO_FACE_SUSTAINED")
        sess["no_face_accum"]+=dt
        if sess["no_face_accum"]>=NO_FACE_DEDUCT_S:
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

    # ── Gaze + centering ─────────────────────────────────────────
    if fc>=1:
        gaze=head_pose(gray,faces[0],iw,sess["pose_votes"])
        if gaze!="FORWARD":
            sess["look_away_streak"]+=dt
            warnings.append("LOOKING_AWAY")
            if gaze=="DOWN": warnings.append("LOOKING_DOWN")
            if sess["look_away_streak"]>=LOOK_AWAY_VIOLATION_S:
                violations.append("LOOKING_AWAY")
                deduct(sess,"LOOKING_AWAY",
                       f"{gaze} {sess['look_away_streak']:.1f}s")
                sess["look_away_streak"]=LOOK_AWAY_VIOLATION_S*0.35
        else:
            sess["look_away_streak"]=max(0.,
                sess["look_away_streak"]-dt*LOOK_AWAY_DECAY)

        fx2,fy2,fw2,fh2=faces[0]
        nx=abs(fx2+fw2/2-iw/2)/(iw/2+1e-6)
        ny=abs(fy2+fh2/2-ih/2)/(ih/2+1e-6)
        if nx>FACE_OFFCENTRE_THR or ny>FACE_OFFCENTRE_THR:
            sess["off_centre_streak"]+=dt; warnings.append("FACE_OUT_OF_FRAME")
            if sess["off_centre_streak"]>=FACE_OFFCENTRE_STREAK_S:
                violations.append("FACE_OUT_OF_FRAME")
                deduct(sess,"FACE_OUT_OF_FRAME",
                       f"Off-centre {sess['off_centre_streak']:.1f}s")
                sess["off_centre_streak"]=0.
        else:
            sess["off_centre_streak"]=max(0.,sess["off_centre_streak"]-dt*2.)

    # ── Phone ────────────────────────────────────────────────────
    if devices:
        sess["phone_streak"]+=1; warnings.append("PHONE_DETECTED")
        if sess["phone_streak"]>=PHONE_CONFIRM_FRAMES:
            violations.append("PHONE_DETECTED")
            poses="+".join(set(d.get("pose","?") for d in devices))
            deduct(sess,"PHONE_DETECTED",
                   f"{', '.join(set(d['label'] for d in devices))} [{poses}]")
    else:
        sess["phone_streak"]=0

    # ── Annotate & save ──────────────────────────────────────────
    ann=annotate(frame,faces,devices,violations,warnings,
                 sess["trust"],sess["look_away_streak"])

    saved_files=[]
    for vkey in violations:
        if cooldown_ok(sid,vkey):
            fn=save_evidence(ann,sid,vkey)
            saved_files.append(fn)
            sess["screenshots"].append({
                "file":fn,"ts":datetime.now().isoformat(timespec="seconds"),
                "violation":vkey,"trust":round(sess["trust"],1),
            })

    # Intermediate look-away screenshot (evidence before violation fires)
    if ("LOOKING_AWAY" in warnings and "LOOKING_AWAY" not in violations
            and sess["look_away_streak"]>=5.0 and cooldown_ok(sid,"LOOK_WARN")):
        fn=save_evidence(ann,sid,"LOOK_WARN")
        saved_files.append(fn)
        sess["screenshots"].append({
            "file":fn,"ts":datetime.now().isoformat(timespec="seconds"),
            "violation":"LOOK_WARN","trust":round(sess["trust"],1),
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
                    "look_away_s":LOOK_AWAY_VIOLATION_S,
                    "periodic_s":PERIODIC_INTERVAL_S})

@app.route("/analyze",methods=["POST"])
def route_analyze():
    d=request.get_json(silent=True) or {}
    if "frame" not in d: return jsonify({"error":"No frame"}),400
    try:
        return jsonify(analyze(d["frame"],d.get("session_id","default")))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error":str(e)}),500

@app.route("/event",methods=["POST"])
def route_event():
    d=request.get_json(silent=True) or {}
    sid=d.get("session_id","default"); sess=get_session(sid)
    ev=d.get("event","")
    if ev=="TAB_SWITCH":
        sess["tab_switches"]+=1
        deduct(sess,"TAB_SWITCH",f"Switch #{sess['tab_switches']}")
    elif ev=="COPY_PASTE":
        deduct(sess,"COPY_PASTE","Clipboard intercepted")
    return jsonify({"trust_score":round(sess["trust"],1)})

@app.route("/snapshot",methods=["POST"])
def route_snapshot():
    d=request.get_json(silent=True) or {}
    if "frame" not in d: return jsonify({"error":"No frame"}),400
    fn=save_evidence(decode_frame(d["frame"]),
                     d.get("session_id","default"),d.get("label","manual"))
    return jsonify({"saved_file":fn})

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
    return jsonify([{"filename":f.name,
                     "size_kb":round(f.stat().st_size/1024,1),
                     "modified":datetime.fromtimestamp(
                         f.stat().st_mtime).isoformat(timespec="seconds")}
                    for f in files[:100]])

@app.route("/evidence/<fname>")
def route_serve(fname):
    return send_from_directory(str(EVIDENCE_DIR.resolve()),fname)

@app.route("/reset/<sid>",methods=["POST"])
def route_reset(sid):
    sessions.pop(sid,None); return jsonify({"status":"reset"})

if __name__=="__main__":
    print("="*68)
    print("  ProctorGuard v7  →  http://localhost:5050")
    print(f"  YOLO            : {'ON' if yolo_available else 'OFF'}")
    print(f"  Face detect     : 4-cascade ensemble, minNeighbors=4")
    print(f"                    Profile cascade ON (catches looking-down faces)")
    print(f"                    Eye gate on secondary faces only")
    print(f"  Gaze: SIDEWAYS  : profile cascade + aspect ratio + lateral drift")
    print(f"  Gaze: DOWN      : eye Y-position < {EYE_DOWN_Y_MAX:.0%} of face height")
    print(f"  Gaze threshold  : {LOOK_AWAY_VIOLATION_S}s continuous → violation")
    print(f"  Phone call-pose : wide lateral band ({'+/-'} 1.6× face width), no exclusions")
    print(f"  Phone hand-held : whole-frame scan, score≥{PHONE_SCORE_THR}")
    print(f"  Multi-face guard: secondary eye-gate + size≥40% + {MULTI_FACE_CONFIRM_FRAMES}f confirm")
    print(f"  Periodic snap   : every {PERIODIC_INTERVAL_S}s")
    print(f"  Evidence folder : {EVIDENCE_DIR.resolve()}")
    print("="*68)
    app.run(host="0.0.0.0",port=5050,debug=False,threaded=True)
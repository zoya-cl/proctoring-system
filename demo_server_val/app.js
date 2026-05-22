import { FaceLandmarker, ObjectDetector, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';

const CONFIG = {
  face: { path: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
  object: { path: "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float16/1/ssd_mobilenet_v2.tflite" }
};

let faceRunner, objectRunner, active = false, startTime = null, violCount = 0;
const video = document.getElementById("videoEl");
const canvas = document.getElementById("overlayCanvas");
const bar = document.getElementById('progress-bar');
const banner = document.getElementById('statusBanner');
const startBtn = document.getElementById('startBtn');

async function init() {
  try {
    bar.style.width = '30%';
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
    bar.style.width = '60%';
    faceRunner = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: CONFIG.face.path },
      runningMode: "VIDEO", numFaces: 1
    });
    bar.style.width = '85%';
    objectRunner = await ObjectDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: CONFIG.object.path },
      runningMode: "VIDEO"
    });
    bar.style.width = '100%';
    banner.textContent = "✓ System Ready";
    banner.className = "ok";
    startBtn.disabled = false;
    console.log("Initialization successful");
  } catch (err) {
    console.error("Initialization error:", err);
    banner.textContent = "⚠ Initialization Failed: " + err.message;
    banner.className = "danger";
  }
}
init();

function runLoop() {
  if (!active) return;
  try {
    if (video.readyState >= 2) {
      const ts = performance.now();
      const faceRes = faceRunner.detectForVideo(video, ts);
      const objRes = objectRunner.detectForVideo(video, ts);
      document.getElementById("statTime").textContent = Math.floor((Date.now() - startTime) / 1000) + "s";
      updateUI(faceRes, objRes);
    }
  } catch (err) {
    console.error("Loop error:", err);
  }
  requestAnimationFrame(runLoop);
}

function updateCheck(id, state, sub) {
  const el = document.getElementById(id);
  if (el) { el.className = "check-item " + state; el.querySelector(".check-sub").textContent = sub; }
}

function updateUI(faceRes, objRes) {
  const ctx = canvas.getContext("2d");
  canvas.width = video.videoWidth; 
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  const faces = faceRes.faceLandmarks?.length || 0;
  const phones = objRes.detections.filter(d => d.categories[0].categoryName === 'cell phone' && d.categories[0].score > 0.3);
  
  // Draw Face
  ctx.strokeStyle = "#00e676"; ctx.lineWidth = 3;
  if (faceRes.faceLandmarks?.[0]) {
    const lms = faceRes.faceLandmarks[0];
    const xs = lms.map(l => l.x), ys = lms.map(l => l.y);
    ctx.strokeRect(Math.min(...xs)*canvas.width, Math.min(...ys)*canvas.height, (Math.max(...xs)-Math.min(...xs))*canvas.width, (Math.max(...ys)-Math.min(...ys))*canvas.height);
  }
  // Draw Phone
  ctx.strokeStyle = "#ff3d5a";
  phones.forEach(d => ctx.strokeRect(d.boundingBox.originX*(canvas.width/video.videoWidth), d.boundingBox.originY*(canvas.height/video.videoHeight), d.boundingBox.width*(canvas.width/video.videoWidth), d.boundingBox.height*(canvas.height/video.videoHeight)));
  
  // Stats & Violations
  document.getElementById("statFaces").textContent = faces;
  if (faces === 0 || faces > 1 || phones.length > 0) {
    violCount++;
    document.getElementById("statViol").textContent = violCount;
    const msg = faces === 0 ? "No face" : phones.length > 0 ? "Phone detected" : "Multiple faces";
    const meta = { faces, phones: phones.length, timestamp: new Date().toISOString() };
    const d = document.createElement("div");
    d.className = "log-entry danger";
    d.innerHTML = `<strong>${msg}</strong><br/><small>${new Date().toLocaleTimeString()}</small><img src="${canvas.toDataURL()}"/>`;
    d.onclick = () => {
      document.getElementById("mTitle").textContent = msg;
      document.getElementById("mImg").src = d.querySelector('img').src;
      document.getElementById("mMeta").textContent = JSON.stringify(meta, null, 2);
      document.getElementById("metaModal").style.display = "flex";
    };
    document.getElementById("violLog").prepend(d);
  }
  
  document.getElementById("statusBanner").textContent = faces === 0 ? "⚠ NO FACE" : phones.length > 0 ? "⚠ PHONE DETECTED" : "✓ CLEAR";
  document.getElementById("statusBanner").className = (faces === 0 || phones.length > 0) ? "danger" : "ok";
  updateCheck("ck-face", faces > 0 ? "ok" : "danger", faces > 0 ? "Face: Detected" : "Face: Missing");
  updateCheck("ck-phone", phones.length === 0 ? "ok" : "danger", phones.length > 0 ? "Phone: Detected!" : "Phone: Clear");
  updateCheck("ck-multi", faces === 1 ? "ok" : (faces > 1 ? "danger" : "warn"), faces === 1 ? "Person: Clear" : (faces > 1 ? "Person: Multiple!" : "Person: None"));
  updateCheck("ck-frame", faces > 0 ? "ok" : "danger", faces > 0 ? "In Frame: Centered" : "Missing");
  
  let gazeStatus = "warn", gazeText = "Gaze: Looking away";
  if (faces > 0) {
    const lms = faceRes.faceLandmarks[0];
    const nose = lms[1], left = lms[33], right = lms[263];
    if (Math.abs(nose.x - (left.x + right.x)/2) < 0.08) { gazeStatus = "ok"; gazeText = "Looking forward"; }
  }
  updateCheck("ck-gaze", gazeStatus, gazeText);
}

startBtn.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: true});
    video.srcObject = stream;
    await new Promise(res => video.onloadedmetadata = res);
    video.play();
    active = true;
    startTime = Date.now();
    runLoop();
    startBtn.disabled = true;
    document.getElementById('stopBtn').disabled = false;
    console.log("Monitoring started");
  } catch(e) { 
    console.error("Camera error:", e);
    alert("Camera error: " + e.message); 
  }
});
document.getElementById('stopBtn').addEventListener('click', () => location.reload());

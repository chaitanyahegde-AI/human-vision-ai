const mouthVal = document.getElementById('mouthVal');
const eyeVal = document.getElementById('eyeVal');
const confVal = document.getElementById('confVal');
const headTiltVal = document.getElementById('headTiltVal');
const headActionEl = document.getElementById('headAction');

function updateUIFromPreds(preds){
  const p = preds[0];

  // existing metric updates...
  mouthVal.textContent = p.mouth_open ? p.mouth_open.toFixed(3) : '—';
  eyeVal.textContent = p.eye_open ? p.eye_open.toFixed(3) : '—';
  confVal.textContent = (p.top_emotion.score*100).toFixed(0) + '%';

  // head tilt & head action (guarded)
  if (headTiltVal) {
    headTiltVal.textContent = (p.head_tilt_deg !== undefined && p.head_tilt_deg !== null) ? `${p.head_tilt_deg}°` : '—';
  }
  if (headActionEl) {
    headActionEl.textContent = p.head_action || '';
  }
  // ...existing code...
}

ws.onmessage = evt => {
  try {
    const data = JSON.parse(evt.data);
    console.log("WS message:", data);            // <--- add this line for debugging
    if (data.error) { status.textContent = 'Server: ' + data.error; return; }
    updateUIFromPreds(data.predictions || []);
    // ...existing code...
  } catch (e) {
    console.error('bad message', e);
  }
};
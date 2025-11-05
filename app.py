# app.py
import os
import sys
import json
import base64
import math
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import numpy as np
import cv2

# mediapipe optional
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=3,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
except Exception:
    face_mesh = None

app = FastAPI()

# mount static under /static and serve index at /
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    async def root():
        index_path = os.path.join("static", "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        return {"message": "Index not found in ./static"}
else:
    # no prints per request
    pass

def jpeg_bytes_to_bgr(jpeg_b64: str):
    if not jpeg_b64:
        return None
    try:
        if "," in jpeg_b64:
            jpeg_b64 = jpeg_b64.split(",", 1)[1]
        imgdata = base64.b64decode(jpeg_b64)
        arr = np.frombuffer(imgdata, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def _dist(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.linalg.norm(a - b))

def _head_tilt_deg_from_landmarks(lm):
    """Robust estimate of head roll (degrees) using several eye landmarks."""
    if not lm or len(lm) < 10:
        return 0.0
    left_idxs = [33, 133, 159, 145, 130]
    right_idxs = [362, 263, 386, 374, 359]
    left_pts = [lm[i] for i in left_idxs if i < len(lm)]
    right_pts = [lm[i] for i in right_idxs if i < len(lm)]
    if not left_pts or not right_pts:
        return 0.0
    left_c = np.mean(np.array(left_pts), axis=0)
    right_c = np.mean(np.array(right_pts), axis=0)
    dx = float(right_c[0] - left_c[0])
    dy = float(right_c[1] - left_c[1])
    if abs(dx) < 1e-6:
        return 0.0
    angle = math.degrees(math.atan2(dy, dx))
    return round(abs(angle), 2)

def analyze_frame(img_bgr):
    results = []
    if face_mesh is None or img_bgr is None:
        return results
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        mesh_results = face_mesh.process(img_rgb)
        if not mesh_results or not mesh_results.multi_face_landmarks:
            return results

        for face_landmarks in mesh_results.multi_face_landmarks:
            lm = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
            xs = [p[0] for p in lm]; ys = [p[1] for p in lm]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            bw = max(1, maxx - minx); bh = max(1, maxy - miny)
            x, y = minx, miny

            left_mouth = lm[61] if len(lm) > 61 else None
            right_mouth = lm[291] if len(lm) > 291 else None
            top_lip = lm[13] if len(lm) > 13 else None
            bottom_lip = lm[14] if len(lm) > 14 else None

            left_eye_top = lm[159] if len(lm) > 159 else None
            left_eye_bottom = lm[145] if len(lm) > 145 else None
            right_eye_top = lm[386] if len(lm) > 386 else None
            right_eye_bottom = lm[374] if len(lm) > 374 else None

            mouth_width = _dist(left_mouth, right_mouth) if left_mouth and right_mouth else None
            mouth_height = _dist(top_lip, bottom_lip) if top_lip and bottom_lip else None
            mouth_open_score = None; smile_score = None
            if mouth_width and mouth_height and mouth_height > 0:
                mouth_open_score = float(mouth_height / bh)
                smile_score = float(mouth_width / (mouth_height + 1e-6))

            eye_open_score = None
            eye_scores = []
            if left_eye_top and left_eye_bottom:
                eye_scores.append(_dist(left_eye_top, left_eye_bottom) / bh)
            if right_eye_top and right_eye_bottom:
                eye_scores.append(_dist(right_eye_top, right_eye_bottom) / bh)
            if eye_scores:
                eye_open_score = float(np.mean(eye_scores))

            mouth_corner_offset_norm = 0.0
            if left_mouth and right_mouth and top_lip and bottom_lip:
                corners_avg_y = (left_mouth[1] + right_mouth[1]) / 2.0
                center_lip_y = (top_lip[1] + bottom_lip[1]) / 2.0
                mouth_corner_offset_norm = float((corners_avg_y - center_lip_y) / (bh + 1e-6))

            head_tilt_deg = _head_tilt_deg_from_landmarks(lm)

            # simple heuristics
            label = "neutral"; conf = 0.5
            if mouth_open_score is not None and eye_open_score is not None and mouth_open_score > 0.35 and eye_open_score > 0.16:
                label = "shocked"; conf = min(0.99, 0.5 + mouth_open_score + eye_open_score)
            elif mouth_open_score is not None and eye_open_score is not None and mouth_open_score > 0.22 and eye_open_score > 0.10:
                label = "surprised"; conf = min(0.98, 0.5 + mouth_open_score + eye_open_score)
            elif eye_open_score is not None and mouth_open_score is not None and eye_open_score > 0.14 and mouth_open_score < 0.08 and (smile_score is None or smile_score < 2.5):
                label = "anxious"; conf = min(0.95, 0.55 + (eye_open_score - 0.14) * 3.0)
            elif smile_score is not None and smile_score > 3.0:
                label = "happy"; conf = min(0.95, 0.55 + (smile_score - 3.0) / 5.0)
            elif mouth_corner_offset_norm > 0.02 and (smile_score is None or smile_score < 2.2):
                label = "sad"; conf = min(0.92, 0.55 + mouth_corner_offset_norm * 3.0)
            elif eye_open_score is not None and eye_open_score < 0.06 and (mouth_open_score is None or mouth_open_score < 0.10):
                label = "angry"; conf = min(0.9, 0.55 + (0.08 - eye_open_score) * 6.0)
            else:
                label = "neutral"; conf = 0.5

            advice = ""
            if label == "sad":
                advice = "Offer empathy — a short encouraging line can help."
            elif label == "happy":
                advice = "Appreciate them — say something positive about their energy."
            elif label == "anxious":
                advice = "Suggest a short pause and deep breaths; be patient and supportive."
            elif label in ("surprised", "shocked"):
                advice = "Ask a clarifying question — they may need context."
            elif label == "angry":
                advice = "Use a calm tone and offer space to respond."

            head_action = None
            if head_tilt_deg > 18.0:
                head_action = "Person may be copying (head tilted)."

            results.append({
                "box": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                "top_emotion": {"label": label, "score": round(float(conf), 3)},
                "all_emotions": {
                    "happy": round(float(conf) if label == "happy" else 0.0, 3),
                    "surprised": round(float(conf) if label == "surprised" else 0.0, 3),
                    "shocked": round(float(conf) if label == "shocked" else 0.0, 3),
                    "anxious": round(float(conf) if label == "anxious" else 0.0, 3),
                    "sad": round(float(conf) if label == "sad" else 0.0, 3),
                    "angry": round(float(conf) if label == "angry" else 0.0, 3),
                    "neutral": round(float(conf) if label == "neutral" else 0.0, 3)
                },
                "mouth_open": mouth_open_score,
                "eye_open": eye_open_score,
                "mouth_corner_offset": mouth_corner_offset_norm,
                "head_tilt": round(head_tilt_deg, 2),           # name used by front-end
                "head_tilt_deg": round(head_tilt_deg, 2),       # legacy name
                "advice": advice,
                "head_action": head_action
            })
    except Exception:
        traceback.print_exc()
    return results

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                await websocket.send_text(json.dumps({"error": "invalid json", "predictions": []}))
                continue
            frame_b64 = data.get("frame")
            if not frame_b64:
                await websocket.send_text(json.dumps({"error": "no frame", "predictions": []}))
                continue
            frame = jpeg_bytes_to_bgr(frame_b64)
            if frame is None:
                await websocket.send_text(json.dumps({"error": "bad frame", "predictions": []}))
                continue
            preds = analyze_frame(frame)
            await websocket.send_text(json.dumps({"predictions": preds}))
    except WebSocketDisconnect:
        return
    except Exception:
        return

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
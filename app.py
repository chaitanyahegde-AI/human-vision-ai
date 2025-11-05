import os
import json
import base64
import math
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import numpy as np
import cv2

# optional MediaPipe; server still runs without it
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

# thresholds (tweak if needed)
HEAD_TILT_ALERT_THRESHOLD = 2.0
ANGRY_BROW_OFFSET_THRESH = -0.02
ANGRY_BROW_DIST_THRESH = 0.20
ANGRY_EYE_SQUINT_THRESH = 0.10
SAD_MOUTH_CORNER_THRESH = 0.01
FEAR_BROW_RAISE_THRESH = 0.03
FEAR_EYE_WIDE_THRESH = 0.18

app = FastAPI()
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    async def root():
        p = os.path.join("static", "index.html")
        if os.path.isfile(p):
            return FileResponse(p)
        return {"message": "Index not found"}
else:
    pass


def jpeg_bytes_to_bgr(jpeg_b64: str):
    if not jpeg_b64:
        return None
    try:
        if "," in jpeg_b64:
            jpeg_b64 = jpeg_b64.split(",", 1)[1]
        b = base64.b64decode(jpeg_b64)
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _dist(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.linalg.norm(a - b))


def _head_tilt_deg_from_landmarks(lm):
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


def _compute_eyebrow_metrics(lm, bw, bh):
    left_brow_idxs = [70, 63, 105, 66, 107]
    right_brow_idxs = [336, 296, 334, 293, 300]
    left_eye_idxs = [33, 133, 159, 145]
    right_eye_idxs = [362, 263, 386, 374]

    left_brow_pts = [lm[i] for i in left_brow_idxs if i < len(lm)]
    right_brow_pts = [lm[i] for i in right_brow_idxs if i < len(lm)]
    left_eye_pts = [lm[i] for i in left_eye_idxs if i < len(lm)]
    right_eye_pts = [lm[i] for i in right_eye_idxs if i < len(lm)]

    if not left_brow_pts or not right_brow_pts or not left_eye_pts or not right_eye_pts:
        return {"brow_dist_norm": None, "brow_vs_eye_offset": None}

    lb = np.mean(np.array(left_brow_pts), axis=0)
    rb = np.mean(np.array(right_brow_pts), axis=0)
    le = np.mean(np.array(left_eye_pts), axis=0)
    re = np.mean(np.array(right_eye_pts), axis=0)

    brow_dist = float(np.linalg.norm(rb - lb))
    brow_dist_norm = brow_dist / (bw + 1e-6)
    brow_y = (lb[1] + rb[1]) / 2.0
    eye_y = (le[1] + re[1]) / 2.0
    brow_vs_eye_offset = float((brow_y - eye_y) / (bh + 1e-6))
    return {"brow_dist_norm": round(brow_dist_norm, 4), "brow_vs_eye_offset": round(brow_vs_eye_offset, 4)}


def _compute_brow_side_metrics(lm):
    """Return (left_brow_y, right_brow_y) averaged from candidate indices or (None,None)."""
    left_idxs = [70, 63, 105, 66, 107]
    right_idxs = [336, 296, 334, 293, 300]
    left_pts = [lm[i] for i in left_idxs if i < len(lm)]
    right_pts = [lm[i] for i in right_idxs if i < len(lm)]
    if not left_pts or not right_pts:
        return None, None
    left_y = float(np.mean([p[1] for p in left_pts]))
    right_y = float(np.mean([p[1] for p in right_pts]))
    return left_y, right_y


def _mouth_asymmetry(lm, bw):
    """Normalized vertical difference between mouth corners (left_y - right_y) / bw."""
    if len(lm) <= 291:
        return None
    l = lm[61]; r = lm[291]
    return float((l[1] - r[1]) / (bw + 1e-6))


def _nose_wrinkle_score(lm, bh):
    """Rough proxy for nose wrinkle: vertical distance between nose root/top and nostril region."""
    idxs_top = [1, 2]    # approximate
    idxs_nost = [98, 327]
    top_pts = [lm[i] for i in idxs_top if i < len(lm)]
    nost_pts = [lm[i] for i in idxs_nost if i < len(lm)]
    if not top_pts or not nost_pts:
        return None
    top_y = float(np.mean([p[1] for p in top_pts]))
    nost_y = float(np.mean([p[1] for p in nost_pts]))
    return float((nost_y - top_y) / (bh + 1e-6))


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
            # defaults
            label = "neutral"; conf = 0.5
            try:
                lm = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
                if not lm or len(lm) < 10:
                    results.append({
                        "box": {"x": 0, "y": 0, "w": 0, "h": 0},
                        "top_emotion": {"label": label, "score": conf},
                        "all_emotions": {},
                        "mouth_open": None,
                        "eye_open": None,
                        "mouth_corner_offset": None,
                        "head_tilt_deg": 0.0,
                        "advice": "",
                        "head_action": None
                    })
                    continue

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
                brow_metrics = _compute_eyebrow_metrics(lm, bw, bh)
                brow_dist = brow_metrics.get("brow_dist_norm")
                brow_offset = brow_metrics.get("brow_vs_eye_offset")

                left_brow_y, right_brow_y = _compute_brow_side_metrics(lm)
                brow_side_diff = None
                if left_brow_y is not None and right_brow_y is not None:
                    brow_side_diff = float((left_brow_y - right_brow_y) / (bh + 1e-6))  # positive => left higher

                mouth_asym = _mouth_asymmetry(lm, bw)
                nose_wr = _nose_wrinkle_score(lm, bh)

                # classification priority
                if mouth_open_score is not None and eye_open_score is not None and mouth_open_score > 0.35 and eye_open_score > 0.16:
                    label = "shocked"; conf = min(0.99, 0.5 + mouth_open_score + eye_open_score)
                elif (brow_offset is not None and brow_offset > FEAR_BROW_RAISE_THRESH) and (eye_open_score is not None and eye_open_score > FEAR_EYE_WIDE_THRESH):
                    label = "fear"; conf = min(0.98, 0.5 + eye_open_score + (brow_offset * 5.0))
                elif mouth_open_score is not None and eye_open_score is not None and mouth_open_score > 0.22 and eye_open_score > 0.10:
                    label = "surprised"; conf = min(0.98, 0.5 + mouth_open_score + eye_open_score)
                elif eye_open_score is not None and mouth_open_score is not None and eye_open_score > 0.14 and mouth_open_score < 0.08 and (smile_score is None or smile_score < 2.5):
                    label = "anxious"; conf = min(0.95, 0.55 + (eye_open_score - 0.14) * 3.0)
                elif smile_score is not None and smile_score > 3.0:
                    label = "happy"; conf = min(0.95, 0.55 + (smile_score - 3.0) / 5.0)
                elif ((brow_offset is not None and brow_offset < ANGRY_BROW_OFFSET_THRESH) and
                      (brow_dist is not None and brow_dist < ANGRY_BROW_DIST_THRESH) and
                      (eye_open_score is not None and eye_open_score < ANGRY_EYE_SQUINT_THRESH)):
                    label = "angry"; conf = min(0.98, 0.55 + (abs(brow_offset) * 4.0))
                elif mouth_corner_offset_norm > SAD_MOUTH_CORNER_THRESH and (smile_score is None or smile_score < 2.2):
                    label = "sad"; conf = min(0.92, 0.55 + mouth_corner_offset_norm * 3.0)
                # add disorder-specific branches (placed with appropriate priority)
                # e.g. disgust: nose wrinkle or upper lip raise (nose_wr large)
                elif nose_wr is not None and nose_wr > 0.025:
                    label = "disgust"; conf = min(0.95, 0.5 + (nose_wr * 6.0))
                # contempt: asymmetric mouth corner (one corner higher/lifted)
                elif mouth_asym is not None and abs(mouth_asym) > 0.035 and (smile_score is None or smile_score < 3.0):
                    label = "contempt"; conf = min(0.94, 0.55 + abs(mouth_asym) * 4.0)
                # confused: one brow noticeably higher than the other + neutral mouth
                elif brow_side_diff is not None and abs(brow_side_diff) > 0.035 and (mouth_open_score is None or mouth_open_score < 0.12):
                    label = "confused"; conf = min(0.92, 0.55 + abs(brow_side_diff) * 3.0)
                # bored: droopy eyes + low mouth activity
                elif eye_open_score is not None and eye_open_score < 0.05 and (mouth_open_score is None or mouth_open_score < 0.06):
                    label = "bored"; conf = min(0.9, 0.55 + (0.06 - eye_open_score) * 6.0)
                else:
                    label = "neutral"; conf = 0.5

                advice = ""
                if label == "sad":
                    advice = "Offer empathy — a short encouraging line can help."
                elif label == "happy":
                    advice = "Appreciate them — say something positive about their energy."
                elif label == "anxious":
                    advice = "Suggest a short pause and deep breaths; be patient and supportive."
                elif label == "fear":
                    advice = "They look fearful — slow, calm reassurance may help."
                elif label in ("surprised", "shocked"):
                    advice = "Ask a clarifying question — they may need context."
                elif label == "angry":
                    advice = "Use a calm tone and offer space to respond."
                elif label == "disgust":
                    advice = "They appear disgusted — acknowledge and give space."
                elif label == "contempt":
                    advice = "They show contempt — keep questions neutral and concise."
                elif label == "confused":
                    advice = "They look confused — clarify or rephrase your point."
                elif label == "bored":
                    advice = "They seem disengaged — try a different approach or ask a question."

                head_action = None
                if head_tilt_deg > HEAD_TILT_ALERT_THRESHOLD:
                    head_action = "Person may be copying (head tilted)."

                results.append({
                    "box": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                    "top_emotion": {"label": label, "score": round(float(conf), 3)},
                    "all_emotions": {
                        "happy": round(float(conf) if label == "happy" else 0.0, 3),
                        "surprised": round(float(conf) if label == "surprised" else 0.0, 3),
                        "shocked": round(float(conf) if label == "shocked" else 0.0, 3),
                        "fear": round(float(conf) if label == "fear" else 0.0, 3),
                        "anxious": round(float(conf) if label == "anxious" else 0.0, 3),
                        "sad": round(float(conf) if label == "sad" else 0.0, 3),
                        "angry": round(float(conf) if label == "angry" else 0.0, 3),
                        "disgust": round(float(conf) if label == "disgust" else 0.0, 3),
                        "contempt": round(float(conf) if label == "contempt" else 0.0, 3),
                        "confused": round(float(conf) if label == "confused" else 0.0, 3),
                        "bored": round(float(conf) if label == "bored" else 0.0, 3),
                        "neutral": round(float(conf) if label == "neutral" else 0.0, 3)
                    },
                    "mouth_open": mouth_open_score,
                    "eye_open": eye_open_score,
                    "mouth_corner_offset": mouth_corner_offset_norm,
                    "head_tilt": round(head_tilt_deg, 2),
                    "head_tilt_deg": round(head_tilt_deg, 2),
                    "eyebrow": {"dist_norm": brow_dist, "offset": brow_offset},
                    "advice": advice,
                    "head_action": head_action
                })

            except Exception:
                traceback.print_exc()
                continue

    except Exception:
        traceback.print_exc()

    return results


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
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

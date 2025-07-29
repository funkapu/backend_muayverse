from fastapi import FastAPI, WebSocket
import numpy as np
import tensorflow as tf
import keras
from train_model import contrastive_loss, siamese_accuracy, L2Distance
import time

app = FastAPI()

# ✅ Load Siamese Model สำหรับ Jab
jab_model = keras.models.load_model(
    "siamese_model.keras",
    custom_objects={
        "contrastive_loss": contrastive_loss,
        "siamese_accuracy": siamese_accuracy,
        "L2Distance": L2Distance
    }
)

# ✅ Load Jab Reference Keypoints
jab_refs = np.load("jab_ref_video.npy")  # shape = (N, 99)

# ✅ Helper: Elbow angle (ใช้แขนซ้าย)
def get_elbow_angle(keypoints):
    shoulder = np.array(keypoints[12][:2])  # Left shoulder
    elbow = np.array(keypoints[14][:2])     # Left elbow
    wrist = np.array(keypoints[16][:2])     # Left wrist
    a = shoulder - elbow
    b = wrist - elbow
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

# ✅ Helper: Normalize keypoints
def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 3)
    hip = (keypoints[23] + keypoints[24]) / 2
    keypoints -= hip
    flat = keypoints.flatten()
    norm = np.linalg.norm(flat)
    return flat / (norm + 1e-6), keypoints

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

    prev_wrist = None
    score = 0
    last_score_time = 0
    cooldown = 1.0
    initialized = False

    while True:
        try:
            data = await websocket.receive_json()
            raw = data["keypoints"]
            pose_type = data.get("poseType", "Jab")

            if pose_type != "Jab":
                await websocket.send_json({
                    "similarity": 0,
                    "elbow_angle": 0,
                    "score": score,
                    "feedback": "unsupported_pose"
                })
                continue

            flat, kps = normalize_keypoints(raw)
            elbow_angle = get_elbow_angle(kps)

            # Siamese prediction
            user = np.expand_dims(flat, axis=0)
            batch = np.repeat(user, repeats=jab_refs.shape[0], axis=0)
            dists = jab_model.predict([batch, jab_refs], verbose=0).flatten()
            sim = 1 - np.min(dists)
            sim = np.clip(sim, 0, 1)
            percent = int(np.clip((sim - 0.7) / 0.3, 0, 1) * 100) * 2

            # ความเร็วข้อมือซ้าย
            wrist = kps[15][:2]  # ← Left wrist
            dx = np.linalg.norm(wrist - prev_wrist) if prev_wrist is not None else 0
            prev_wrist = wrist

            feedback = ""
            now = time.time()
            if initialized:
                if percent >= 60 and 30 <= elbow_angle <= 50 and dx > 0.01:
                    if now - last_score_time >= cooldown:
                        score += 1
                        last_score_time = now
                        feedback = "good"
                elif dx < 0.005:
                    feedback = "slow"
                elif elbow_angle < 30:
                    feedback = "elbow_up"
                elif elbow_angle > 50:
                    feedback = "elbow_down"
            else:
                initialized = True

            await websocket.send_json({
                "similarity": percent,
                "elbow_angle": int(elbow_angle),
                "score": score,
                "feedback": feedback,
            })

        except Exception as e:
            print("❌ Error:", e)
            break

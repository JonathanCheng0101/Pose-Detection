# ProClone: Your NBA Shooting Posture Mirror

**ProClone** is a lightweight, phone-video–based system that tells you **“You shoot like …”** an NBA superstar. By extracting five key joint angles frame-by-frame, it matches your biomechanical signature against 20 legends—no sensors required.

---

## 🚀 Key Features

- **Video-only input** — just record on your phone  
- **Pose detection** via OpenCV + MediaPipe (shoulder, elbow, wrist, hip, knee)  
- **Residual MLP** classification  
  - 90 % frame-level accuracy  
  - 70 % shot-level majority-vote accuracy  
- **Nearest-neighbor search** on z-score-normalized angle vectors  
- **Real-time inference**: under 3 s per clip on a laptop GPU  
- **Top-3 matches** so you know which pros you resemble  

---

## 🎯 Why ProClone?

Most shooting apps point out errors; ProClone tells you _who_ you shoot like—making practice more engaging, motivating, and personalized.

---

## 🛠️ How It Works

### 1. Capture & Preprocess

1. Record your jump shot on any phone camera.  
2. Extract keypoint coordinates & compute five joint angles per frame.  
3. Resample each clip to a fixed 34-frame sequence for consistency.

### 2. Pose Detection

We use MediaPipe’s Pose solution to detect 33 landmarks, then compute five joint angles:

```python
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_angles(frame: np.ndarray) -> dict:
    # Convert BGR → RGB
    results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return {}

    lm = results.pose_landmarks.landmark
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    pts = {p.name: (lm[p].x, lm[p].y, lm[p].z)
           for p in mp.solutions.pose.PoseLandmark}

    return {
        "shoulder": angle(pts["RIGHT_HIP"],    pts["RIGHT_SHOULDER"], pts["RIGHT_ELBOW"]),
        "elbow":    angle(pts["RIGHT_SHOULDER"], pts["RIGHT_ELBOW"],    pts["RIGHT_WRIST"]),
        "wrist":    angle(pts["RIGHT_ELBOW"],    pts["RIGHT_WRIST"],    pts["RIGHT_INDEX"]),
        "hip":      angle(pts["RIGHT_SHOULDER"], pts["RIGHT_HIP"],      pts["RIGHT_KNEE"]),
        "knee":     angle(pts["RIGHT_HIP"],      pts["RIGHT_KNEE"],     pts["RIGHT_ANKLE"]),
    }

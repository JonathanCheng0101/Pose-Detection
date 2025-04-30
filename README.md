#ProClone: Your NBA Shooting Posture Mirror
ProClone is a lightweight, phone-video–based system that tells you “You shoot like …” an NBA superstar. By extracting five key joint angles frame-by-frame, it matches your biomechanical signature against 20 legends—no sensors required.

🚀 Key Features
Video-only input—just record on your phone

Pose detection via OpenCV + MediaPipe (shoulder, elbow, wrist, hip, knee)

Residual MLP classification

90 % frame-level accuracy

70 % shot-level majority-vote accuracy

Nearest-neighbor search on z-score‐normalized angle vectors

Real-time inference: under 3 s per clip on a laptop GPU

Top-3 matches so you know which pros you resemble

🎯 Why ProClone?
Most shooting apps point out errors; ProClone tells you who you shoot like—making practice more engaging, motivating, and personalized.

🛠️ How It Works
1. Capture & Preprocess
Record your jump shot on any phone camera.

Resample each video clip to a fixed 34-frame sequence for consistency.

2. Pose Detection
We use MediaPipe’s Pose solution to detect 33 landmarks, then compute five joint angles:

python
Copy
Edit
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_angles(frame: np.ndarray):
    # Convert BGR→RGB
    results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    lm = results.pose_landmarks
    if not lm: return None

    # helper to compute angle between points a–b–c
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    pts = {p.name: (lm.landmark[p].x, lm.landmark[p].y, lm.landmark[p].z)
           for p in mp.solutions.pose.PoseLandmark}
    return {
      "shoulder": angle(pts["RIGHT_HIP"], pts["RIGHT_SHOULDER"], pts["RIGHT_ELBOW"]),
      "elbow":    angle(pts["RIGHT_SHOULDER"], pts["RIGHT_ELBOW"], pts["RIGHT_WRIST"]),
      "wrist":    angle(pts["RIGHT_ELBOW"], pts["RIGHT_WRIST"], pts["RIGHT_INDEX"]),
      "hip":      angle(pts["RIGHT_SHOULDER"], pts["RIGHT_HIP"], pts["RIGHT_KNEE"]),
      "knee":     angle(pts["RIGHT_HIP"], pts["RIGHT_KNEE"], pts["RIGHT_ANKLE"]),
    }
Loop this over each frame in your clip, then assemble into a DataFrame and resample to 34 frames.

3. Feature Normalization & Matching
Flatten angles → 170-D vector

Z-score normalize against our NBA dataset

Compute cosine similarity for nearest-neighbor lookup

4. Inference & Feedback
Residual MLP predicts your top match (90 % frame-level)

Shot-level majority vote gives 70 % accuracy across full clips

Nearest-neighbor returns Top-3 “You shoot like …” recommendations

📦 Installation & Usage
bash
Copy
Edit
git clone https://github.com/your-org/proclone.git
cd proclone
pip install -r requirements.txt
python
Copy
Edit
from proclone import ProClone

pc = ProClone(
    mongo_uri="mongodb+srv://…",
    db_name="pose_db_new",
    n_frames=34
)
# Pass in your video frames as a DataFrame of raw images:
matches = pc.match_user_shot("Jon", video_frames_df)
print(matches)
# → [("Jalen_Green", 95.2), ("Chris_Paul", 94.7), ("Donovan_Mitchell", 93.8)]

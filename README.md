# ProClone: Your NBA Shooting Posture Mirror

**ProClone** is a lightweight, phone-video–based system that tells you **“You shoot like …”** an NBA superstar. Instead of generic tips, ProClone captures your jump shot, extracts five key joint angles frame‐by‐frame, and instantly matches your biomechanical signature against 20 NBA legends.

## 🚀 Features

- **Video-only input**—no sensors or markers required  
- **Pose extraction** with OpenCV + MediaPipe to get shoulder, elbow, wrist, hip, and knee angles  
- **Residual MLP** classifier:  
  - 90 % frame‐level accuracy  
  - 70 % shot‐level majority‐vote accuracy  
- **Cosine‐similarity search** on z-score normalized angle vectors for “nearest‐neighbor” style matches  
- **Real‐time inference**: under 3 s per clip on a laptop GPU  
- **Top-3 recommendations** so you can see your closest style doppelgängers  

## 🎯 Motivation

Perfecting a basketball shot is a full-body endeavor—from legs through core to release—but most apps only flag flaws. ProClone fills that gap by telling you which pro you resemble, making training more engaging and actionable.

## 🛠️ How It Works

1. **Capture & preprocess**  
   - Film your jump shot on your phone  
   - Extract keypoint coordinates & compute five joint angles  
   - Resample each shot to a fixed 34-frame sequence  

2. **Feature normalization & matching**  
   - Flatten angles into a 170-D vector, then z-score normalize against our NBA dataset  
   - Compute cosine similarities to find your closest stylistic match  

3. **Inference & feedback**  
   - Run a residual MLP for classification + nearest‐neighbor lookup  
   - Return a ranked “You shoot like …” list (e.g. **Jalen Green**, Chris Paul, Donovan Mitchell)  

## 📦 Installation

```bash
git clone https://github.com/your-org/proclone.git
cd proclone
pip install -r requirements.txt


⚙️ Usage
from proclone import ProClone

model = ProClone(
    mongo_uri="mongodb+srv://…",
    players_collection="pose_db_new",
    n_frames=34
)

# Analyze your shot:
top_matches = model.match_user_shot("Jon", video_frames_df)
print(top_matches)
# -> [("Jalen_Green", 95.2), ("Chris_Paul", 94.7), ("Donovan_Mitchell", 93.8)]

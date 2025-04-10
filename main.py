import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

import cv2
import mediapipe as mp
import numpy as np

# ---- 1. Load env & connect to MongoDB Atlas ----
# Use your connection string. You may also load this from an .env file.
MONGODB_URI = "mongodb+srv://JonCheng:Jona0101@cluster0.nhjixga.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client     = MongoClient(MONGODB_URI)
db         = client["pose_db"]

# Determine the player name from the video file name.
video_file = "Videos/Chris_Paul.mp4"
player_name = os.path.splitext(os.path.basename(video_file))[0] 
collection = db[player_name]  # Use the player name as the collection name

# Choose which side to track: "right" or "left"
# You can also load this from env or pass as an argument
HAND = os.getenv("HAND", "right").lower()
if HAND not in ("right", "left"):
    raise ValueError("HAND must be 'right' or 'left'")

# ---- 2. MediaPipe Pose init & angle calculation function ----
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculate the angle at point b formed by points a–b–c.
    All points are (x, y) normalized to [0, 1].
    Returns the angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

# ---- 3. Read video & collect angles ----
angle_data = []
cap = cv2.VideoCapture(video_file)  # or 0 for webcam

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    frame_counter = 0

    # Map side → the right enum names in PoseLandmark for the upper body and lower body
    LM = mp_pose.PoseLandmark
    if HAND == "right":
        SHOULDER = LM.RIGHT_SHOULDER
        ELBOW    = LM.RIGHT_ELBOW
        WRIST    = LM.RIGHT_WRIST
        HIP      = LM.RIGHT_HIP
        INDEX    = LM.RIGHT_INDEX
        KNEE     = LM.RIGHT_KNEE
        ANKLE    = LM.RIGHT_ANKLE
    else:
        SHOULDER = LM.LEFT_SHOULDER
        ELBOW    = LM.LEFT_ELBOW
        WRIST    = LM.LEFT_WRIST
        HIP      = LM.LEFT_HIP
        INDEX    = LM.LEFT_INDEX
        KNEE     = LM.LEFT_KNEE
        ANKLE    = LM.LEFT_ANKLE

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = pose.process(image_rgb)

        if results.pose_landmarks:
            # Convert back to BGR for OpenCV display
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            lm = results.pose_landmarks.landmark

            # Extract keypoints for the arm and upper body
            p_sh  = [lm[SHOULDER.value].x, lm[SHOULDER.value].y]
            p_el  = [lm[ELBOW.value].x,    lm[ELBOW.value].y]
            p_wr  = [lm[WRIST.value].x,    lm[WRIST.value].y]
            p_hp  = [lm[HIP.value].x,      lm[HIP.value].y]
            p_idx = [lm[INDEX.value].x,    lm[INDEX.value].y]

            # Extract keypoints for the leg
            p_kn  = [lm[KNEE.value].x,     lm[KNEE.value].y]
            p_ank = [lm[ANKLE.value].x,    lm[ANKLE.value].y]

            # Calculate angles for the arm/upper body
            angle_shoulder = calculate_angle(p_el, p_sh, p_hp)  # shoulder angle using elbow, shoulder, hip
            angle_elbow    = calculate_angle(p_sh, p_el, p_wr)  # elbow angle using shoulder, elbow, wrist
            angle_wrist    = calculate_angle(p_el, p_wr, p_idx)  # wrist angle using elbow, wrist, index finger

            # Calculate angles for the lower body on the same side
            angle_hip   = calculate_angle(p_sh, p_hp, p_kn)       # hip angle using shoulder, hip, knee
            angle_knee  = calculate_angle(p_hp, p_kn, p_ank)      # knee angle using hip, knee, ankle

            # Append the data for this frame
            angle_data.append({
                "frame": frame_counter,
                "side": HAND,
                "shoulder_angle": float(f"{angle_shoulder:.2f}"),
                "elbow_angle":    float(f"{angle_elbow:.2f}"),
                "wrist_angle":    float(f"{angle_wrist:.2f}"),
                "hip_angle":      float(f"{angle_hip:.2f}"),
                "knee_angle":     float(f"{angle_knee:.2f}")
            })

            # Overlay the angle values on the image
            h, w, _ = image_bgr.shape
            cv2.putText(image_bgr, f"Sh: {angle_shoulder:.1f}",
                        (int(p_sh[0]*w), int(p_sh[1]*h)-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(image_bgr, f"El: {angle_elbow:.1f}",
                        (int(p_el[0]*w), int(p_el[1]*h)-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(image_bgr, f"Wr: {angle_wrist:.1f}",
                        (int(p_wr[0]*w), int(p_wr[1]*h)-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(image_bgr, f"Hip: {angle_hip:.1f}",
                        (int(p_hp[0]*w), int(p_hp[1]*h)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(image_bgr, f"Knee: {angle_knee:.1f}",
                        (int(p_kn[0]*w), int(p_kn[1]*h)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            cv2.imshow("Pose Detection", image_bgr)
        else:
            cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# ---- 4. Batch-insert into MongoDB Atlas ----
if angle_data:
    result = collection.insert_many(angle_data)
    print(f"Inserted {len(result.inserted_ids)} records (player={player_name}, side={HAND})")
else:
    print("No data to insert")

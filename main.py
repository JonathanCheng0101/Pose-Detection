# main.py

import os
import glob
from dotenv import load_dotenv
from pymongo import MongoClient

import cv2
import mediapipe as mp
import numpy as np

# ---- 1. Load environment variables & connect to MongoDB Atlas ----
load_dotenv()
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://JonCheng:Jona0101@cluster0.nhjixga.mongodb.net/?retryWrites=true&w=majority"
)
client = MongoClient(MONGODB_URI)
db     = client["pose_db_new"]

# ---- 2. Determine which hand/side to track ----
HAND = os.getenv("HAND", "right").lower()
if HAND not in ("right", "left"):
    raise ValueError("HAND must be 'right' or 'left'")

# ---- 3. Initialize MediaPipe & define angle calculation ----
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
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) \
            - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

# ---- 4. Find all Stephen Curry videos ----
video_files = glob.glob("Videos/Stephen_Curry*.mp4")
if not video_files:
    print(" No matching videos found in the 'Videos/' directory.")
    exit(1)

# ---- 5. Process each video ----
for video_file in video_files:
    raw_name = os.path.splitext(os.path.basename(video_file))[0]
    parts = raw_name.split("_")

    # If the last segment is numeric, use it as video_id and remove it
    if parts[-1].isdigit():
        video_id = int(parts[-1])
        parts = parts[:-1]
    else:
        # No numeric suffix: default to video 1
        video_id = 1

    player_name = "_".join(parts)      # e.g. "Stephen_Curry"
    collection  = db[player_name]

    print(f"\n Processing video `{video_id}.mp4`, inserting into collection `{player_name}`...")
    angle_data = []
    cap = cv2.VideoCapture(video_file)

    # Configure landmarks based on the chosen side
    LM = mp_pose.PoseLandmark
    if HAND == "right":
        SHOULDER, ELBOW, WRIST = LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST
        HIP, INDEX, KNEE, ANKLE = LM.RIGHT_HIP, LM.RIGHT_INDEX, LM.RIGHT_KNEE, LM.RIGHT_ANKLE
    else:
        SHOULDER, ELBOW, WRIST = LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST
        HIP, INDEX, KNEE, ANKLE = LM.LEFT_HIP, LM.LEFT_INDEX, LM.LEFT_KNEE, LM.LEFT_ANKLE

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = pose.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Extract keypoints
                p_sh  = [lm[SHOULDER.value].x, lm[SHOULDER.value].y]
                p_el  = [lm[ELBOW.value].x,    lm[ELBOW.value].y]
                p_wr  = [lm[WRIST.value].x,    lm[WRIST.value].y]
                p_hp  = [lm[HIP.value].x,      lm[HIP.value].y]
                p_idx = [lm[INDEX.value].x,    lm[INDEX.value].y]
                p_kn  = [lm[KNEE.value].x,     lm[KNEE.value].y]
                p_ank = [lm[ANKLE.value].x,    lm[ANKLE.value].y]

                # Calculate body part angles
                angle_shoulder = calculate_angle(p_el, p_sh,  p_hp)
                angle_elbow    = calculate_angle(p_sh, p_el,  p_wr)
                angle_wrist    = calculate_angle(p_el, p_wr,  p_idx)
                angle_hip      = calculate_angle(p_sh, p_hp,  p_kn)
                angle_knee     = calculate_angle(p_hp, p_kn,  p_ank)

                # Collect data
                angle_data.append({
                    "frame":           frame_counter,
                    "side":            HAND,
                    "shoulder_angle":  float(f"{angle_shoulder:.2f}"),
                    "elbow_angle":     float(f"{angle_elbow:.2f}"),
                    "wrist_angle":     float(f"{angle_wrist:.2f}"),
                    "hip_angle":       float(f"{angle_hip:.2f}"),
                    "knee_angle":      float(f"{angle_knee:.2f}"),
                    "video_id":        video_id
                })

                # (Optional) Overlay angles on the frame
                h, w, _ = frame.shape
                cv2.putText(frame, f"Sh:{angle_shoulder:.1f}", (int(p_sh[0]*w), int(p_sh[1]*h)-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(frame, f"El:{angle_elbow:.1f}",    (int(p_el[0]*w), int(p_el[1]*h)-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(frame, f"Wr:{angle_wrist:.1f}",    (int(p_wr[0]*w), int(p_wr[1]*h)-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(frame, f"Hip:{angle_hip:.1f}",     (int(p_hp[0]*w), int(p_hp[1]*h)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(frame, f"Kn:{angle_knee:.1f}",     (int(p_kn[0]*w), int(p_kn[1]*h)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow("Pose Detection", frame)
            else:
                cv2.imshow("Pose Detection", frame)

            # Press 'q' to quit processing this video and move to the next one
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # ---- 6. Insert into MongoDB ----
    if angle_data:
        result = collection.insert_many(angle_data)
        print(f" Inserted {len(result.inserted_ids)} records into collection `{player_name}` (video_id `{video_id}`).")
    else:
        print(f" No pose data detected for video_id `{video_id}`; nothing was inserted.")

print(" All videos have been processed.")

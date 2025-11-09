import cv2
import mediapipe as mp
import pandas as pd
import os
import math

# ---------- CONFIG ----------
input_folder = "data/dataset_frames/Good"   # 📁 folder with images
output_csv = "data/posture_features_good.csv"  # ✅ output CSV
# ----------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

# ✅ Helper Functions
def angle_between(p1, p2):
    """Angle in degrees between two points (p1->p2)."""
    return math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))

def distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return math.sqrt((p1.x - p2.x)*2 + (p1.y - p2.y)2 + (p1.z - p2.z)*2)

def extract_posture_features(landmarks):
    """Extracts 9 geometric posture features from Mediapipe landmarks."""

    # Key landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
    mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]
    left_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Derived midpoints
    mid_sh_x = (left_sh.x + right_sh.x) / 2
    mid_sh_y = (left_sh.y + right_sh.y) / 2
    mid_sh_z = (left_sh.z + right_sh.z) / 2

    shoulder_width = distance(left_sh, right_sh)

    # ---- Core Features ----
    # 1️⃣ Shoulder Angle
    shoulder_angle = angle_between(left_sh, right_sh)

    # 2️⃣ Head Tilt Angle
    head_tilt_angle = angle_between(left_eye, right_eye)

    # 3️⃣ Neck Inclination (nose vs shoulder midpoint)
    neck_inclination = math.degrees(math.atan2(nose.y - mid_sh_y, nose.x - mid_sh_x))

    # 4️⃣ Center Offset Ratio (horizontal offset of nose)
    center_offset_ratio = abs(nose.x - mid_sh_x) / shoulder_width if shoulder_width > 0 else 0

    # 5️⃣ Forward Lean Depth (z difference)
    forward_lean_depth = nose.z - mid_sh_z

    # 6️⃣ Face Rotation (nose x vs shoulders midpoint)
    face_rotation = (nose.x - mid_sh_x) / shoulder_width if shoulder_width > 0 else 0

    # 7️⃣ Eye-Level Difference
    eye_level_diff = abs(left_eye.y - right_eye.y)

    # 8️⃣ Ear-Level Difference
    ear_level_diff = abs(left_ear.y - right_ear.y)

    # 9️⃣ Mouth Skew Angle
    mouth_skew_angle = abs(angle_between(mouth_left, mouth_right))

    return [
        shoulder_angle,
        head_tilt_angle,
        neck_inclination,
        center_offset_ratio,
        forward_lean_depth,
        face_rotation,
        eye_level_diff,
        ear_level_diff,
        mouth_skew_angle
    ]

# ---------------- Main Loop ----------------
data = []

for file in os.listdir(input_folder):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipped unreadable image: {file}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"⚠️ No landmarks detected for {file}")
        continue

    try:
        features = extract_posture_features(results.pose_landmarks.landmark)
        data.append([file] + features)
    except Exception as e:
        print(f"❌ Error on {file}: {e}")
        continue

# ✅ Save to CSV
cols = [
    "filename",
    "Shoulder_Angle",
    "Head_Tilt_Angle",
    "Neck_Inclination",
    "Center_Offset_Ratio",
    "Forward_Lean_Depth",
    "Face_Rotation",
    "Eye_Level_Difference",
    "Ear_Level_Difference",
    "Mouth_Skew_Angle"
]

df = pd.DataFrame(data, columns=cols)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

print(f"\n✅ Extracted {len(df)} posture feature rows and saved to {output_csv}")
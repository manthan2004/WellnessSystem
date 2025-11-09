import cv2
import os

video_name = "manthan_1.mp4"
label_name = "Good"
video_path = f"data/videos/{video_name}"
output_folder = f"data/dataset_frames/{label_name}"
frame_interval = 2  # seconds

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"🎥 Extracting from {video_path}")
print(f"Detected FPS: {fps}, Total frames: {total_frames}")

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % (fps * frame_interval) == 0:
        frame = cv2.resize(frame, (1280, 720))
        filename = f"{output_folder}/frame_{saved_count:03d}.jpg"
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Done! Extracted {saved_count} frames → saved in '{output_folder}'")

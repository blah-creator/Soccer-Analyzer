import cv2
import os


video_path = "germany_ball.mp4"

output_folder = "germany_ball_dataset"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Save every 10th frame
    if frame_count % 10 == 0:
        filename = os.path.join(output_folder, f"germany_frame_{saved_count:05d}.jpg")

        cv2.imwrite(filename, frame)

        saved_count += 1

    frame_count += 1


cap.release()

print(f"Saved {saved_count} frames")

import mediapipe as mp
import cv2
from body_parts_coords import get_coords
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from angles import angle
from ultralytics import YOLO
from detect_contact import detect_contact
import math

while True:
    pref_foot = input("What is your preferred foot? (left/right): ")
    if pref_foot in ["left", "right"]:
        break
    else:
        print("Invalid input. Please enter 'left' or 'right'.")

model = YOLO("yolo26n.pt")
soccer_ball_model = YOLO("runs/detect/train-2/weights/best.pt")

base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=False
)

detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture("soccer_kick.mp4")
sb_dia_ft = 8.8 / 12
frame_count = 0
ball = True
distance_dict = {}
knee_angles = {}
ankle_angles = {}
foot_coords = {}
while cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    second = frame_count / fps
    if not ret:
        break
    soccer_ball_results = soccer_ball_model(frame, imgsz=640)
    big_height, big_width, _ = frame.shape
    max_height = 700
    max_width = 800
    scale = min(max_width / big_width, max_height / big_height)
    width = int(big_width * scale)
    height = int(big_height * scale)
    frame = cv2.resize(frame, (width, height))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    results = detector.detect(mp_image)

    body_coords = get_coords(results)
    if pref_foot == "left":
        right_foot_list = body_coords["right_foot_index"]
        foot_coords[frame_count] = (
            right_foot_list[0] * width,
            right_foot_list[1] * height,
        )
    elif pref_foot == "right":
        left_foot_list = body_coords["left_foot_index"]
        foot_coords[frame_count] = (
            left_foot_list[0] * width,
            left_foot_list[1] * height,
        )

    right_knee_angle = angle(
        body_coords["right_hip"], body_coords["right_knee"], body_coords["right_ankle"]
    )
    left_knee_angle = angle(
        body_coords["left_hip"], body_coords["left_knee"], body_coords["left_ankle"]
    )
    knee_angles[round(second, 2)] = (
        round(right_knee_angle, 2),
        round(left_knee_angle, 2),
    )
    right_ankle_angle = angle(
        body_coords["right_knee"],
        body_coords["right_ankle"],
        body_coords["right_foot_index"],
    )
    left_ankle_angle = angle(
        body_coords["left_knee"],
        body_coords["left_ankle"],
        body_coords["left_foot_index"],
    )
    ankle_angles[round(second, 2)] = (
        round(right_ankle_angle, 2),
        round(left_ankle_angle, 2),
    )

    for part, coords in body_coords.items():
        x = int(coords[0] * width)
        y = int(coords[1] * height)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    connections = [
        ("left_foot_index", "left_heel"),
        ("left_heel", "left_ankle"),
        ("left_ankle", "left_knee"),
        ("left_knee", "left_hip"),
        ("left_hip", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_foot_index", "right_heel"),
        ("right_heel", "right_ankle"),
        ("right_ankle", "right_knee"),
        ("right_knee", "right_hip"),
        ("right_hip", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_ear", "left_eye"),
        ("left_eye", "nose"),
        ("nose", "right_eye"),
        ("right_eye", "right_ear"),
    ]

    for start, end in connections:
        start_coords = body_coords[start]
        end_coords = body_coords[end]

        start_x = int(start_coords[0] * width)
        start_y = int(start_coords[1] * height)
        end_x = int(end_coords[0] * width)
        end_y = int(end_coords[1] * height)

        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    nose_coords = body_coords["nose"]
    nose_x = int(nose_coords[0] * width)
    nose_y = int(nose_coords[1] * height)
    left_shoulder_coords = body_coords["left_shoulder"]
    left_shoulder_x = int(left_shoulder_coords[0] * width)
    left_shoulder_y = int(left_shoulder_coords[1] * height)
    right_shoulder_coords = body_coords["right_shoulder"]
    right_shoulder_x = int(right_shoulder_coords[0] * width)
    right_shoulder_y = int(right_shoulder_coords[1] * height)
    rightest_x = max(left_shoulder_x, right_shoulder_x)
    leftest_x = min(left_shoulder_x, right_shoulder_x)
    middle_shoulder_x = int(rightest_x - ((rightest_x - leftest_x) / 2))
    lowest_y = max(left_shoulder_y, right_shoulder_y)
    highest_y = min(left_shoulder_y, right_shoulder_y)
    middle_shoulder_y = int(lowest_y - ((lowest_y - highest_y) / 2))
    cv2.line(
        frame, (nose_x, nose_y), (middle_shoulder_x, middle_shoulder_y), (0, 255, 0), 2
    )
    cv2.circle(frame, (middle_shoulder_x, middle_shoulder_y), 5, (0, 255, 0), -1)
    if len(soccer_ball_results[0].boxes.xyxy) > 0 and ball == True:
        ox1, oy1, ox2, oy2 = map(int, soccer_ball_results[0].boxes.xyxy[0])
        x1 = int(ox1 * scale)
        y1 = int(oy1 * scale)
        x2 = int(ox2 * scale)
        y2 = int(oy2 * scale)
        center_ball_coords = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.circle(frame, (x1, (int((y2 - y1) / 2)) + y1), 5, (0, 255, 0), -1)
        cv2.circle(frame, (x2, (int((y2 - y1) / 2)) + y1), 5, (0, 255, 0), -1)
        cv2.circle(frame, ((int((x2 - x1) / 2) + x1), y1), 5, (0, 255, 0), -1)
        cv2.circle(frame, ((int((x2 - x1) / 2) + x1), y2), 5, (0, 255, 0), -1)
        cv2.circle(frame, center_ball_coords, 5, (0, 255, 0), -1)

        distance, foot_x, foot_y = detect_contact(
            body_coords, center_ball_coords, pref_foot, width, height
        )
        cv2.line(frame, (int(foot_x), int(foot_y)), center_ball_coords, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Distance: {distance:.2f}",
            (int(foot_x + 10), int(foot_y + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        ball_dia_px = y2 - y1
        px_a_ft = ball_dia_px * (12 / 8.8)
        distance_dict[frame_count] = round(distance, 3)
    elif ball == False:
        ball = False
    elif len(soccer_ball_results[0].boxes.xyxy) <= 0:
        ball = False
        last_ball_second = second
    if ball == False and second - last_ball_second >= 0.1:
        ball = True

    

    frame_count += 1
    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

contact_frame = min(distance_dict, key=distance_dict.get)
distance_traveled_foot = math.hypot(
    foot_coords[contact_frame][0] - foot_coords[contact_frame - 1][0],
    foot_coords[contact_frame][1] - foot_coords[contact_frame - 1][1],
)
foot_speed_pxh = (distance_traveled_foot / px_a_ft) * fps
print(f"Knee Angles: {knee_angles}")
print(f"Ankle Angles: {ankle_angles}")
print(f"DISTANCE DICTIONARY: {distance_dict}")
print(f"Contact Time: {contact_frame / fps}")
print(f"Foot Speed at Contact: {foot_speed_pxh} feet/sec")
cap.release()
cv2.destroyAllWindows()

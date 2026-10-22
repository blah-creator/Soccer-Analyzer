body_parts_coords = {
    "nose": None,
    "left_eye": None,
    "right_eye": None,
    "left_ear": None,
    "right_ear": None,
    "left_shoulder": None,
    "right_shoulder": None,
    "left_elbow": None,
    "right_elbow": None,
    "left_wrist": None,
    "right_wrist": None,
    "left_hip": None,
    "right_hip": None,
    "left_knee": None,
    "right_knee": None,
    "left_ankle": None,
    "right_ankle": None,
    "left_heel": None,
    "right_heel": None,
    "left_foot_index": None,
    "right_foot_index": None,
}

body_part_index = {
    "nose": 0,
    "left_eye_inner": 1,
    "right_eye_inner": 4,
    "left_eye": 2,
    "right_eye": 5,
    "left_eye_outer": 3,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def get_coords(results):
    for part in body_parts_coords:
        index = body_part_index[part]
        body_parts_coords[part] = (
            results.pose_landmarks[0][index].x,
            results.pose_landmarks[0][index].y,
            results.pose_landmarks[0][index].z,
            results.pose_landmarks[0][index].visibility,
            results.pose_landmarks[0][index].presence,
        )

    return body_parts_coords

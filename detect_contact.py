import math


def detect_contact(body_coords, center_ball_coords, pref_foot, width, height):
    if pref_foot == "right":
        foot_coords = body_coords["left_foot_index"]
    elif pref_foot == "left":
        foot_coords = body_coords["right_foot_index"]
    foot_x = foot_coords[0] * width
    foot_y = foot_coords[1] * height
    ball_x, ball_y = center_ball_coords
    distance = math.hypot(foot_x - ball_x, foot_y - ball_y)

    return distance, foot_x, foot_y

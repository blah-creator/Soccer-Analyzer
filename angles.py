import math


def angle(high, mid, low):
    vector1 = (high[0] - mid[0], high[1] - mid[1], high[2] - mid[2])
    vector2 = (low[0] - mid[0], low[1] - mid[1], low[2] - mid[2])
    dot_product = (
        vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]
    )
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2 + vector1[2] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2 + vector2[2] ** 2)
    angle_rads = math.acos(dot_product / (magnitude1 * magnitude2))
    angle = math.degrees(angle_rads)
    return angle

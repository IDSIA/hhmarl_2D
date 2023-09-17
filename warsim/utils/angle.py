"""
    Angles computations
"""

import math

DEG_TO_RAD = math.pi / 180


def normalize_angle(a: float) -> float:
    while a >= 360.0:
        a -= 360
    while a < 0.0:
        a += 360
    return a


def sum_angles(a: float, b: float) -> float:
    return normalize_angle(a + b)


def signed_heading_diff(actual: float, desired: float) -> float:
    # actual and desired in [0, 360)
    delta = desired - actual
    if delta < -180:
        delta = 360 + delta
    if delta > 180:
        delta = -360 + delta
    return delta

import math


def wrap2pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

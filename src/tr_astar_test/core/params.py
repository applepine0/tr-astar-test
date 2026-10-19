from dataclasses import dataclass

"""单位约定:rps=rad/s
           mps=m/s
"""


@dataclass(frozen=True)
class RobotParams:
    """
    Parameters for the robot.
    """

    max_steer_rate_rps: float = 200
    length_mm: int = 700
    width_mm: int = 700

    max_drive_rate_mps: int = 3


@dataclass(frozen=True)
class UnmoveableZone:
    """
    Parameters for the unmoveable zone.
    默认为矩形，以起点为矩形靠近原点的对角点，终点为矩形远离原点的对角点
    """

    start_point: tuple[int, int]
    end_point: tuple[int, int]


UNMOVEABLE_ZONE: tuple[UnmoveableZone, ...] = (
    UnmoveableZone((2500, 2500), (8500, 5500)),  # L1
    UnmoveableZone((2800, 1500), (7200, 2500)),  # 阶梯，台阶，交接区
)


@dataclass(frozen=True)
class FieldParams:
    """
    Parameters for the field.
    """

    field_width_mm: int = 5500
    field_length_mm: int = 11000
    unmoveable_zones = UNMOVEABLE_ZONE

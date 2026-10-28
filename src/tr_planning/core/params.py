from dataclasses import dataclass

"""单位约定:rps=rad/s
           mps=m/s
"""


@dataclass(frozen=True)
class RobotParams:
    """
    Parameters for the robot.
    """

    # 车体度量参数 占位待实测
    wheel_radius_m: float = 0.1
    length_m: float = 0.7
    width_m: float = 0.7
    wheel_count: int = 4

    # 舵轮限幅参数

    max_steer_rate_rps: int = 6
    max_drive_rate_mps: int = 3

    @property
    def circumradius_m(self) -> float:
        """车体外接圆半径，供外扩用的保守值"""
        return 0.5 * (self.length_m**2 + self.width_m**2) ** 0.5


@dataclass(frozen=True)
class UnmoveableZone:
    """
    Parameters for the unmoveable zone.
    默认为矩形，以起点为矩形靠近原点的对角点，终点为矩形远离原点的对角点
    """

    start_point: tuple[float, float]
    end_point: tuple[float, float]


UNMOVEABLE_ZONE: tuple[UnmoveableZone, ...] = (
    UnmoveableZone((2.5, 2.5), (8.5, 5.5)),  # L1
    UnmoveableZone((2.8, 1.5), (7.2, 2.5)),  # 阶梯，台阶，交接区，以下称L0_5
)


@dataclass(frozen=True)
class FieldParams:
    """
    Parameters for the field.
    """

    start_point: tuple[float, float] = (0, 0)
    end_point: tuple[float, float] = (11.0, 5.5)
    unmoveable_zones = UNMOVEABLE_ZONE
    fence_halfthickness_m: float = 0.025  # 场地围栏厚度的一半

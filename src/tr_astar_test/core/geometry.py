import math
from dataclasses import dataclass


def wrap2pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def is_near_zero(num: float):
    return abs(num) < 1e-6


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    theta_rad: float

    def canonicalized(self) -> "Pose2D":
        return Pose2D(self.x_m, self.y_m, wrap2pi(self.theta_rad))


@dataclass(frozen=True)
class Polygon:
    # 简单多边形
    vertices: tuple[tuple[float, float], ...]

    def __post_init__(self):
        # 检查顶点数量是否至少为3个
        if len(self.vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

    def expand(self, r: float):
        """沿各顶点法线向外偏移 r 的粗略版本（非严格 Minkowski）。

        仅适用于凸多边形；对矩形等规则形状够用。
        严格 Minkowski 留给后续用 shapely 实现。
        """

        cx = sum(v[0] for v in self.vertices) / len(self.vertices)
        cy = sum(v[1] for v in self.vertices) / len(self.vertices)
        out = []
        for i in range(0, len(self.vertices), 1):
            x = self.vertices[i][0]
            y = self.vertices[i][1]
            dx = x - cx
            dy = y - cy
            ds = math.sqrt(dx**2 + dy**2)
            if is_near_zero(ds):
                out.append((x, y))
                continue
            out.append((x + r * dx / ds, y + r * dy / ds))
        return Polygon(tuple(out))

    @staticmethod
    def from_rect(x0: float, y0: float, x1: float, y1: float):
        """从矩形的左上角和右下角坐标构造逆时针多边形。"""
        # todo:防范传反坐标与退化图形
        return Polygon(((x0, y0), (x1, y1), (x0, y1)))
        return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    cr: float

    def expand(self, r: float):
        """沿圆心法线向外偏移 r 的粗略版本。"""
        return Circle(self.cx, self.cy, self.cr + r)

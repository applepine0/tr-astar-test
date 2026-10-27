import math
from dataclasses import dataclass

from shapely.geometry import Polygon as sgPolygon


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
    shape: str | None = None

    def __post_init__(self):
        # 检查顶点数量是否至少为3个
        if len(self.vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

    def expand(self, r: float):
        """依赖shapely的严格minkowski扩展，适用于任意多边形。"""
        shapely_geom = sgPolygon(self.vertices)
        buffered_geom = shapely_geom.buffer(r, quad_segs=8)  # 八段圆弧精度外扩

        if buffered_geom.geom_type == "Polygon":
            new_vertices = tuple(buffered_geom.exterior.coords)

        else:
            # todo: 处理缓冲区结果为 MultiPolygon 的情况，暂时抛出异常
            raise ValueError("Expanded geometry is not a polygon")
        return Polygon(new_vertices[:-1], self.shape)  # type: ignore # todo:对数据进行清洗保证符合格式

    def original_expand(self, r: float):
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
    def from_rect(d0: tuple[float, float], d1: tuple[float, float]):
        """从矩形的左上角和右下角坐标构造逆时针多边形。"""
        # todo:防范传反坐标与退化图形
        return Polygon(
            ((d0[0], d0[1]), (d1[0], d0[1]), (d1[0], d1[1]), (d0[0], d1[1]))
        )  # 此处应按照时针顺序传入以防止建图时产生交叉多边形


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    cr: float

    def expand(self, r: float):
        """沿圆心法线向外偏移 r 的粗略版本。"""
        return Circle(self.cx, self.cy, self.cr + r)

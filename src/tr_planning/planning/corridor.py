from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.geometry import is_convex_ccw, is_near_zero, signed_area


@dataclass(frozen=True, eq=False)  # eq=False以防止错自动生成逐元素比较的内置等于
class Halfspaces:
    """半空间：
    A储存M个归一化的法向量，b储存M个常数，法向量作为边界外法向量，常数作为边界到原点距离，唯一确定M个边界及其半空间
    """

    A: np.ndarray  # 法向量矩阵
    b: np.ndarray  # 常数向量

    def contains(self, x: float, y: float) -> bool:
        """输入点的横纵坐标，检查点是否在半空间内"""
        distance = self.margin(x, y)
        return distance >= 0 or is_near_zero(distance)

    def margin(self, x: float, y: float) -> float:
        """输入点的横纵坐标，返回点到半空间边界的最小距离(负表示超出)"""
        point = np.array([x, y])
        distances = self.b - self.A @ point
        return float(np.min(distances))


@dataclass(frozen=True)
class Corridor:
    vertices: tuple[tuple[float, float], ...]  # 走廊顶点，应按逆时针顺序排列
    _hs: Halfspaces = field(
        init=False, repr=False, compare=False
    )  # 半空间对象，供内部使用

    def __post_init__(self):
        v = tuple((float(x), float(y)) for x, y in self.vertices)
        if len(v) < 3:
            raise ValueError(f"走廊数量至少为3个，实际为{len(v)}")
        # 顺序必须先判：is_convex_ccw 在非逆时针时会直接抛异常
        if not signed_area(v) > 0:
            raise ValueError("走廊顶点必须按逆时针顺序排列")
        if not is_convex_ccw(v):
            raise ValueError("走廊顶点必须为凸多边形")
        object.__setattr__(self, "vertices", v)
        object.__setattr__(self, "_hs", _compile(v))  # 初始化半空间对象

    @property
    def halfspaces(self) -> Halfspaces:
        return self._hs

    def contains(self, x: float, y: float) -> bool:
        """检查点是否在走廊内"""
        return self._hs.contains(x, y)

    @property
    def centroid(self) -> tuple[float, float]:
        """走廊顶点形心"""
        x_sum = sum(x for x, y in self.vertices)
        y_sum = sum(y for x, y in self.vertices)
        n = len(self.vertices)
        return (x_sum / n, y_sum / n)

    @staticmethod
    def from_rect(x0: float, y0: float, x1: float, y1: float) -> Corridor:
        """从矩形坐标创建走廊"""
        vertices = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        return Corridor(vertices)


def _compile(v: tuple[tuple[float, float], ...]) -> Halfspaces:
    """将走廊顶点'编译'生成半空间对象"""
    n = len(v)
    A: list[tuple[float, float]] = []
    b: list[float] = []
    for i in range(n):
        x0, y0 = v[i]
        x1, y1 = v[(i + 1) % n]  # 循环取点
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        nx = -dx / length
        ny = dy / length
        A.append((ny, nx))  # 法向量
        b.append(x0 * (ny) + y0 * (nx))  # 距离
    return Halfspaces(np.array(A, dtype=float), np.array(b, dtype=float))


def centerline(
    corridors: tuple[Corridor, ...],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    """走廊'中心线':起点->各走廊中心点->终点"""
    return (start, *[c.centroid for c in corridors], end)

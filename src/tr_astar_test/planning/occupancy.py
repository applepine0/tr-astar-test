from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as Mplpath

from ..core.field import Field
from ..core.geometry import Circle, Polygon
from ..core.params import RobotParams


@dataclass(frozen=True)
class OccupancyGrid:
    res_m: float = 0.05  # 栅格分辨率
    padding_m: float | None = None  # 障碍外扩量，若None则用机器人外接圆半径

    def resolve_padding(self, robot: RobotParams) -> float:
        return self.padding_m if self.padding_m is not None else robot.circumradius_m


@dataclass
class Occupancy:
    """
    占据栅格。
    free[i,j]==True 表示栅格(i,j)为空闲,False表示被障碍占据。
    约定i沿x,j沿y,压缩为一维数组存储,索引为i*ny+j
    """

    occupied_grid: np.ndarray  # 储存每个栅格的阻碍状态，0表示可行
    resolution_m: float  # 栅格分辨率
    origin_m: tuple[float, float]  # 栅格原点在世界坐标
    padding_m: float  # 通用障碍外扩量，若None则用机器人外接圆半径

    # 动态阻塞层,True=被临时阻塞
    blocked_dynamic: np.ndarray = field(default=None)  # type: ignore[assignment] 简单忽略类型检查，但是在保障前init中一定会进行初始化的基础上

    def __post_init__(self) -> None:
        if self.blocked_dynamic is None:
            self.blocked_dynamic = np.zeros_like(
                self.occupied_grid, dtype=bool
            )  # 若未传入则自己创建
        if self.blocked_dynamic.shape != self.occupied_grid.shape:
            raise ValueError(
                f"注入动态层形状{self.blocked_dynamic.shape} "
                f"需与静态层形状{self.occupied_grid.shape}一致"
            )

    # 获取栅格行列数
    @property
    def nx(self) -> int:
        return self.occupied_grid.shape[0]

    @property
    def ny(self) -> int:
        return self.occupied_grid.shape[1]

    @property
    def occupied_ratio(self) -> float:
        if self.blocked_dynamic is None:
            return float(np.mean(self.occupied_grid))
        return float(np.mean(self.occupied_grid | self.blocked_dynamic))

    def world2ij(self, x_m, y_m) -> tuple[int, int]:
        """
        将世界坐标转换为栅格坐标,栅格包括其负向边界
        """
        i = math.floor((x_m - self.origin_m[0]) / self.resolution_m + 0.5)
        j = math.floor((y_m - self.origin_m[1]) / self.resolution_m + 0.5)
        return (i, j)

    def ij2world(self, i: int, j: int) -> tuple[float, float]:
        """
        将栅格坐标转换为其格点中心的世界坐标
        """
        x_m = self.origin_m[0] + i * self.resolution_m
        y_m = self.origin_m[1] + j * self.resolution_m
        return (x_m, y_m)

    def is_in_bounds(self, i: int, j: int) -> bool:
        """判断栅格坐标是否在栅格范围内"""
        return 0 <= i < self.nx and 0 <= j < self.ny

    # ---------- 可行性查询 ----------
    def is_free(self, i: int, j: int) -> bool:
        """判断栅格坐标是否为空闲"""
        if self.blocked_dynamic is None:
            return self.is_in_bounds(i, j) and not self.occupied_grid[i, j]
        else:
            return (
                self.is_in_bounds(i, j)
                and not self.occupied_grid[i, j]
                and not self.blocked_dynamic[i, j]
            )

    # --------- 动态阻塞层操作 ----------
    def clear_dynamic_blocked(self) -> None:
        """清空动态阻塞层"""
        self.blocked_dynamic.fill(False)

    '''
    很美丽但是暂时废弃的障碍设置函数
    def set_dynamic_circle(
        self, center_x: float, center_y: float, radius: float
    ) -> None:
        """利用一些神秘语法和机制实现在动态阻塞层中高效设置圆形障碍"""

        # 压缩待处理空间
        ci, cj = self.world2ij(center_x, center_y)
        r_cells = math.ceil(radius / self.resolution_m)

        i_min = max(0, ci - r_cells - 1)
        i_max = min(self.nx - 1, ci + r_cells + 1)
        j_min = max(0, cj - r_cells - 1)
        j_max = min(self.ny - 1, cj + r_cells + 1)

        if i_min > i_max or j_min > j_max:
            return  # 圆心在栅格外部?

        ii, jj = np.ogrid[
            i_min : i_max + 1, j_min : j_max + 1
        ]  # 生成目标栅格索引网格的行列向量

        xs = self.origin_m[0] + ii * self.resolution_m
        ys = self.origin_m[1] + jj * self.resolution_m

        mask = (xs - center_x) ** 2 + (
            ys - center_y
        ) ** 2 <= radius**2  # 利用numpy广播机制计算圆内的栅格掩码
        self.blocked_dynamic[i_min : i_max + 1, j_min : j_max + 1] |= mask  # 切片按位或
        '''

    def set_blocked(self, shape: Polygon | Circle, is_dynamic: bool = True) -> None:
        """在阻塞层中高效设置多边形或圆形障碍并进行扩张"""

        shape = _expand(shape, self.resolution_m)  # 外扩障碍
        # 压缩待处理空间
        if isinstance(shape, Circle):
            x_min = shape.cx - shape.cr
            x_max = shape.cx + shape.cr
            y_min = shape.cy - shape.cr
            y_max = shape.cy + shape.cr

        elif isinstance(shape, Polygon):
            # 获取多边形的包围盒
            x_min = min(v[0] for v in shape.vertices)
            x_max = max(v[0] for v in shape.vertices)
            y_min = min(v[1] for v in shape.vertices)
            y_max = max(v[1] for v in shape.vertices)

        i_min, j_min = self.world2ij(x_min, y_min)
        i_max, j_max = self.world2ij(x_max, y_max)

        # 安全限幅？
        i_min = max(0, i_min)
        j_min = max(0, j_min)
        i_max = min(self.nx - 1, i_max)
        j_max = min(self.ny - 1, j_max)

        ii, jj = np.ogrid[
            i_min : i_max + 1, j_min : j_max + 1
        ]  # 生成目标待处理栅格索引网格的行列向量

        x_grid = ii * self.resolution_m + self.origin_m[0]
        y_grid = jj * self.resolution_m + self.origin_m[1]
        mask = _rasterize(shape, x_grid, y_grid)

        if is_dynamic:
            self.blocked_dynamic[i_min : i_max + 1, j_min : j_max + 1] |= (
                mask  # 对动态障碍区切片按位或
            )
        else:
            self.occupied_grid[i_min : i_max + 1, j_min : j_max + 1] |= (
                mask  # 对静态障碍区切片按位或
            )


def _expand(shape: Polygon | Circle, pad: float) -> Polygon | Circle:
    return shape.expand(pad)


def _rasterize(
    shape: Polygon | Circle, grid_x_vector: np.ndarray, grid_y_vector: np.ndarray
) -> np.ndarray:
    """输入几何形状及栅格矩阵行列坐标向量，返回栅格化后的掩码，有障碍返回1。"""
    xs, ys = np.broadcast_arrays(grid_x_vector, grid_y_vector)  # 广播构造横纵坐标矩阵
    if isinstance(shape, Circle):  # 若为圆形则直接通过逐元素运算实现
        return (xs - shape.cx) ** 2 + (ys - shape.cy) ** 2 <= shape.cr**2
    path = Mplpath(
        np.asarray(shape.vertices, dtype=float)
    )  # 调用mplpath将多边形的vertices转换为Path对象
    pts = np.column_stack([xs.ravel(), ys.ravel()])  # 将坐标矩阵展平并按列堆叠为点集

    return path.contains_points(pts).reshape(
        xs.shape
    )  # 调用Path.contains_points判断点集是否在多边形内，并恢复为矩阵


def build_occupancy(
    field: Field, robot: RobotParams, cfg: OccupancyGrid | None = None
) -> Occupancy:
    """构建占据栅格"""
    cfg = cfg if cfg is not None else OccupancyGrid()
    pad = cfg.resolve_padding(robot)
    res = cfg.res_m

    # 更健壮地获取场地边界的栅格范围
    xs_bound = [v[0] for v in field.bound.vertices]
    ys_bound = [v[1] for v in field.bound.vertices]
    x0, x1 = min(xs_bound), max(xs_bound)
    y0, y1 = min(ys_bound), max(ys_bound)

    # 得到栅格尺寸并检查
    nx = math.ceil((x1 - x0) / res)
    ny = math.ceil((y1 - y0) / res)

    if nx <= 2 or ny <= 2:
        raise ValueError(f"场地边界尺寸过小,栅格化异常: nx={nx}, ny={ny}, res={res}")

    # 构建栅格
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    origin = x0 + 0.5 * res, y0 + 0.5 * res  # 栅格原点为左下角第一个栅格的中心点
    x_grid = ii * res + origin[0]
    y_grid = jj * res + origin[1]

    occupied = np.zeros((nx, ny), dtype=bool)

    margin = pad + field.params.fence_halfthickness_m  # 场地边界外扩量
    occupied[x_grid < x0 + margin] = True
    occupied[x_grid > x1 - margin] = True
    occupied[y_grid < y0 + margin] = True
    occupied[y_grid > y1 - margin] = True

    occupancy = Occupancy(
        occupied_grid=occupied,
        resolution_m=res,
        origin_m=origin,
        padding_m=pad,
    )

    # 处理障碍物和禁入区
    for shape in field.obstacles + field.keepout:
        shape = _expand(shape, pad)  # 外扩障碍
        occupancy.set_blocked(shape, False)
    return occupancy

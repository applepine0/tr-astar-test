"""pytest 共享配置与 fixture。

注意：本文件必须在 import tr_astar_test 之前设置 MPLCONFIGDIR，
因为 planning.occupancy 在模块级 import 了 matplotlib.path。
"""

from __future__ import annotations

import os
import tempfile

# 必须在导入被测包之前完成（matplotlib 的配置目录可能只读）
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg-"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tr_astar_test.core.field import Field
from tr_astar_test.core.params import (
    UNMOVEABLE_ZONE,
    FieldParams,
    RobotParams,
)
from tr_astar_test.planning.occupancy import (
    Occupancy,
    OccupancyGrid,
    build_occupancy,
)


@pytest.fixture
def robot() -> RobotParams:
    """默认机器人参数（占位值，待实测替换）。"""
    return RobotParams()


@pytest.fixture
def field() -> Field:
    """默认场地（11.0 x 5.5 的半场，含 L1 与阶梯/交接区）。"""
    return Field.make_field(FieldParams())


@pytest.fixture
def occ_factory():
    """按分辨率构建占据栅格的工厂。

    用工厂而不是固定 fixture：动态层测试会修改 Occupancy（可变对象），
    每个测试必须拿到独立实例，否则会互相污染。
    """

    def _make(res_m: float = 0.05, padding_m: float | None = None) -> Occupancy:
        return build_occupancy(
            Field.make_field(FieldParams()),
            RobotParams(),
            OccupancyGrid(res_m=res_m, padding_m=padding_m),
        )

    return _make


@pytest.fixture
def occ(occ_factory) -> Occupancy:
    return occ_factory(0.05)


@pytest.fixture
def measure_padding_band(occ_factory):
    """沿 y=const 由西向东扫描，量出"有效外扩量"。

    返回函数 res -> float | None（单位 m）。
    原理：L1 西立面在 x = zone.start_point[0]，外扩后该立面以西
    会形成一条"障碍带"；扫描第一个由自由变占据的位置，即可反推有效外扩。
    """

    x_face = UNMOVEABLE_ZONE[0].start_point[0]
    y_ref = 4.0  # L1 中部高度，避开阶梯区

    def _measure(res_m: float) -> float | None:
        occupancy = occ_factory(res_m)
        prev_occupied = None
        for x in np.arange(1.0, x_face + 0.1, 0.002):
            i, j = occupancy.world2ij(x, y_ref)
            is_occupied = bool(occupancy.occupied_grid[i, j])
            if prev_occupied is False and is_occupied:
                return x_face - x
            prev_occupied = is_occupied
        return None

    return _measure


@pytest.fixture
def free_bfs(occ):
    """在 free 栅格上做 4 邻域 BFS，判断两点是否连通。

    返回函数 (start_xy, goal_xy) -> bool。仅用于地图级连通性检查。
    """
    from collections import deque

    def _bfs(start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> bool:
        start = occ.world2ij(*start_xy)
        goal = occ.world2ij(*goal_xy)
        if not occ.is_free(*start) or not occ.is_free(*goal):
            return False
        seen = {start}
        queue = deque([start])
        while queue:
            i, j = queue.popleft()
            if (i, j) == goal:
                return True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (i + di, j + dj)
                if nxt in seen or not occ.is_free(*nxt):
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return False

    return _bfs

"""planning.occupancy 的测试。

分三类：
    1. 不变量     —— 索引换算、边界、形状（纯数学，最容易抓低级错误）
    2. 几何正确性 —— 哪些地方该通、哪些该堵
    3. 分辨率无关 —— 同一场景在不同 res 下结论必须一致（抓隐蔽的 res 依赖）

已知缺陷统一放在 TestKnownDefects 里用 xfail 标记，并写明实测值；
修好后它们会变成 XPASS，届时把 xfail 去掉改成硬断言即可。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tr_planning.core.field import Field
from tr_planning.core.params import UNMOVEABLE_ZONE, FieldParams, RobotParams
from tr_planning.planning.occupancy import (
    Occupancy,
    OccupancyGrid,
    build_occupancy,
)

# ---------------------------------------------------------------- 1. 不变量


class TestInvariants:
    def test_grid_shape_matches_field_size(self, occ):
        """网格形状必须与场地尺寸/分辨率一致（能抓 i/j 转置）。"""
        params = FieldParams()
        expected_nx = math.ceil(
            (params.end_point[0] - params.start_point[0]) / occ.resolution_m
        )
        expected_ny = math.ceil(
            (params.end_point[1] - params.start_point[1]) / occ.resolution_m
        )
        assert occ.nx == expected_nx
        assert occ.ny == expected_ny
        assert occ.occupied_grid.shape == (occ.nx, occ.ny)

    @pytest.mark.parametrize(
        "x,y",
        [(0.3, 0.3), (1.0, 1.0), (5.5, 0.8), (10.5, 5.0), (0.55, 4.0)],
    )
    def test_index_roundtrip(self, occ, x: float, y: float):
        i, j = occ.world2ij(x, y)
        wx, wy = occ.ij2world(i, j)
        assert abs(wx - x) <= 0.5 * occ.resolution_m + 1e-9
        assert abs(wy - y) <= 0.5 * occ.resolution_m + 1e-9

    def test_out_of_bounds_is_not_free(self, occ):
        """越界查询必须返回 False，且不能抛异常。"""
        assert occ.is_free(-1, -1) is False
        assert occ.is_free(occ.nx + 5, occ.ny + 5) is False
        assert occ.is_free(0, -3) is False

    def test_out_of_field_world_coords(self, occ):
        for x, y in [(-1.0, 1.0), (12.0, 1.0), (1.0, -1.0), (1.0, 7.0)]:
            i, j = occ.world2ij(x, y)
            assert occ.is_free(i, j) is False

    def test_declared_padding_defaults_to_circumradius(self, occ, robot):
        assert occ.padding_m == pytest.approx(robot.circumradius_m)

    def test_custom_padding_is_respected(self, occ_factory):
        occ = occ_factory(res_m=0.05, padding_m=0.2)
        assert occ.padding_m == pytest.approx(0.2)

    def test_occupied_ratio_in_plausible_range(self, occ):
        ratio = occ.occupied_ratio
        assert 0.3 < ratio < 0.95, f"占据比异常: {ratio}"

    def test_rejects_too_coarse_resolution(self):
        """分辨率粗到网格不足 3 格时必须显式报错。"""
        with pytest.raises(ValueError):
            build_occupancy(
                Field.make_field(FieldParams()),
                RobotParams(),
                OccupancyGrid(res_m=10.0),
            )


# ---------------------------------------------------------------- 2. 几何正确性


class TestGeometry:
    def test_l1_interior_is_occupied(self, occ):
        """L1 高台内部对 TR 是禁入 + 物理障碍，必须全部为占据。"""
        zone = UNMOVEABLE_ZONE[0]
        xs = np.linspace(zone.start_point[0] + 0.5, zone.end_point[0] - 0.5, 5)
        ys = np.linspace(zone.start_point[1] + 0.5, zone.end_point[1] - 0.5, 4)
        for x in xs:
            for y in ys:
                i, j = occ.world2ij(float(x), float(y))
                assert occ.occupied_grid[i, j], f"L1 内部 ({x:.2f},{y:.2f}) 未被占据"

    def test_stair_and_transfer_zone_is_occupied(self, occ):
        """阶梯/交接区现阶段按"未建模 → 禁行"处理。"""
        zone = UNMOVEABLE_ZONE[1]
        for x in np.linspace(zone.start_point[0] + 0.3, zone.end_point[0] - 0.3, 4):
            for y in np.linspace(zone.start_point[1] + 0.2, zone.end_point[1] - 0.2, 2):
                i, j = occ.world2ij(float(x), float(y))
                assert occ.occupied_grid[i, j], (
                    f"阶梯区 ({x:.2f},{y:.2f}) 未被占据：它会被规划器当成平地穿越"
                )

    @pytest.mark.parametrize(
        "point",
        [(0.8, 0.8), (10.2, 0.8), (0.8, 4.0), (10.2, 4.0), (5.5, 0.8)],
    )
    def test_corridor_points_are_free(self, occ, point: tuple[float, float]):
        i, j = occ.world2ij(*point)
        assert occ.is_free(i, j), f"走廊点 {point} 被判为占据"

    def test_fence_margin_blocks_border(self, occ, robot):
        """围栏内侧一圈必须被占据（外扩 + 半墙厚）。"""
        margin = robot.circumradius_m + FieldParams().fence_halfthickness_m
        for x, y in [(0.2, 2.0), (10.8, 2.0), (5.5, 0.2)]:
            assert margin > 0.1  # 防止参数退化成 0
            i, j = occ.world2ij(x, y)
            assert occ.occupied_grid[i, j], f"边界附近 ({x},{y}) 未被占据"

    def test_free_space_is_not_empty(self, occ):
        assert occ.occupied_grid.size > 0
        assert (~occ.occupied_grid).any()

    def test_corridor_is_connected_around_l1(self, free_bfs):
        """西侧走廊到东侧走廊必须连通（绕 L1 南侧），否则任务不可达。"""
        assert free_bfs((0.8, 0.8), (10.2, 4.0)), "走廊不连通：地图把通路堵死了"


# ---------------------------------------------------------------- 3. 动态层


class TestDynamicLayer:
    def test_blocks_and_clears(self, occ):
        target = (1.0, 1.0)
        i, j = occ.world2ij(*target)
        assert occ.is_free(i, j)

        from tr_planning.core.geometry import Circle

        occ.set_blocked(Circle(target[0], target[1], 0.2), is_dynamic=True)
        assert occ.is_free(i, j) is False, "动态层未生效"

        occ.clear_dynamic_blocked()
        assert occ.is_free(i, j), "清空动态层后未恢复"

    def test_dynamic_does_not_touch_static(self, occ):
        """清空动态层不能把静态障碍一起清掉（最容易写错的一处）。"""
        zone = UNMOVEABLE_ZONE[0]
        i, j = occ.world2ij(0.5 * (zone.start_point[0] + zone.end_point[0]), 4.0)
        assert occ.is_free(i, j) is False

        occ.clear_dynamic_blocked()
        assert occ.is_free(i, j) is False, "L1 被动态层的清空操作误删了"

    def test_dynamic_counts_into_ratio(self, occ):
        from tr_planning.core.geometry import Circle

        before = occ.occupied_ratio
        occ.set_blocked(Circle(1.0, 1.0, 0.3), is_dynamic=True)
        assert occ.occupied_ratio > before

    def test_dynamic_shape_mismatch_rejected(self, occ):
        with pytest.raises(ValueError):
            Occupancy(
                occupied_grid=occ.occupied_grid,
                resolution_m=occ.resolution_m,
                origin_m=occ.origin_m,
                padding_m=occ.padding_m,
                blocked_dynamic=np.zeros((2, 2), dtype=bool),
            )


# ---------------------------------------------------------------- 4. 分辨率无关


class TestResolutionIndependence:
    def test_occupied_ratio_is_stable(self, occ_factory):
        """占据比不应随分辨率明显变化（粗网格近似误差允许 2%）。"""
        ratios = [occ_factory(res).occupied_ratio for res in (0.05, 0.02, 0.01)]
        spread = max(ratios) - min(ratios)
        assert spread < 0.02, f"占据比随分辨率漂移 {spread:.4f}：{ratios}"

    def test_effective_padding_matches_declared_at_fine_res(
        self, measure_padding_band, robot
    ):
        """在细分辨率下，有效外扩应接近声明值（容差 1 格）。"""
        measured = measure_padding_band(0.01)
        assert measured is not None, "未能测到外扩边界，扫描逻辑或场地参数已变"
        assert abs(measured - robot.circumradius_m) <= 0.01, (
            f"有效外扩 {measured:.4f} m 与声明值 {robot.circumradius_m:.4f} m 相差过大"
        )


# ---------------------------------------------------------------- 已知缺陷


class TestKnownDefects:
    @pytest.mark.xfail(
        reason=(
            "已知缺陷：有效外扩量随分辨率变化。"
            "实测 res=0.05 -> 0.550 m，res=0.01 -> 0.500 m（声明值 0.495 m），"
            "差异 5 cm，远超 1 个细格。"
            "根因：Occupancy.set_blocked() 内部又做了一次 _expand(shape, resolution_m)，"
            "使有效外扩 = pad + res，随 res 漂移。"
        ),
        strict=False,
    )
    def test_effective_padding_is_resolution_independent(self, measure_padding_band):
        coarse = measure_padding_band(0.05)
        fine = measure_padding_band(0.01)
        assert coarse is not None and fine is not None
        assert abs(coarse - fine) <= 0.01, (
            f"外扩量随分辨率漂移：res=0.05 -> {coarse:.4f}，res=0.01 -> {fine:.4f}"
        )

    @pytest.mark.xfail(
        reason=(
            "已知缺陷：外扩量大到把所有格子涂黑时，build_occupancy 不报错，"
            "而是静默返回一张全占据的地图（后续搜索会得到'无解'而不明原因）。"
        ),
        strict=False,
    )
    def test_raises_when_no_free_cell_remains(self):
        huge = RobotParams(length_m=20.0, width_m=20.0)
        with pytest.raises(RuntimeError):
            build_occupancy(
                Field.make_field(FieldParams()), huge, OccupancyGrid(res_m=0.10)
            )

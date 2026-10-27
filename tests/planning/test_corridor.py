"""planning.corridor 的测试。

结构：
    1. 构造与校验          —— 现在全部应通过
    2. 半空间对象的形状/归一化 —— 现在应通过
    3. 几何语义（哪边是里）  —— 受已知缺陷影响，整类标记 xfail
    4. 区域一致性          —— 受已知缺陷影响，整类标记 xfail
    5. 形心与中心线        —— 不依赖半空间，应通过
    6. 已知缺陷            —— 单独列出，写明实测值与修法

修好 `_compile` 的法向后，被 xfail 的测试会变成 XPASS。
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon as SgPolygon

from tr_astar_test.planning.corridor import Corridor, Halfspaces, centerline

RECT = (0.0, 0.0, 1.0, 1.0)

# 法向符号 bug 的统一说明，避免每处重复
NORMAL_BUG = (
    "已知缺陷：_compile 里外法向的 y 分量符号写反。"
    "实测 from_rect(0,0,1,1) 得到 A=[[0,1],[1,0],[0,-1],[-1,0]]、b=[0,1,-1,0]，"
    "中心点 margin=-0.5、contains=False。正确外法向应为 (dy/L, -dx/L)，"
    "现为 (dy/L, +dx/L)，使半空间区域关于 x 轴镜像（竖直边正确，水平边朝外）。"
)


def rect() -> Corridor:
    return Corridor.from_rect(*RECT)


def region_matches_polygon(corridor: Corridor, samples_per_axis: int = 40) -> float:
    """半空间判定与 shapely 多边形判定的不一致比例（0 = 完全一致）。"""
    v = corridor.vertices
    x0, x1 = min(p[0] for p in v), max(p[0] for p in v)
    y0, y1 = min(p[1] for p in v), max(p[1] for p in v)
    poly = SgPolygon(v)

    xs = np.linspace(x0 - 0.2, x1 + 0.2, samples_per_axis)
    ys = np.linspace(y0 - 0.2, y1 + 0.2, samples_per_axis)
    mismatch = 0
    total = 0
    for x in xs:
        for y in ys:
            if corridor.contains(float(x), float(y)) != poly.covers(
                Point(float(x), float(y))
            ):
                mismatch += 1
            total += 1
    return mismatch / total


# ---------------------------------------------------------------- 1. 构造与校验


class TestConstruction:
    def test_from_rect_has_four_vertices(self):
        assert len(rect().vertices) == 4

    def test_from_rect_is_counter_clockwise(self):
        v = rect().vertices
        area = sum(
            v[i][0] * v[(i + 1) % 4][1] - v[(i + 1) % 4][0] * v[i][1] for i in range(4)
        )
        assert area > 0

    def test_from_rect_keeps_coordinates(self):
        c = Corridor.from_rect(2.5, 0.6, 8.5, 0.9)
        xs = [p[0] for p in c.vertices]
        ys = [p[1] for p in c.vertices]
        assert (min(xs), max(xs)) == (2.5, 8.5)
        assert (min(ys), max(ys)) == (0.6, 0.9)

    def test_vertex_coordinates_are_casted_to_float(self):
        c = Corridor(((0, 0), (1, 0), (1, 1), (0, 1)))
        assert all(
            isinstance(v[0], float) and isinstance(v[1], float) for v in c.vertices
        )

    def test_rejects_two_vertices(self):
        with pytest.raises(ValueError):
            Corridor(((0.0, 0.0), (1.0, 1.0)))

    def test_rejects_clockwise_order(self):
        with pytest.raises(ValueError):
            Corridor(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)))

    def test_rejects_concave(self):
        with pytest.raises(ValueError):
            Corridor(((0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (2.0, 2.0), (0.0, 2.0)))

    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError

        c = rect()
        with pytest.raises(FrozenInstanceError):
            c.vertices = ()  # type: ignore[misc]


# ---------------------------------------------------------------- 2. 半空间对象


class TestHalfspaceObject:
    def test_shapes(self):
        hs = rect().halfspaces
        assert hs.A.shape == (4, 2)
        assert hs.b.shape == (4,)

    def test_normals_are_unit_length(self):
        hs = rect().halfspaces
        assert np.allclose(np.linalg.norm(hs.A, axis=1), 1.0)

    def test_constraint_count_equals_vertex_count(self):
        tri = Corridor(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)))
        assert tri.halfspaces.A.shape[0] == 3

    def test_halfspaces_are_cached(self):
        c = rect()
        assert c.halfspaces is c.halfspaces

    def test_eq_is_disabled(self):
        """eq=False：应为对象同一性，而不是逐元素数组比较。"""
        hs = rect().halfspaces
        same = hs
        assert (hs == same) is True
        assert (hs == Halfspaces(hs.A.copy(), hs.b.copy())) is False

    def test_margin_returns_float_like(self):
        """margin 至少要是 float 或 np.float64（能参与比较与格式化）。"""
        m = rect().halfspaces.margin(0.5, 0.5)
        assert isinstance(m, float)
        assert f"{m:.2f}" == "-0.50" or f"{m:.2f}" == "0.50"


# ---------------------------------------------------------------- 3. 几何语义


@pytest.mark.xfail(reason=NORMAL_BUG, strict=False)
class TestHalfspaceSemantics:
    """内部点 margin > 0、外部点 margin < 0。"""

    def test_center_margin_is_positive(self):
        assert rect().halfspaces.margin(0.5, 0.5) == pytest.approx(0.5)

    def test_center_is_contained(self):
        assert rect().contains(0.5, 0.5) is True

    def test_interior_near_corner_is_contained(self):
        assert rect().contains(0.05, 0.05) is True

    def test_boundary_point_is_contained(self):
        assert rect().contains(1.0, 0.5) is True
        assert rect().contains(0.0, 0.0) is True

    def test_all_vertices_satisfy_own_halfspaces(self):
        c = rect()
        for x, y in c.vertices:
            assert c.halfspaces.margin(x, y) >= -1e-9

    def test_margin_equals_distance_to_nearest_edge(self):
        # 点 (0.7, 0.2) 到四边的距离：0.7 / 0.3 / 0.8 / 0.2 → 最小 0.2
        assert rect().halfspaces.margin(0.7, 0.2) == pytest.approx(0.2)


# ---------------------------------------------------------------- 4. 区域一致性


@pytest.mark.xfail(reason=NORMAL_BUG, strict=False)
class TestRegionConsistency:
    def test_region_matches_polygon(self):
        frac = region_matches_polygon(rect())
        assert frac < 0.02, f"{frac:.1%} 采样点判定不一致，半空间区域与多边形不符"

    def test_triangle_region_matches_polygon(self):
        tri = Corridor(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)))
        assert region_matches_polygon(tri) < 0.02

    def test_off_center_corridor_is_self_consistent(self):
        """不在 y=0 附近的走廊必须同样自洽（抓"只在原点附近才对"的错误）。"""
        c = Corridor.from_rect(2.5, 0.6, 8.5, 0.9)
        cx, cy = c.centroid
        assert c.contains(cx, cy) is True
        assert c.contains(cx, 0.75) is True
        assert c.contains(cx, 1.5) is False


# ---------------------------------------------------------------- 5. 形心与中心线


class TestCentroid:
    def test_centroid_is_vertex_mean(self):
        c = Corridor.from_rect(0.0, 0.0, 2.0, 4.0)
        assert c.centroid == pytest.approx((1.0, 2.0))

    def test_triangle_centroid(self):
        tri = Corridor(((0.0, 0.0), (3.0, 0.0), (0.0, 3.0)))
        assert tri.centroid == pytest.approx((1.0, 1.0))


class TestCenterline:
    def test_includes_start_and_end(self):
        c = Corridor.from_rect(1.0, 0.6, 2.0, 0.9)
        pts = centerline((c,), (0.5, 0.75), (2.5, 0.75))
        assert pts[0] == (0.5, 0.75)
        assert pts[-1] == (2.5, 0.75)

    def test_includes_each_corridor_centroid(self):
        c1 = Corridor.from_rect(1.0, 0.6, 2.0, 0.9)
        c2 = Corridor.from_rect(2.0, 0.6, 3.0, 0.9)
        pts = centerline((c1, c2), (0.5, 0.75), (3.5, 0.75))
        assert len(pts) == 4
        assert pts[1] == pytest.approx(c1.centroid)
        assert pts[2] == pytest.approx(c2.centroid)

    def test_returns_tuple(self):
        c = Corridor.from_rect(1.0, 0.6, 2.0, 0.9)
        assert isinstance(centerline((c,), (0.5, 0.75), (2.5, 0.75)), tuple)

    @pytest.mark.xfail(reason=NORMAL_BUG, strict=False)
    def test_all_points_inside_corridor(self):
        c = Corridor.from_rect(1.0, 0.6, 3.0, 0.9)
        for x, y in centerline((c,), (1.2, 0.75), (2.8, 0.75)):
            assert c.contains(x, y) is True


# ---------------------------------------------------------------- 6. 已知缺陷


class TestKnownDefects:
    @pytest.mark.xfail(
        reason=(
            "已知缺陷：contains() 返回 np.bool_ 而不是 Python bool。"
            "`distance >= 0` 得到 np.bool_，`or` 会原样返回它，"
            "导致 `x is True` 失败、json.dumps 也会失败（if 判断不受影响）。"
            "修法：return bool(distance >= 0 or is_near_zero(distance))。"
        ),
        strict=False,
    )
    def test_contains_returns_python_bool(self):
        hs = Halfspaces(np.array([[1.0, 0.0]]), np.array([0.0]))
        assert hs.contains(0.5, 0.0) is True
        assert isinstance(hs.contains(0.5, 0.0), bool)

    @pytest.mark.xfail(
        reason=(
            "已知缺陷：顺时针输入时错误信息来自 geometry.is_convex_ccw"
            "（'请传入按逆时针顺序排列的顶点'），而不是 corridor 自己的"
            "'走廊顶点必须按逆时针顺序排列'。根因：is_convex_ccw 在顺序错误时"
            "直接 raise，导致 __post_init__ 里那句 `if not is_convex_ccw(...)` 永不执行。"
        ),
        strict=False,
    )
    def test_clockwise_error_message_comes_from_corridor(self):
        with pytest.raises(ValueError, match="走廊顶点"):
            Corridor(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)))

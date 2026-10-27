"""core.geometry 的测试。

只测"几何性质"，不测实现方式：
    - 顶点顺序（是否逆时针、是否自交）
    - 包含判定（用 shapely 做独立参照，避免用被测代码自己验证自己）
    - 外扩（面积增大、包含关系保持）
    - 角度折算的边界
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon as SgPolygon

from tr_astar_test.core.geometry import Circle, Polygon, Pose2D, wrap2pi

# ---------------------------------------------------------------- 独立参照


def signed_area(vertices: tuple[tuple[float, float], ...]) -> float:
    """鞋带公式；正数表示逆时针。"""
    total = 0.0
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def shapely_contains(shape: Polygon | Circle, x: float, y: float) -> bool:
    if isinstance(shape, Circle):
        return math.hypot(x - shape.cx, y - shape.cy) <= shape.cr
    return SgPolygon(shape.vertices).covers(Point(x, y))


# ---------------------------------------------------------------- Polygon


class TestFromRect:
    def test_has_four_distinct_vertices(self):
        poly = Polygon.from_rect((2.5, 2.5), (8.5, 5.5))
        assert len(poly.vertices) == 4
        assert len(set(poly.vertices)) == 4, "矩形四角必须互不相同"

    def test_is_counter_clockwise(self):
        """顶点顺序必须使有向面积为正，否则光栅化会出现自交/空洞。"""
        poly = Polygon.from_rect((2.5, 2.5), (8.5, 5.5))
        assert signed_area(poly.vertices) > 0

    def test_bounding_box_matches_input(self):
        poly = Polygon.from_rect((1.0, 2.0), (4.0, 6.0))
        xs = [v[0] for v in poly.vertices]
        ys = [v[1] for v in poly.vertices]
        assert min(xs) == pytest.approx(1.0)
        assert max(xs) == pytest.approx(4.0)
        assert min(ys) == pytest.approx(2.0)
        assert max(ys) == pytest.approx(6.0)

    def test_area_equals_rect_area(self):
        poly = Polygon.from_rect((0.0, 0.0), (3.0, 2.0))
        assert abs(signed_area(poly.vertices)) == pytest.approx(6.0)


class TestPolygonValidation:
    def test_rejects_fewer_than_three_vertices(self):
        with pytest.raises(ValueError):
            Polygon(((0.0, 0.0), (1.0, 1.0)))

    def test_accepts_triangle(self):
        tri = Polygon(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
        assert len(tri.vertices) == 3


class TestPolygonExpand:
    """外扩的两个必须成立的性质：面积变大、原点仍在内。"""

    def test_area_grows(self):
        poly = Polygon.from_rect((0.0, 0.0), (2.0, 2.0))
        grown = poly.expand(0.5)
        assert abs(signed_area(grown.vertices)) > abs(signed_area(poly.vertices))

    def test_original_vertices_stay_inside(self):
        poly = Polygon.from_rect((2.5, 2.5), (8.5, 5.5))
        grown = poly.expand(0.5)
        for x, y in poly.vertices:
            assert shapely_contains(grown, x, y), f"原点 {(x, y)} 在外扩后跑到了外面"

    def test_offset_distance_along_axis(self):
        """沿坐标轴方向的偏移量应约等于 r（这才是"外扩 r"的定义）。"""
        poly = Polygon.from_rect((0.0, 0.0), (2.0, 2.0))
        r = 0.4
        grown = poly.expand(r)
        x_min = min(v[0] for v in grown.vertices)
        # shapely buffer 用圆弧近似，允许 2% 的数值误差
        assert x_min == pytest.approx(-r, abs=0.02)

    def test_interior_point_still_inside(self):
        poly = Polygon.from_rect((2.5, 2.5), (8.5, 5.5))
        grown = poly.expand(0.3)
        assert shapely_contains(grown, 5.5, 4.0)


# ---------------------------------------------------------------- Circle


class TestCircle:
    def test_expand_increases_radius(self):
        c = Circle(cx=1.0, cy=2.0, cr=0.1)
        grown = c.expand(0.5)
        assert grown.cx == pytest.approx(1.0)
        assert grown.cy == pytest.approx(2.0)
        assert grown.cr == pytest.approx(0.6)

    def test_contains(self):
        c = Circle(cx=0.0, cy=0.0, cr=1.0)
        assert shapely_contains(c, 0.5, 0.0)
        assert not shapely_contains(c, 1.5, 0.0)


# ---------------------------------------------------------------- 角度


class TestWrap2Pi:
    @pytest.mark.parametrize(
        "angle",
        [
            0.0,
            1.0,
            -1.0,
            math.pi,
            -math.pi,
            3.0 * math.pi,
            -3.0 * math.pi,
            100.0,
            -100.0,
        ],
    )
    def test_result_in_range(self, angle: float):
        wrapped = wrap2pi(angle)
        assert -math.pi - 1e-12 <= wrapped < math.pi + 1e-12

    @pytest.mark.parametrize("angle", [0.3, -0.3, 2.9, -2.9, 7.0])
    def test_idempotent(self, angle: float):
        once = wrap2pi(angle)
        assert wrap2pi(once) == pytest.approx(once)

    def test_preserves_equivalence_mod_2pi(self):
        angle = 5.0
        assert math.sin(wrap2pi(angle)) == pytest.approx(math.sin(angle))
        assert math.cos(wrap2pi(angle)) == pytest.approx(math.cos(angle))


class TestPose2D:
    def test_canonicalized_keeps_position(self):
        pose = Pose2D(1.0, 2.0, 3.0 * math.pi)
        out = pose.canonicalized()
        assert (out.x_m, out.y_m) == (1.0, 2.0)

    def test_canonicalized_wraps_theta(self):
        pose = Pose2D(0.0, 0.0, 3.0 * math.pi)
        out = pose.canonicalized()
        assert out.theta_rad == pytest.approx(wrap2pi(3.0 * math.pi))

    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError

        pose = Pose2D(0.0, 0.0, 0.0)
        with pytest.raises(FrozenInstanceError):
            pose.x_m = 1.0  # type: ignore[misc]

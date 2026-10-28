"""core.field 的测试。

Field 只描述"场地是什么"，不涉及栅格化。所以这里的断言都是几何/结构层面的：
    - 场地边界与参数一致
    - 所有"应当不可通行"的区域都被 obstacles 覆盖
    - 自定义参数能被正确采用
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon as SgPolygon

from tr_planning.core.field import Field
from tr_planning.core.geometry import Circle, Polygon
from tr_planning.core.params import UNMOVEABLE_ZONE, FieldParams


def covered_by(shapes: tuple[Polygon | Circle, ...], x: float, y: float) -> bool:
    """点 (x, y) 是否被任意一个形状覆盖（用 shapely 做独立判定）。"""
    for shape in shapes:
        if isinstance(shape, Circle):
            if math.hypot(x - shape.cx, y - shape.cy) <= shape.cr:
                return True
        elif SgPolygon(shape.vertices).covers(Point(x, y)):
            return True
    return False


class TestBound:
    def test_bounding_box_matches_params(self):
        params = FieldParams()
        field = Field.make_field(params)
        xs = [v[0] for v in field.bound.vertices]
        ys = [v[1] for v in field.bound.vertices]
        assert min(xs) == pytest.approx(params.start_point[0])
        assert max(xs) == pytest.approx(params.end_point[0])
        assert min(ys) == pytest.approx(params.start_point[1])
        assert max(ys) == pytest.approx(params.end_point[1])

    def test_bound_is_counter_clockwise(self):
        field = Field.make_field()
        verts = field.bound.vertices
        total = 0.0
        for i in range(len(verts)):
            x0, y0 = verts[i]
            x1, y1 = verts[(i + 1) % len(verts)]
            total += x0 * y1 - x1 * y0
        assert total > 0


class TestObstacles:
    def test_l1_is_covered(self):
        field = Field.make_field()
        zone = UNMOVEABLE_ZONE[0]
        cx = 0.5 * (zone.start_point[0] + zone.end_point[0])
        cy = 0.5 * (zone.start_point[1] + zone.end_point[1])
        assert covered_by(field.obstacles, cx, cy), "L1 高台中心未被任何 obstacles 覆盖"

    def test_stair_and_transfer_zone_is_covered(self):
        """阶梯/交接区必须显式出现在地图里。

        现阶段它按"不可通行"处理（尚未建模），但绝不能默默当成平地。
        """
        field = Field.make_field()
        zone = UNMOVEABLE_ZONE[1]
        cx = 0.5 * (zone.start_point[0] + zone.end_point[0])
        cy = 0.5 * (zone.start_point[1] + zone.end_point[1])
        assert covered_by(field.obstacles, cx, cy), "阶梯/交接区未被任何 obstacles 覆盖"

    def test_obstacles_non_empty(self):
        field = Field.make_field()
        assert len(field.obstacles) >= 2

    def test_keepout_is_separate_from_obstacles(self):
        """keepout 是语义禁入，不应与物理障碍混在一起。"""
        field = Field.make_field()
        assert isinstance(field.keepout, tuple)
        for shape in field.keepout:
            assert isinstance(shape, (Polygon, Circle))


class TestCustomParams:
    def test_custom_field_size_is_used(self):
        params = FieldParams(start_point=(0.0, 0.0), end_point=(6.0, 3.0))
        field = Field.make_field(params)
        xs = [v[0] for v in field.bound.vertices]
        ys = [v[1] for v in field.bound.vertices]
        assert max(xs) == pytest.approx(6.0)
        assert max(ys) == pytest.approx(3.0)

    def test_none_params_falls_back_to_default(self):
        field = Field.make_field(None)
        assert isinstance(field.params, FieldParams)


# ---------------------------------------------------------------- 已知缺陷


class TestKnownDefects:
    @pytest.mark.xfail(
        reason="已知缺陷：FieldParams.unmoveable_zones 是类属性而非 dataclass 字段，"
        "无法通过构造参数覆盖；且 make_field 读取模块级常量而不是 p.unmoveable_zones。",
        strict=False,
    )
    def test_unmoveable_zones_is_a_real_field(self):
        params = FieldParams(unmoveable_zones=UNMOVEABLE_ZONE[:1])
        assert params.unmoveable_zones == UNMOVEABLE_ZONE[:1]

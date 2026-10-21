"""场地的几何，只描述是什么，与栅格化解耦"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Circle, Polygon
from .params import UNMOVEABLE_ZONE, FieldParams


@dataclass(frozen=True)
class Field:
    params: FieldParams
    obstacles: tuple[Polygon | Circle, ...]  # 场地内障碍物
    keepout: tuple[Polygon | Circle, ...]  # 语义禁入区域，并非实际障碍物

    @property
    def bound(self) -> Polygon:
        """场地边界矩形"""
        return Polygon.from_rect(
            (self.params.start_point[0], self.params.start_point[1]),
            (self.params.end_point[0], self.params.end_point[1]),
        )

    @staticmethod
    def make_field(p: FieldParams | None = None) -> Field:
        """构造一个默认场地"""
        if p is None:
            p = FieldParams()

        layer1 = Polygon.from_rect(
            UNMOVEABLE_ZONE[0].start_point,
            UNMOVEABLE_ZONE[0].end_point,
        )
        box = None  # 动态静态障碍：单次场地仿真中不变
        keepout = []
        if box is not None:
            keepout.append(box)
        return Field(
            params=p,
            obstacles=(layer1,),
            keepout=tuple(keepout),
        )

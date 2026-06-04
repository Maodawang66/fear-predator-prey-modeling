"""统一的时间序列数据结构（供 ODE 拟合使用）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PredatorPreySeries:
    """观测到的捕食者—猎物时间序列。"""

    name: str
    t: np.ndarray
    prey: np.ndarray
    predator: np.ndarray
    time_unit: str = "year"
    prey_label: str = "prey"
    predator_label: str = "predator"
    source_path: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.prey = np.asarray(self.prey, dtype=float)
        self.predator = np.asarray(self.predator, dtype=float)
        if self.t.ndim != 1 or self.prey.shape != self.t.shape or self.predator.shape != self.t.shape:
            raise ValueError(f"{self.name}: t/prey/predator 长度须一致")
        if self.t.size < 4:
            raise ValueError(f"{self.name}: 至少需要 4 个时间点才能拟合")

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def n_points(self) -> int:
        return int(self.t.size)

    def scaled_copy(self, prey_scale: float | None = None, predator_scale: float | None = None) -> PredatorPreySeries:
        """按尺度归一化（便于数值积分），meta 中记录 scale 以便还原。"""
        ps = prey_scale or max(float(np.max(self.prey)), 1.0)
        qs = predator_scale or max(float(np.max(self.predator)), 1.0)
        return PredatorPreySeries(
            name=self.name,
            t=self.t.copy(),
            prey=self.prey / ps,
            predator=self.predator / qs,
            time_unit=self.time_unit,
            prey_label=self.prey_label,
            predator_label=self.predator_label,
            source_path=self.source_path,
            meta={
                **self.meta,
                "prey_scale": ps,
                "predator_scale": qs,
            },
        )

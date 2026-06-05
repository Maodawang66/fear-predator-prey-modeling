"""模型参数定义与预设情景。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class BaselineParams:
    """无恐惧基线：逻辑斯蒂猎物 + Holling II + 捕食者死亡。"""

    r: float = 1.0
    K: float = 100.0
    a: float = 0.04325
    theta: float = 0.0052
    e: float = 0.51
    mu: float = 0.4


@dataclass(frozen=True)
class FearMemoryParams(BaselineParams):
    """恐惧繁殖抑制 + 指数核记忆（Wang 类模型，辅助变量 M）。"""

    phi: float = 0.02
    delta: float = 1.0


@dataclass(frozen=True)
class FearSaturatingParams(BaselineParams):
    """饱和型恐惧繁殖抑制：r(1 - phi*y/(y+h))（实验可饱和响应）。"""

    phi: float = 0.8
    h: float = 8.0


@dataclass(frozen=True)
class FearForagingParams(BaselineParams):
    """觅食抑制型：有效攻击率 a/(1+psi*y)（Lima / Preisser 行为机制）。"""

    psi: float = 0.12


@dataclass(frozen=True)
class FearHandlingParams(BaselineParams):
    """警觉增大型：有效半饱和 theta*(1+psi*y)（处理时间延长）。"""

    psi: float = 0.15


@dataclass(frozen=True)
class BDAFearParams:
    """
    Beddington-DeAngelis + Wang 恐惧因子（Myint et al. 2025, arXiv:2506.22070）。
    无量纲局部动力学（无扩散项）；u=猎物，v=捕食者。
    """

    r: float = 2.5
    d: float = 0.5
    a: float = 0.1
    k: float = 0.08
    p: float = 1.0
    q: float = 0.1
    c: float = 0.8
    m: float = 0.6


class MechanismId(str, Enum):
    """文献机制与代码映射。"""

    BASELINE = "baseline"
    FEAR_MEMORY = "fear_memory"
    FEAR_INSTANT = "fear_instant"
    FEAR_SATURATING = "fear_saturating"
    FEAR_FORAGING = "fear_foraging"
    FEAR_HANDLING = "fear_handling"
    BDA_BASELINE = "bda_baseline"
    BDA_FEAR = "bda_fear"


MECHANISM_LABELS: dict[MechanismId, str] = {
    MechanismId.BASELINE: "Baseline (no fear)",
    MechanismId.FEAR_MEMORY: "Reproduction suppression + memory (Wang/MacDonald)",
    MechanismId.FEAR_INSTANT: "Reproduction suppression, no memory",
    MechanismId.FEAR_SATURATING: "Saturating reproduction suppression (Zanette-type)",
    MechanismId.FEAR_FORAGING: "Foraging suppression (Lima/Preisser-type)",
    MechanismId.FEAR_HANDLING: "Alertness / handling-time extension",
    MechanismId.BDA_BASELINE: "B-D baseline (k=0)",
    MechanismId.BDA_FEAR: "B-D + fear 1/(1+kv) (Myint 2025)",
}


baseline_default = BaselineParams()
fear_default = FearMemoryParams(phi=0.02, delta=1.0)
fear_high = FearMemoryParams(phi=0.045, delta=0.4)
fear_saturating_default = FearSaturatingParams(phi=0.75, h=8.0)
fear_foraging_default = FearForagingParams(psi=0.12)
fear_handling_default = FearHandlingParams(psi=0.15)
bda_fear_default = BDAFearParams()
bda_no_fear_default = BDAFearParams(k=0.0)

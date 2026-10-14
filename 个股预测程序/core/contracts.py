# -*- coding: utf-8 -*-
"""
数据契约层（Data Contracts）—— 全项目唯一的模块间接口约定。

设计原则
--------
1. **零依赖**：本模块只依赖标准库（numpy 为可选），因此任何模块都可以安全导入它。
2. **因子化**：所有分析模块（技术面 / 基本面 / 资金面 / 情绪面 / 宏观 / 消息面 /
   周期 / 理论模型）都只做一件事——把自己看到的信息压缩成一批 :class:`Factor`
   对象，装进 :class:`AnalysisBundle`。预测引擎完全不关心因子从哪来。
3. **方向语义统一**：``direction`` ∈ {+1 看涨, -1 看跌, 0 中性}，
   ``strength`` ∈ [0,1] 为方向上的强度，``reliability`` ∈ [0,1] 为"我对这条
   因子有多大把握"（数据缺失、样本不足、信号冲突时下调）。
4. **可降级**：任何模块拿不到数据都必须返回空列表或带 ``reliability`` 折扣的
   因子，并在 ``notes`` 中说明，**绝不允许抛异常打断整体分析**。
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import math
from typing import Any

try:  # numpy 可选：契约层在纯标准库环境下也要能用
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

__all__ = [
    "DIR_BULL", "DIR_BEAR", "DIR_NEUTRAL",
    "Category", "Factor", "AnalysisBundle", "ProbabilityForecast",
    "PredictionResult", "HorizonSpec", "HORIZONS", "HORIZON_KEYS",
    "horizon_spec", "clamp", "safe_float", "pct_change", "now_cn",
    "FactorSet",
]

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
DIR_BULL = 1
DIR_BEAR = -1
DIR_NEUTRAL = 0


class Category:
    """因子大类（与 :mod:`core.config` 的权重矩阵一一对应）。"""

    TECH = "technical"        # 技术面
    FUND = "fundamental"      # 基本面
    CAPITAL = "capital"       # 资金面与主力动向
    SENTIMENT = "sentiment"   # 市场情绪与博弈心理
    MACRO = "macro"           # 宏观与行业环境
    NEWS = "news"             # 消息面与突发事件
    CYCLE = "cycle"           # 时间周期与季节性
    THEORY = "theory"         # 金融理论模型（道氏/波浪/江恩/量价时空…）
    STAGE = "stage"           # 个股特性、所处阶段、定价权、量化占比
    ML = "ml"                 # 机器学习模型输出

    ALL = (TECH, FUND, CAPITAL, SENTIMENT, MACRO, NEWS, CYCLE, THEORY, STAGE, ML)

    LABEL = {
        TECH: "技术面",
        FUND: "基本面",
        CAPITAL: "资金面/主力",
        SENTIMENT: "情绪/博弈",
        MACRO: "宏观/行业",
        NEWS: "消息/事件",
        CYCLE: "周期/季节",
        THEORY: "理论模型",
        STAGE: "个股特性/阶段",
        ML: "AI模型",
    }


def _clean_display(v: Any, dash: str = "—") -> Any:
    """清理因子的展示值：把 None/NaN/inf 统一换成"—"。

    因子值多数直接取自数据源，取不到时是 None 或 NaN。界面与因子表会把它
    原样打印（例如"PEG 估值 → None"），用户会误以为是程序出错。
    这里在 Factor 构造时统一处理；字典/列表则递归清理内部值。
    同时把嵌套值里的 None/NaN 也换掉，避免出现 {'历史分位': nan} 这种字样。
    """
    if v is None:
        return dash
    if isinstance(v, float):
        return dash if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(v, dict):
        return {k: _clean_display(x, dash) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_display(x, dash) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in ("nan", "none", "null", "inf", "-inf"):
            return dash
        return v
    return v


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """把 x 限制在 [lo, hi]；NaN/None 一律返回 lo。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return lo
    if math.isnan(v) or math.isinf(v):
        return lo
    return lo if v < lo else (hi if v > hi else v)


def safe_float(x: Any, default: float = float("nan")) -> float:
    """尽力把任意对象转成 float，失败返回 default（不抛异常）。"""
    if x is None:
        return default
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    s = str(x).strip().replace(",", "").replace("%", "").replace("--", "")
    if not s or s in {"-", "None", "nan", "NaN", "null"}:
        return default
    mult = 1.0
    if s.endswith("万亿"):
        s, mult = s[:-2], 1e12
    elif s.endswith("亿"):
        s, mult = s[:-1], 1e8
    elif s.endswith("万"):
        s, mult = s[:-1], 1e4
    try:
        v = float(s) * mult
    except ValueError:
        return default
    return default if (math.isnan(v) or math.isinf(v)) else v


def pct_change(new: float, old: float) -> float:
    """(new-old)/|old|*100，old 为 0/NaN 时返回 NaN（百分比数值，非小数）。"""
    try:
        n, o = float(new), float(old)
    except (TypeError, ValueError):
        return float("nan")
    if o == 0 or math.isnan(o) or math.isnan(n):
        return float("nan")
    return (n - o) / abs(o) * 100.0


def now_cn() -> _dt.datetime:
    """当前北京时间（不依赖时区库，使用 UTC+8 偏移）。"""
    return _dt.datetime.utcnow() + _dt.timedelta(hours=8)


# --------------------------------------------------------------------------- #
# 预测周期定义
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class HorizonSpec:
    """一个预测周期的元信息。

    Attributes
    ----------
    key      : 唯一标识，如 ``"d1"`` / ``"m6"``。
    label    : 中文名，如 ``"未来1天"``。
    days     : 自然日跨度（用于日历效应/事件窗口判断）。
    bars     : 交易日跨度（用于技术面/收益计算），季度≈60、半年≈120、一年≈244。
    kind     : ``"short"`` / ``"medium"`` / ``"long"``，决定权重模板与模型选择。
    vol_mult : 该周期的典型波动放大倍数（相对日波动），用于概率区间换算。
    """

    key: str
    label: str
    days: int
    bars: int
    kind: str
    vol_mult: float


HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec("d1", "未来1天", 1, 1, "short", 1.0),
    HorizonSpec("d5", "未来5天", 5, 5, "short", 2.2),
    HorizonSpec("d10", "未来10天", 10, 10, "short", 3.1),
    HorizonSpec("d20", "未来20天(约1个月)", 20, 20, "medium", 4.4),
    HorizonSpec("m1", "未来1个月", 30, 21, "medium", 4.6),
    HorizonSpec("q1", "未来1个季度", 91, 60, "medium", 7.8),
    HorizonSpec("h1", "未来半年", 182, 120, "long", 11.0),
    HorizonSpec("y1", "未来1年", 365, 244, "long", 15.6),
)

HORIZON_KEYS: tuple[str, ...] = tuple(h.key for h in HORIZONS)
_HORIZON_MAP = {h.key: h for h in HORIZONS}


def horizon_spec(key: str) -> HorizonSpec:
    """按 key 取周期定义；未知 key 抛 KeyError。"""
    if key not in _HORIZON_MAP:
        raise KeyError("未知预测周期: %r，可选: %s" % (key, ", ".join(HORIZON_KEYS)))
    return _HORIZON_MAP[key]


# --------------------------------------------------------------------------- #
# 因子
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Factor:
    """一条可解释的分析因子。

    这是全项目最重要的数据结构：任何一个模块的任何一个观察，最终都要变成它。

    Parameters
    ----------
    id : str
        稳定英文标识（下划线命名），如 ``"ma_bull_alignment"``。用于权重表匹配与调试。
    name : str
        中文短名，如 ``"均线多头排列"``。
    category : str
        :class:`Category` 之一。
    direction : int
        +1 / -1 / 0。
    strength : float
        方向上的强度 0~1。0.3≈弱信号，0.6≈中等，0.9≈极强。
    reliability : float
        可信度 0~1。数据完整、样本充分、逻辑直接 → 高；数据缺失、样本<20、间接推断 → 低。
    value : Any
        该因子的原始观测值（数字或字符串），用于报告展示，如 ``"MA5=10.2>MA20=9.8"``。
    detail : str
        一句话中文解释，说明这个信号为什么看涨/看跌，会直接展示给用户。
    theory : str
        理论依据标签，如 ``"道氏理论"`` / ``"MACD"`` / ``"量价关系"``。
    horizon : tuple[str, ...] | None
        该因子主要作用的周期 key 列表；``None`` 表示对所有周期生效。
        例如日线 MACD 金叉对 ``("d1","d5","d10")`` 更有效。
    weight_multiplier : float
        单因子权重微调（默认 1.0），用于"某因子在当下环境特别灵"的场景。
    """

    id: str
    name: str
    category: str
    direction: int
    strength: float
    reliability: float = 0.6
    value: Any = None
    detail: str = ""
    theory: str = ""
    horizon: tuple | None = None
    weight_multiplier: float = 1.0

    def __post_init__(self) -> None:
        self.direction = int(0 if self.direction is None else self.direction)
        if self.direction > 0:
            self.direction = 1
        elif self.direction < 0:
            self.direction = -1
        self.strength = clamp(self.strength)
        self.reliability = clamp(self.reliability)
        try:
            self.weight_multiplier = float(self.weight_multiplier)
        except (TypeError, ValueError):
            self.weight_multiplier = 1.0
        if self.horizon is not None:
            self.horizon = tuple(self.horizon)
        # 展示值清理：因子值常直接来自数据源，取不到时会是 None/NaN。
        # 这些值会被界面与因子表原样打印，出现"PEG 估值 → None"这种字样，
        # 既难看又让用户以为是程序 bug，所以在源头统一转成"—"。
        self.value = _clean_display(self.value)

    # -- 派生量 ----------------------------------------------------------- #
    @property
    def signed(self) -> float:
        """带方向的强度：看涨为正、看跌为负，范围 [-1, 1]。"""
        return self.direction * self.strength

    def effective(self, weight: float = 1.0) -> float:
        """加权后的有效贡献 = 方向 × 强度 × 可信度 × 权重。"""
        return self.direction * self.strength * self.reliability * float(weight)

    def applies_to(self, horizon_key: str) -> bool:
        """该因子是否作用于给定周期。"""
        return self.horizon is None or horizon_key in self.horizon

    @property
    def is_actionable(self) -> bool:
        """是否值得展示（有方向且强度不为 0）。"""
        return self.direction != 0 and self.strength > 0

    def to_row(self) -> dict:
        """转成扁平 dict，供 DataFrame / 报告使用。"""
        return {
            "因子": self.name,
            "id": self.id,
            "类别": Category.LABEL.get(self.category, self.category),
            "方向": "看涨" if self.direction > 0 else ("看跌" if self.direction < 0 else "中性"),
            "强度": round(self.strength, 3),
            "可信度": round(self.reliability, 3),
            "观测值": self.value,
            "解读": self.detail,
            "理论依据": self.theory,
            "适用周期": "全部" if self.horizon is None else "/".join(self.horizon),
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        d = {1: "多", -1: "空", 0: "中"}[self.direction]
        return "<Factor %s %s %s s=%.2f r=%.2f>" % (
            self.id, self.name, d, self.strength, self.reliability)


class FactorSet(list):
    """:class:`Factor` 的列表，带若干统计便利方法。"""

    def add(self, factor: "Factor | None") -> "FactorSet":
        if factor is not None:
            self.append(factor)
        return self

    def extend_factors(self, factors) -> "FactorSet":
        if factors:
            self.extend([f for f in factors if f is not None])
        return self

    def by_category(self, category: str) -> "FactorSet":
        return FactorSet(f for f in self if f.category == category)

    def actionable(self) -> "FactorSet":
        return FactorSet(f for f in self if f.is_actionable)

    @property
    def net(self) -> float:
        """未加权的净方向分（sum of signed）。"""
        return sum(f.signed for f in self if f.is_actionable)

    def top(self, n: int = 10, by: str = "strength") -> "FactorSet":
        key = (lambda f: abs(f.signed)) if by == "strength" else (lambda f: f.reliability)
        return FactorSet(sorted(self, key=key, reverse=True)[:n])

    def to_frame(self):  # 延迟导入 pandas，保持契约层轻量
        import pandas as pd

        rows = [f.to_row() for f in self]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["因子", "id", "类别", "方向", "强度", "可信度", "观测值", "解读", "理论依据", "适用周期"])


# --------------------------------------------------------------------------- #
# 分析结果容器
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class AnalysisBundle:
    """一次完整分析的全部产物。

    ``factors`` 是唯一的"通用货币"；其余字段是给报告/图表使用的富数据，
    各模块按需填充，缺失即为 None / 空。
    """

    symbol: str = ""
    name: str = ""
    as_of: str = ""                       # 数据截止日 YYYY-MM-DD
    factors: FactorSet = dataclasses.field(default_factory=FactorSet)

    # ---- 富数据（供图表与报告） ---------------------------------------- #
    ohlcv: Any = None                     # pd.DataFrame[date,open,high,low,close,volume,amount]
    kline_weekly: Any = None
    kline_monthly: Any = None
    indicators: dict = dataclasses.field(default_factory=dict)   # 指标名 -> 序列/值
    snapshot: dict = dataclasses.field(default_factory=dict)     # 个股现状：价、量、市值、换手…
    stage: dict = dataclasses.field(default_factory=dict)        # 所处阶段/定价权/量化占比
    valuation: dict = dataclasses.field(default_factory=dict)
    financials: Any = None                                       # pd.DataFrame 财务摘要
    capital: dict = dataclasses.field(default_factory=dict)
    sentiment: dict = dataclasses.field(default_factory=dict)
    macro: dict = dataclasses.field(default_factory=dict)
    news: list = dataclasses.field(default_factory=list)         # [{title,date,source,url,tag,sentiment}]
    seasonality: dict = dataclasses.field(default_factory=dict)
    theory: dict = dataclasses.field(default_factory=dict)
    index_data: Any = None                # 大盘/行业指数 DataFrame
    market: dict = dataclasses.field(default_factory=dict)
    ml: dict = dataclasses.field(default_factory=dict)
    charts: dict = dataclasses.field(default_factory=dict)       # 名称 -> matplotlib Figure

    notes: list = dataclasses.field(default_factory=list)        # 数据源说明/降级提示
    errors: list = dataclasses.field(default_factory=list)       # 各模块异常文本（不中断流程）

    # ---- 便利方法 ------------------------------------------------------- #
    def note(self, msg: str) -> None:
        if msg and msg not in self.notes:
            self.notes.append(msg)

    def error(self, where: str, exc: BaseException) -> None:
        self.errors.append("%s: %s: %s" % (where, type(exc).__name__, exc))

    def add(self, *factors) -> None:
        for f in factors:
            if isinstance(f, (list, tuple, FactorSet)):
                self.factors.extend_factors(f)
            elif f is not None:
                self.factors.append(f)

    @property
    def is_empty(self) -> bool:
        return not self.factors

    def stats(self) -> dict:
        st = {"n_factors": len(self.factors), "n_bull": 0, "n_bear": 0, "n_neutral": 0,
              "by_category": {}}
        for f in self.factors:
            if f.direction > 0:
                st["n_bull"] += 1
            elif f.direction < 0:
                st["n_bear"] += 1
            else:
                st["n_neutral"] += 1
            lab = Category.LABEL.get(f.category, f.category)
            slot = st["by_category"].setdefault(lab, {"看涨": 0, "看跌": 0, "中性": 0})
            slot["看涨" if f.direction > 0 else ("看跌" if f.direction < 0 else "中性")] += 1
        return st


# --------------------------------------------------------------------------- #
# 预测输出
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ProbabilityForecast:
    """单个周期的概率化预测结果。"""

    horizon: str                       # 周期 key
    label: str                         # 中文周期名
    bars: int = 0
    days: int = 0

    p_up: float = 0.5                  # P(区间收益 > 0)
    p_down: float = 0.5                # P(区间收益 < 0) = 1 - p_up
    p_flat: float = 0.0                # P(|收益| < 阈值)
    p_big_up: float = 0.0              # P(收益 > 阈值)   阈值见 big_move_threshold
    p_big_down: float = 0.0

    exp_return: float = 0.0            # 期望收益率（%）
    exp_return_low: float = 0.0        # 乐观情形（分位 ~80%）
    exp_return_high: float = 0.0       # 悲观情形（分位 ~20%）
    ci_low: float = 0.0                # 置信区间下沿（%）
    ci_high: float = 0.0               # 置信区间上沿（%）
    sigma: float = 0.0                 # 该周期的预测波动率（%）
    big_move_threshold: float = 5.0    # "大涨/大跌"的阈值（%）

    trend: str = "震荡"                # 看涨 / 看跌 / 震荡
    confidence: float = 0.5            # 模型自评置信度 0~1
    strength_score: float = 0.0        # 综合得分（-100~100）
    drivers: list = dataclasses.field(default_factory=list)    # 主要驱动因子（dict）
    counter: list = dataclasses.field(default_factory=list)    # 反向因子（风险点）
    model_mix: dict = dataclasses.field(default_factory=dict)  # 各子模型贡献
    note: str = ""

    def __post_init__(self) -> None:
        self.p_up = clamp(self.p_up, 0.0, 1.0)
        self.p_down = clamp(1.0 - self.p_up, 0.0, 1.0)
        self.confidence = clamp(self.confidence, 0.0, 1.0)
        self.p_big_up = clamp(self.p_big_up, 0.0, 1.0)
        self.p_big_down = clamp(self.p_big_down, 0.0, 1.0)
        self.p_flat = clamp(self.p_flat, 0.0, 1.0)

    @property
    def score(self) -> float:
        """-1~1 的方向分，= p_up*2-1。"""
        return self.p_up * 2.0 - 1.0

    def to_row(self) -> dict:
        return {
            "周期": self.label,
            "结论": self.trend,
            "上涨概率": "%.1f%%" % (self.p_up * 100),
            "下跌概率": "%.1f%%" % (self.p_down * 100),
            "期望涨跌幅": "%+.2f%%" % self.exp_return,
            "乐观": "%+.2f%%" % self.exp_return_high,
            "悲观": "%+.2f%%" % self.exp_return_low,
            "波动区间": "%+.1f%% ~ %+.1f%%" % (self.ci_low, self.ci_high),
            "大涨概率": "%.1f%%" % (self.p_big_up * 100),
            "大跌概率": "%.1f%%" % (self.p_big_down * 100),
            "置信度": "%.0f%%" % (self.confidence * 100),
        }

    def summary(self) -> str:
        return "%s：%s｜涨 %.0f%% / 跌 %.0f%%｜期望 %+.2f%%（区间 %+.1f%%~%+.1f%%）" % (
            self.label, self.trend, self.p_up * 100, self.p_down * 100,
            self.exp_return, self.ci_low, self.ci_high)


@dataclasses.dataclass
class PredictionResult:
    """一次预测任务的完整输出。"""

    symbol: str = ""
    name: str = ""
    as_of: str = ""
    price: float = float("nan")
    forecasts: list = dataclasses.field(default_factory=list)   # list[ProbabilityForecast]
    ensemble: dict = dataclasses.field(default_factory=dict)    # 综合投资建议/评分
    bundle: AnalysisBundle | None = None
    meta: dict = dataclasses.field(default_factory=dict)

    def by_key(self, key: str) -> "ProbabilityForecast | None":
        for f in self.forecasts:
            if f.horizon == key:
                return f
        return None

    def to_frame(self):
        import pandas as pd

        rows = [f.to_row() for f in self.forecasts]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def headline(self) -> str:
        if not self.forecasts:
            return "%s：暂无预测" % (self.name or self.symbol)
        f = self.by_key("d20") or self.forecasts[0]
        return "%s(%s) 现价 %.2f｜%s" % (self.name or "", self.symbol, self.price, f.summary())


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    fs = FactorSet()
    fs.append(Factor("ma_bull", "均线多头", Category.TECH, DIR_BULL, 0.8, 0.9,
                     value="MA5>MA20>MA60", detail="短中期均线依次向上", theory="道氏理论"))
    fs.append(Factor("rsi_ob", "RSI超买", Category.TECH, DIR_BEAR, 0.5, 0.7,
                     value=82.3, detail="RSI 进入超买区", theory="超买超卖"))
    b = AnalysisBundle(symbol="600519", name="贵州茅台", as_of="2026-10-09", factors=fs)
    print("HORIZONS:", [h.key for h in HORIZONS])
    print("市值测试 safe_float('1.23万亿') =", safe_float("1.23万亿"))
    print("stats:", b.stats())
    print("net:", b.factors.net)
    f = ProbabilityForecast(horizon="d20", label="未来20天", p_up=0.63, exp_return=2.4,
                            ci_low=-3.1, ci_high=7.9, confidence=0.58)
    print(f.summary())
    print(PredictionResult(symbol="600519", name="贵州茅台", price=1263.0,
                           forecasts=[f]).headline())
    print("contracts self-test OK")

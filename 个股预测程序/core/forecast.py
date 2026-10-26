# -*- coding: utf-8 -*-
"""
多周期概率预测引擎。

核心思路（三层结构）
--------------------
**第一层 · 因子汇总（Factor → 方向分）**
    每个分析模块把自己的观察压缩成 :class:`Factor`。对给定周期 ``h``：

    .. code-block:: text

        score_h = Σ_i  dir_i × strength_i × reliability_i × w(i, h)
                  ─────────────────────────────────────────────────
                              Σ_i  reliability_i × w(i, h)

    分子是"带符号的加权证据"，分母做归一化，因此 ``score_h ∈ [-1, 1]``。
    权重 ``w(i,h)`` 由 :mod:`core.config` 的权重矩阵决定（短线重资金/技术，
    长线重基本面/宏观），单因子还可以精调。

**第二层 · 概率化（方向分 → 概率）**
    用 logit 变换把有界的分数映射到实数域，再乘以周期相关的斜率系数：

    .. code-block:: text

        logit(p) = logit(p_prior) + k_h × score_h × conf_h

    其中 ``k_h`` 是"该周期对方向分的敏感度"（短线噪声大 → 斜率小，
    中线趋势可辨识度高 → 斜率大），``conf_h`` 是证据充分度
    （有效因子数量 × 平均可信度），避免"只有 2 个因子也敢给 90% 概率"。

**第三层 · 幅度与区间（概率 → 收益分布）**
    把方向分换算成标准正态分位，再乘以该周期的预测波动率：

    .. code-block:: text

        σ_h   = σ_daily × √bars_h × 周期校准系数
        E[r_h]= z × σ_h × λ        （λ 为"收益可预测性"折扣，通常 0.35~0.55）
        CI    = E[r_h] ± 0.9σ_h

    这样"涨跌概率"和"涨跌幅度"天然自洽：概率越大，期望幅度越大，
    且区间始终覆盖合理的波动范围（不会出现"看涨 90% 但幅度 ±0.1%"的荒谬结果）。

**多周期一致性**
    短中长周期结论若一致 → 上调置信度；若冲突 → 标注为"周期背离"，
    并在报告中提示"短期反弹但中期趋势未改"这类结构性判断。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .contracts import (
    HORIZONS, Category, Factor, FactorSet, HorizonSpec, ProbabilityForecast,
    PredictionResult, clamp, horizon_spec,
)
from .utils import get_logger, logit, norm_cdf, norm_ppf, sigmoid

log = get_logger()
_EPS = 1e-6


# =========================================================================== #
# 一、因子汇总
# =========================================================================== #
@dataclass
class Aggregated:
    """某个周期的因子汇总结果。"""

    horizon: str
    score: float = 0.0                    # -1 ~ 1
    raw_score: float = 0.0                # 未按可信度归一化的加权和
    weight_sum: float = 0.0
    n_factors: int = 0
    n_bull: int = 0
    n_bear: int = 0
    avg_reliability: float = 0.0
    evidence: float = 0.0                 # 证据充分度 0~1
    category_scores: dict = field(default_factory=dict)   # 大类 -> 带符号加权分
    by_category: dict = field(default_factory=dict)       # 大类 -> 因子数
    contributions: list = field(default_factory=list)     # [(Factor, weight, contribution)]

    @property
    def score_pct(self) -> float:
        """便于展示的 -100 ~ 100 分。"""
        return self.score * 100.0


def aggregate_factors(factors, horizon_key: str) -> Aggregated:
    """把因子集合按指定周期的权重矩阵汇总成方向分。

    Parameters
    ----------
    factors : iterable[Factor] | FactorSet
    horizon_key : 周期 key（"d1"... "y1"）

    Returns
    -------
    Aggregated
    """
    spec = horizon_spec(horizon_key)
    agg = Aggregated(horizon=horizon_key)
    # 类内因子计数：用于同类因子递减权重（防同一信息被重复计权）
    cat_seen: dict = {}

    for f in factors or []:
        if not isinstance(f, Factor) or not f.is_actionable:
            continue
        if not f.applies_to(horizon_key):
            continue
        # 剔除「空因子占位」：强度≈0 且可信度很低时，该因子不携带任何信息，
        # 却仍会占用 n_factors 与 weight_sum，稀释真实信号的权重
        # （AI 核对指出「散户情绪/股吧/搜索热度 强0.00 信0.15-0.20 却保留」）。
        if abs(float(f.strength or 0.0)) <= 0.001 and float(f.reliability or 0) < 0.20:
            continue
        # 反向指标：登记在案的因子需要翻转方向（散户情绪等）
        direction = f.direction
        if f.id in config.CONTRARIAN_FACTOR_IDS:
            direction = -direction
        w = config.factor_weight(horizon_key, spec.kind, f.id, f.category)
        w *= f.weight_multiplier
        # ---- 类内递减权重（防同一信息被重复计权）----
        # 同类别第 n 条因子 × 1/√n（1.0 / 0.71 / 0.58…）。
        # 宏观类曾有 8 条都含趋势位置信息、sentiment 类 3 条都反映板块热点、
        # technical 类 3 条都描述波段形态，简单相加等于给同一证据加了多倍权重。
        # 递减只压重复，不改方向。
        _n = cat_seen.get(f.category, 0) + 1
        cat_seen[f.category] = _n
        if _n > 1:
            w *= 1.0 / (float(_n) ** 0.5)
        if spec.kind == "long" and f.category == Category.FUND:
            w *= config.FORECAST.get("long_horizon_fundamental_boost", 1.0)
        rel = max(f.reliability, 0.05)
        # ---- 有效强度归一化（兜住"构造后被改写 strength"的因子）----
        # Factor.__post_init__ 已做过封顶，但部分模块在构造后直接改写
        # .strength，绕过了封顶（AI 核对图2：方向=中却强度 0.95）。
        # 这里在聚合前再统一压一次，保证：中性≤0.25、非中性≤可信度+0.25。
        _st = float(f.strength or 0.0)
        if direction == 0:
            _st = min(_st, 0.25)
        else:
            _st = min(_st, max(0.15, rel + 0.25))
        contribution = direction * _st * rel * w
        agg.raw_score += contribution
        agg.weight_sum += rel * w
        agg.n_factors += 1
        if direction > 0:
            agg.n_bull += 1
        elif direction < 0:
            agg.n_bear += 1
        agg.avg_reliability += rel
        agg.contributions.append((f, w, contribution))
        agg.by_category[f.category] = agg.by_category.get(f.category, 0) + 1
        agg.category_scores[f.category] = agg.category_scores.get(f.category, 0.0) + contribution

    if agg.n_factors:
        agg.avg_reliability /= agg.n_factors
        agg.score = clamp(agg.raw_score / max(agg.weight_sum, 1e-9), -1.0, 1.0)
        # 证据充分度：因子条数 + 平均可信度 + 大类覆盖广度
        n_eff = min(agg.n_factors / 25.0, 1.0)
        rel_eff = min(agg.avg_reliability / 0.7, 1.0)
        breadth = min(len(agg.by_category) / 6.0, 1.0)
        agg.evidence = clamp(0.45 * n_eff + 0.35 * rel_eff + 0.20 * breadth)
    return agg


# =========================================================================== #
# 二、波动率估计
# =========================================================================== #
def estimate_volatility(df: pd.DataFrame, span: int = 60) -> dict:
    """估计日波动率及其结构特征。

    Returns
    -------
    dict(daily_pct, annual_pct, vol_20, vol_60, vol_ratio, regime, atr_pct, sigma_floor)
    """
    out = {"daily_pct": float("nan"), "annual_pct": float("nan"), "vol_20": float("nan"),
           "vol_60": float("nan"), "vol_ratio": 1.0, "regime": "normal",
           "atr_pct": float("nan"), "sigma_floor": 0.8}
    if df is None or len(df) < 25 or "close" not in df.columns:
        return out
    c = df["close"].astype(float).to_numpy()
    r = np.diff(np.log(np.maximum(c, 1e-9)))
    if len(r) < 20:
        return out
    w = 2.0 / (span + 1.0)
    var = float(np.var(r[:20], ddof=1))
    for x in r[20:]:
        var = (1 - w) * var + w * x * x
    ann = math.sqrt(244.0)
    out["daily_pct"] = math.sqrt(max(var, 0)) * 100.0
    out["annual_pct"] = out["daily_pct"] * ann
    out["vol_20"] = float(np.std(r[-20:], ddof=1)) * 100.0
    out["vol_60"] = float(np.std(r[-60:], ddof=1)) * 100.0 if len(r) >= 60 else out["vol_20"]
    out["vol_ratio"] = out["vol_20"] / max(out["vol_60"], 1e-9)
    if "high" in df.columns and "low" in df.columns and len(df) >= 15:
        h = df["high"].astype(float).to_numpy()
        lo = df["low"].astype(float).to_numpy()
        tr = np.maximum(h[1:] - lo[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))
        out["atr_pct"] = float(np.mean(tr[-14:])) / max(c[-1], 1e-9) * 100.0
    ap = out["annual_pct"]
    if ap == ap:
        if ap > config.STAGE_RULES["high_vol_annual_pct"]:
            out["regime"] = "high"
        elif ap < config.STAGE_RULES["low_vol_annual_pct"]:
            out["regime"] = "low"
    # 波动率下限：避免极低波动股给出荒谬的窄区间
    dp = out["daily_pct"]
    out["sigma_floor"] = 0.55 if dp != dp else max(0.55, min(dp, 6.0) * 0.45)
    return out


def horizon_sigma(vol: dict, horizon_key: str, sigma_hist: float | None = None,
                  df: pd.DataFrame | None = None) -> float:
    """给定周期的预测波动率（%）。

    三路估计加权，避免"用当期波动率外推长期"这一常见错误：

    1. **理论扩散**：``σ_日 × √(交易日数)`` —— 只反映**当前**波动状态，
       低波动期会严重低估长周期的真实不确定性；
    2. **历史分布**（最可靠）：直接统计该股过去 N 年**每个周期的真实区间收益标准差**，
       它天然包含了波动率聚集、均值回复与重尾，是长期 σ 的首选；
    3. **经验先验**：A 股宽基的典型区间波动幅度，用于历史样本不足时的兜底。

    权重按周期拉长而向"历史分布"倾斜（短线当前状态更重要，长线历史分布更重要）。
    """
    spec = horizon_spec(horizon_key)
    dp = vol.get("daily_pct", float("nan"))
    if dp != dp or dp <= 0:
        dp = 2.0                                   # 兜底：年化约 31%
    theo = dp * math.sqrt(max(spec.bars, 1)) * 0.92   # 0.92：重尾下的温和折扣

    # ---- 历史分布 ------------------------------------------------------- #
    emp = float("nan")
    if df is not None and len(df) > spec.bars + 20:
        try:
            emp = float(historical_horizon_sigma(df, spec.bars))
        except Exception:  # noqa: BLE001
            emp = float("nan")

    # ---- 经验先验（把年化波动率按 √t 缩放，并给出下限） ------------------ #
    ann = vol.get("annual_pct", float("nan"))
    if ann != ann or ann <= 0:
        ann = 32.0
    prior = ann / math.sqrt(244.0) * math.sqrt(max(spec.bars, 1)) * 0.95

    parts: list[tuple[float, float]] = []          # (值, 权重)
    if spec.kind == "short":
        parts.append((theo, 0.55))
        if emp == emp:
            parts.append((emp, 0.35))
        parts.append((prior, 0.10))
    elif spec.kind == "medium":
        parts.append((theo, 0.35))
        if emp == emp:
            parts.append((emp, 0.50))
        parts.append((prior, 0.15))
    else:
        parts.append((theo, 0.22))
        if emp == emp:
            parts.append((emp, 0.58))
        parts.append((prior, 0.20))
    if emp != emp:                                 # 无历史分布时把权重让给理论值
        parts = [(v, w * (1.6 if v == theo else 1.0)) for v, w in parts]
    wsum = sum(w for _, w in parts) or 1.0
    sigma = sum(v * w for v, w in parts) / wsum

    # AI 模块给出的区间收益标准差再做一次收缩（样本外验证过的口径）
    if sigma_hist and sigma_hist == sigma_hist and sigma_hist > 0:
        sigma = 0.72 * sigma + 0.28 * float(sigma_hist)

    floor = max(vol.get("sigma_floor", 0.55) * math.sqrt(spec.bars), 0.25)
    return float(max(sigma, floor))


#: 各周期的经验波动下限（%，A 股宽基长期口径）——防止极端低波动股给出荒谬的窄区间
SIGMA_PRIOR_FLOOR = {"d1": 1.1, "d5": 2.6, "d10": 3.7, "d20": 5.4,
                     "m1": 5.8, "q1": 10.0, "h1": 14.8, "y1": 21.5}


def historical_horizon_sigma(df: pd.DataFrame, bars: int, max_lookback: int = 1250) -> float:
    """统计该股历史上"持有 ``bars`` 个交易日"的收益标准差（%）。

    这是**无模型**的区间波动估计：把历史上所有重叠的 bars 日收益算出来取标准差。
    重叠样本会高估有效样本量，但用于估计波动率（而非检验显著性）是可接受的，
    且正好反映"任意时点买入并持有"的真实离散度。
    """
    if df is None or len(df) < bars + 20:
        return float("nan")
    c = df["close"].astype(float).to_numpy()
    n = len(c)
    start = max(0, n - max_lookback)
    seg = c[start:]
    if len(seg) <= bars:
        return float("nan")
    fwd = seg[bars:] / seg[:-bars] - 1.0
    fwd = fwd[np.isfinite(fwd)]
    if len(fwd) < 20:
        return float("nan")
    return float(np.std(fwd, ddof=1) * 100.0)


def apply_prior_floor(sigma: float, horizon_key: str) -> float:
    """给 σ 加一个经验下限，避免模型在低波动期过度自信。"""
    return max(float(sigma), SIGMA_PRIOR_FLOOR.get(horizon_key, 3.0))


# =========================================================================== #
# 三、单周期预测
# =========================================================================== #
def forecast_one(agg: Aggregated, vol: dict, horizon_key: str,
                 ml_pred=None, *,
                 prior: float | None = None,
                 price: float = float("nan"),
                 adj_for_high_vol: bool = True,
                 df: pd.DataFrame | None = None) -> ProbabilityForecast:
    """把因子汇总 + 波动率（+ 可选 AI 模型输出）转成一个周期的概率预测。

    ``df`` 为该股日线数据，用于估计"该周期历史上的真实波动分布"（长期 σ 的关键）。
    """
    _df_for_sigma = df
    spec = horizon_spec(horizon_key)
    F = config.FORECAST
    prior = F["prior_p_up"] if prior is None else prior

    score = agg.score
    # ---- 高波动衰减：高波动股的方向判断更容易失效 ---------------------- #
    if adj_for_high_vol and vol.get("regime") == "high":
        score *= F.get("high_vol_attenuation", 0.75)

    # ---- 概率：规则模型 ------------------------------------------------- #
    k = F["score_to_prob_scale"].get(horizon_key, 1.2)
    conf_rule = 0.35 + 0.65 * agg.evidence
    z_rule = logit(prior) + k * score * conf_rule
    p_rule = clamp(sigmoid(z_rule), F["prob_clip"][0], F["prob_clip"][1])

    # ---- 概率：AI 模型融合 --------------------------------------------- #
    mix: dict = {"规则因子模型": 0.0, "AI模型": 0.0}
    p_ml = None
    blend = float(F.get("ml_blend", {}).get(horizon_key, 0.3))
    if ml_pred is not None and getattr(ml_pred, "usable", False) and F.get("ml_enabled", True):
        p_ml = clamp(float(ml_pred.p_up), F["prob_clip"][0], F["prob_clip"][1])
        # AI 可信度低时降低其权重
        w_ml = blend * clamp(getattr(ml_pred, "confidence", 0.3) / 0.6, 0.15, 1.0)
        z = (1 - w_ml) * logit(p_rule) + w_ml * logit(p_ml)
        mix["AI模型"] = w_ml
        mix["规则因子模型"] = 1 - w_ml
    else:
        z = logit(p_rule)
        mix["规则因子模型"] = 1.0
    p_up = clamp(sigmoid(z), F["prob_clip"][0], F["prob_clip"][1])

    # ---- 波动率与幅度区间 ---------------------------------------------- #
    sigma_hist = float(getattr(ml_pred, "sigma", 0.0) or 0.0) if ml_pred is not None else 0.0
    sigma = horizon_sigma(vol, horizon_key, sigma_hist, df=_df_for_sigma)
    sigma = apply_prior_floor(sigma, horizon_key)
    lam = {  # 收益可预测性折扣：周期越长，方向分能解释的幅度比例越低
        "d1": 0.42, "d5": 0.44, "d10": 0.46, "d20": 0.48,
        "m1": 0.48, "q1": 0.45, "h1": 0.42, "y1": 0.38,
    }.get(horizon_key, 0.45)
    z_ret = norm_ppf(p_up)                       # 由概率反推标准正态分位
    exp_ret = z_ret * sigma * lam
    # AI 给出的幅度（KNN 历史类比）作为幅度锚点微调
    if ml_pred is not None and getattr(ml_pred, "usable", False):
        ml_ret = float(getattr(ml_pred, "exp_return", 0.0) or 0.0)
        if ml_ret == ml_ret and abs(ml_ret) < sigma * 2.5:
            exp_ret = 0.75 * exp_ret + 0.25 * ml_ret
    ci_low = exp_ret - 0.90 * sigma
    ci_high = exp_ret + 0.90 * sigma
    exp_low = exp_ret - 0.75 * sigma              # 悲观（~20% 分位）
    exp_high = exp_ret + 0.75 * sigma             # 乐观（~80% 分位）

    # ---- 大涨/大跌/横盘概率 -------------------------------------------- #
    big = float(F["big_move_pct"].get(horizon_key, 5.0))
    flat_band = float(F["flat_band_pct"].get(horizon_key, 1.0))
    sd = max(sigma, 1e-6)
    p_big_up = 1.0 - norm_cdf(big, exp_ret, sd)
    p_big_down = norm_cdf(-big, exp_ret, sd)
    p_flat = norm_cdf(flat_band, exp_ret, sd) - norm_cdf(-flat_band, exp_ret, sd)

    # ---- 结论与置信度 --------------------------------------------------- #
    edge = p_up - 0.5
    # 用"信号显著度"而不是裸阈值判方向：概率偏离 + 幅度是否超过噪声
    z_edge = abs(z_ret)
    if edge > 0 and z_edge >= 0.30:
        trend = "看涨"
    elif edge < 0 and z_edge >= 0.30:
        trend = "看跌"
    else:
        trend = "震荡"
    # 一致性：因子多空比例越悬殊越可信
    tot = max(agg.n_bull + agg.n_bear, 1)
    consensus = abs(agg.n_bull - agg.n_bear) / tot
    confidence = clamp(
        0.28 * agg.evidence +
        0.30 * min(z_edge / 1.6, 1.0) +
        0.22 * consensus +
        0.20 * (clamp(getattr(ml_pred, "confidence", 0.0)) if ml_pred is not None else 0.0)
    )

    # ---- 驱动因子与反向风险 -------------------------------------------- #
    contribs = sorted(agg.contributions, key=lambda t: abs(t[2]), reverse=True)
    drivers, counter = [], []
    for f, w, c in contribs:
        eff_dir = -f.direction if f.id in config.CONTRARIAN_FACTOR_IDS else f.direction
        row = {
            "因子": f.name, "id": f.id,
            "类别": Category.LABEL.get(f.category, f.category),
            "方向": "看涨" if eff_dir > 0 else "看跌",
            "贡献": round(c / max(agg.weight_sum, 1e-9) * 100, 1),
            "强度": round(f.strength, 2), "可信度": round(f.reliability, 2),
            "解读": f.detail, "理论依据": f.theory, "观测值": f.value,
        }
        if c > 0 and len(drivers) < 8:
            drivers.append(row)
        elif c < 0 and len(counter) < 6:
            counter.append(row)

    fc = ProbabilityForecast(
        horizon=horizon_key, label=spec.label, bars=spec.bars, days=spec.days,
        p_up=p_up, p_down=1 - p_up, p_flat=float(p_flat),
        p_big_up=float(p_big_up), p_big_down=float(p_big_down),
        exp_return=float(exp_ret), exp_return_low=float(exp_low), exp_return_high=float(exp_high),
        ci_low=float(ci_low), ci_high=float(ci_high), sigma=float(sigma),
        big_move_threshold=big, trend=trend, confidence=confidence,
        strength_score=round(agg.score_pct, 1), drivers=drivers, counter=counter,
        model_mix={k2: round(v2, 3) for k2, v2 in mix.items() if v2 > 0},
    )
    # 保留"未经自我学习校准"的概率，供预测档案核账时区分两个口径
    try:
        fc._p_rule = float(p_up)
    except Exception:  # noqa: BLE001
        pass
    note_bits = []
    if agg.evidence < 0.4:
        note_bits.append("有效因子偏少，结论仅供参考")
    if vol.get("regime") == "high":
        note_bits.append("该股处于高波动状态，方向判断已做衰减处理")
    if ml_pred is not None and not getattr(ml_pred, "usable", False):
        note_bits.append("AI 模型样本不足，本周期仅用规则模型")
    # ---- 概率来源披露（AI 核对图5 第1条：q1=29% 缺因子支撑）----
    # 概率由「规则因子分 → p_rule」与「AI 模型 → p_ml」加权融合得到。
    # 若最终概率与 p_rule 偏离很大，说明主导权在 AI 模型或先验，
    # 而不是因子——这点必须明说，否则用户会误以为因子支撑了该结论。
    try:
        _pr = float(p_rule)
        _gap = abs(float(p_up) - _pr)
        if _gap >= 0.06:
            note_bits.append(
                "本周期概率主要来自%s（因子分对应 %.0f%%，最终 %.0f%%）"
                % (("AI 模型" if mix.get("AI模型", 0) >= 0.5 else "模型融合"),
                   _pr * 100, float(p_up) * 100))
        fc._p_rule_detail = {"p_rule": round(_pr, 4), "p_ml": (round(p_ml, 4)
                              if p_ml is not None else None),
                              "w_ml": round(float(mix.get("AI模型", 0.0)), 3),
                              "prior": round(float(prior), 4)}
    except Exception:  # noqa: BLE001
        pass
    fc.note = "；".join(note_bits)
    return fc


# =========================================================================== #
# 四、多周期一致性
# =========================================================================== #
def cross_horizon_consistency(forecasts: list) -> dict:
    """检查短/中/长周期结论是否共振，并据此调整置信度。"""
    buckets = {"short": [], "medium": [], "long": []}
    for f in forecasts:
        try:
            spec = horizon_spec(f.horizon)
        except KeyError:
            continue
        buckets[spec.kind].append(f)
    signs = {}
    for kindname, items in buckets.items():
        if not items:
            continue
        val = float(np.mean([f.score for f in items]))
        signs[kindname] = val
    all_signs = [s for s in signs.values()]
    resonance = bool(all_signs) and all(s > 0 for s in all_signs) or \
        (bool(all_signs) and all(s < 0 for s in all_signs))
    direction = 0
    if all_signs:
        m = float(np.mean(all_signs))
        if resonance:
            direction = 1 if m > 0 else -1
    # 共振 → 提升全部周期置信度；分歧 → 降低
    if resonance:
        for f in forecasts:
            f.confidence = clamp(f.confidence * 1.18 + 0.04)
    else:
        for f in forecasts:
            f.confidence = clamp(f.confidence * 0.90)
    detail = "、".join("%s %+.2f" % ({"short": "短期", "medium": "中期", "long": "长期"}[k], v)
                       for k, v in signs.items())
    return {
        "resonance": resonance,
        "direction": direction,
        "by_bucket": {k: round(v, 3) for k, v in signs.items()},
        "detail": ("多周期共振（%s）：方向一致，结论可信度提升" % detail) if resonance
                  else ("周期背离（%s）：短中长期判断不一致，建议以中期方向为主" % detail),
    }


# =========================================================================== #
# 五、历史规律特征（供 AI 模型当特征用的"规则分"）
# =========================================================================== #
def fast_rule_score_history(df: pd.DataFrame, horizon_key: str) -> np.ndarray:
    """逐根 K 线计算一个**快速近似规则分**（-1~1），用作 AI 模型的输入特征。

    为什么需要它：AI 模型如果只吃原始价量特征，就无法学习"在什么市场状态下
    规则模型更准"。把规则分作为特征注入，模型可以自动学到
    "当规则分与动量一致时胜率更高 / 当两者冲突时应打折"。

    实现上只取因子矩阵的一个**因果子集**（动量、均线、RSI、MACD、量能、
    价格位置），逐根用 ``tanh`` 压缩后加权，保证与 :func:`aggregate_factors`
    的方向语义一致，但不需要跑全部模块。
    """
    n = len(df)
    out = np.zeros(n)
    if df is None or n < 40:
        return out
    c = df["close"].astype(float).to_numpy()
    h = df["high"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    v = df["volume"].astype(float).to_numpy() if "volume" in df.columns else np.ones(n)

    def _ma(x, k):
        return pd.Series(x).rolling(k).mean().to_numpy()

    ma5, ma20, ma60 = _ma(c, 5), _ma(c, 20), _ma(c, 60)
    e12, e26 = pd.Series(c).ewm(span=12, adjust=False).mean().to_numpy(), \
        pd.Series(c).ewm(span=26, adjust=False).mean().to_numpy()
    dif, dea = e12 - e26, pd.Series(e12 - e26).ewm(span=9, adjust=False).mean().to_numpy()
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / 14.0, adjust=False).mean().to_numpy()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / 14.0, adjust=False).mean().to_numpy()
    rsi = 100.0 * up / np.maximum(up + dn, 1e-9)
    vma5, vma20 = _ma(v, 5), _ma(v, 20)
    hi20 = pd.Series(h).rolling(20).max().to_numpy()
    lo20 = pd.Series(lo).rolling(20).min().to_numpy()
    hi60 = pd.Series(h).rolling(60).max().to_numpy()

    spec = horizon_spec(horizon_key)
    # 周期越长，越看均线排列与长期位置；越短越看摆动指标
    w = {"mom": 1.0, "ma": 1.0, "rsi": 1.0, "macd": 1.0, "vol": 1.0, "pos": 1.0}
    if spec.kind == "short":
        w.update({"rsi": 1.3, "macd": 1.2, "vol": 1.15, "ma": 0.8, "pos": 0.8})
    elif spec.kind == "long":
        w.update({"ma": 1.4, "pos": 1.3, "rsi": 0.6, "macd": 0.7, "vol": 0.6})

    with np.errstate(invalid="ignore", divide="ignore"):
        mom = np.tanh((c / np.where(ma20 > 0, ma20, np.nan) - 1.0) * 12.0)
        ma_s = np.tanh(((ma5 - ma20) / np.where(ma20 > 0, ma20, np.nan)) * 30.0)
        ma_l = np.tanh(((ma20 - ma60) / np.where(ma60 > 0, ma60, np.nan)) * 22.0)
        rsi_s = np.tanh((rsi - 50.0) / 18.0)
        macd_s = np.tanh((dif - dea) / np.where(c > 0, c, np.nan) * 900.0)
        vol_s = np.tanh((v / np.where(vma20 > 0, vma20, np.nan) - 1.0) * 0.8) * \
            np.sign(np.nan_to_num(c / np.where(ma5 > 0, ma5, np.nan) - 1.0))
        pos_s = np.tanh(((c - lo20) / np.where(hi20 > lo20, hi20 - lo20, np.nan) - 0.5) * 1.6)
        dd = np.tanh((c / np.where(hi60 > 0, hi60, np.nan) - 1.0) * 6.0)

    num = (w["mom"] * mom + w["ma"] * (0.5 * ma_s + 0.5 * ma_l) + w["rsi"] * rsi_s +
           w["macd"] * macd_s + w["vol"] * vol_s + w["pos"] * (0.6 * pos_s + 0.4 * dd))
    den = (w["mom"] + w["ma"] + w["rsi"] + w["macd"] + w["vol"] + w["pos"])
    score = np.nan_to_num(num / max(den, 1e-9), nan=0.0)
    # 预热期置 0（无观点），避免把"数据不足"当成看跌
    warm = np.minimum(np.arange(n) / 60.0, 1.0)
    return np.clip(score, -1, 1) * warm


# =========================================================================== #
# 六、主入口
# =========================================================================== #
# 概率区分度自检：距离 50% 不足 5 个百分点时，说明因子证据不足以
# 支撑方向判断，此时应如实提示"接近随机"，而不是让用户误以为有把握。
NEAR_RANDOM_BAND = 0.05


def near_random(p_up: float) -> str:
    """返回提示文本；概率距 50% 足够远时返回空串。"""
    try:
        gap = abs(float(p_up) - 0.5)
    except Exception:  # noqa: BLE001
        return ""
    if gap < NEAR_RANDOM_BAND:
        return ("证据不足，接近随机（距 50%% 仅 %.1f 个百分点）" % (gap * 100))
    return ""

def build_forecasts(bundle, *, ml_models: dict | None = None,
                    use_ml: bool = True, horizons=HORIZONS,
                    df_override: pd.DataFrame | None = None) -> PredictionResult:
    """对一个已填充好的 :class:`AnalysisBundle` 生成 8 个周期的概率预测。

    Parameters
    ----------
    bundle : AnalysisBundle
    ml_models : dict | None
        由 :func:`core.mlmodel.train_models` 训练的模型；为 None 时自动训练
        （耗时较长，Streamlit 快速模式可传 ``use_ml=False`` 跳过）。
    use_ml : bool
        是否启用 AI 模型融合。
    horizons : 预测周期集合

    Returns
    -------
    PredictionResult
    """
    df = df_override if df_override is not None else getattr(bundle, "ohlcv", None)
    price = float("nan")
    if df is not None and len(df) and "close" in df.columns:
        price = float(df["close"].iloc[-1])
    if price != price and bundle.snapshot:
        price = float(bundle.snapshot.get("price") or float("nan"))

    vol = estimate_volatility(df)
    bundle.indicators["volatility"] = vol

    # ---- 自我学习校准：读取该股历史核账得出的校准参数 ------------------ #
    adjust_map: dict = {}
    dim_weights: dict = {}
    if config.FORECAST.get("self_learning", True):
        try:
            from . import learning as _learn

            for h in horizons:
                adj = _learn.get_adjustments(bundle.symbol, h.key)
                if adj:
                    adjust_map[h.key] = adj
            dim_weights = _learn.get_dimension_weights(bundle.symbol) or {}
            if adjust_map:
                bundle.indicators["self_learning"] = {
                    "enabled": True,
                    "horizons": {k: {"samples": v.get("samples"),
                                     "hit_rate": v.get("hit_rate"),
                                     "prob_adjust_logit": v.get("prob_adjust_logit"),
                                     "confidence_mult": v.get("confidence_mult")}
                                 for k, v in adjust_map.items()},
                    "dimension_weights": dim_weights,
                }
        except Exception as exc:  # noqa: BLE001
            log.debug("读取自我学习校准失败: %s", exc)

    # ---- AI 模型：训练或使用传入的模型 --------------------------------- #
    ml_preds: dict = {}
    ml_meta: dict = {"enabled": bool(use_ml), "trained": False}
    if use_ml and config.FORECAST.get("ml_enabled", True) and df is not None and len(df) >= 120:
        try:
            models = ml_models
            if models is None:
                ctx = {"index_df": getattr(bundle, "index_data", None)}
                # 用快速规则分的历史序列作为额外特征
                fs = {h.key: float(aggregate_factors(bundle.factors, h.key).score)
                      for h in HORIZONS}
                ctx["factor_score"] = fs
                models = None  # 训练时不能用"当前规则分"当历史特征
                models = train_models_with_rule_feature(df, bundle, horizons)
                ml_meta["trained"] = True
            ctx_pred = {"index_df": getattr(bundle, "index_data", None),
                        "factor_score": {h.key: float(aggregate_factors(bundle.factors, h.key).score)
                                         for h in HORIZONS}}
            from .mlmodel import predict as ml_predict
            ml_preds = ml_predict(models or {}, df, ctx_pred)
            ml_meta["models"] = {k: {"n_train": v.n_train, "auc": v.oos.get("auc"),
                                     "model": "+".join(v.models.keys()) if v.models else "—"}
                                 for k, v in (models or {}).items()}
        except Exception as exc:  # noqa: BLE001
            bundle.error("mlmodel", exc)
            log.warning("AI 模型不可用，退回纯规则模型: %s", exc)
            ml_meta["error"] = str(exc)
    ml_meta["n_predictions"] = len(ml_preds)
    bundle.ml = {**bundle.ml, **ml_meta}

    # ---- 逐周期预测 ----------------------------------------------------- #
    forecasts: list[ProbabilityForecast] = []
    aggs: dict[str, Aggregated] = {}
    for h in horizons:
        agg = aggregate_factors(bundle.factors, h.key)
        aggs[h.key] = agg
        fc = forecast_one(agg, vol, h.key, ml_preds.get(h.key),
                          price=price, prior=_prior_for(bundle, h), df=df)
        # 应用自我学习校准（用真实的核账记录修正概率与置信度）
        adj = adjust_map.get(h.key)
        if adj:
            try:
                from .utils import logit as _logit, sigmoid as _sigmoid

                lo, hi = config.FORECAST["prob_clip"]
                # ---- 校准强度限幅（AI 核对图5 第1条）----
                # 自我学习按该股历史核账记录调概率，但各周期样本量不同
                # （实测 40~60 条），小样本校准会把概率推到因子完全
                # 支撑不了的位置（如 q1=29% 而因子分只有 +0.03）。
                # 规则：调整量按样本量打折，且单次不超过 ±8 个百分点等效 logit。
                _n_s = int(adj.get("samples") or 0)
                _raw = float(adj.get("prob_adjust_logit") or 0.0)
                # 样本 < 200 时按 sqrt(n/200) 缩水，最低保留 25%
                _shrink = max(0.25, min(1.0, (_n_s / 200.0) ** 0.5))
                _adj = max(-0.35, min(0.35, _raw * _shrink))
                z = _logit(fc.p_up) + _adj
                fc.p_up = clamp(_sigmoid(z), lo, hi)
                if abs(_raw - _adj) > 0.02:
                    fc.note = ((fc.note + "；") if fc.note else "") + \
                        ("校准强度已按样本量收缩（%d 条，系数 %.2f，"
                         "原始调整 %.2f → 实际 %.2f）" % (_n_s, _shrink, _raw, _adj))
                fc.p_down = 1 - fc.p_up
                fc.confidence = clamp(fc.confidence *
                                      float(adj.get("confidence_mult") or 1.0), 0.03, 0.95)
                # 概率变了，期望幅度与区间也要跟着重算，保持自洽
                z_ret = norm_ppf(fc.p_up)
                fc.exp_return = float(z_ret * fc.sigma * 0.45)
                fc.ci_low = fc.exp_return - 0.90 * fc.sigma
                fc.ci_high = fc.exp_return + 0.90 * fc.sigma
                fc.exp_return_low = fc.exp_return - 0.75 * fc.sigma
                fc.exp_return_high = fc.exp_return + 0.75 * fc.sigma
                sd = max(fc.sigma, 1e-6)
                big = fc.big_move_threshold
                fc.p_big_up = 1.0 - norm_cdf(big, fc.exp_return, sd)
                fc.p_big_down = norm_cdf(-big, fc.exp_return, sd)
                fc.trend = ("看涨" if fc.p_up > 0.5 and abs(z_ret) >= 0.30
                            else ("看跌" if fc.p_up < 0.5 and abs(z_ret) >= 0.30 else "震荡"))
                mark = ("本周期已应用自我学习校准（基于该股 %s 条历史核账记录，"
                        "历史命中率 %s）" % (
                            adj.get("samples"),
                            ("%.0f%%" % (adj["hit_rate"] * 100))
                            if adj.get("hit_rate") is not None else "—"))
                fc.note = ("%s；%s" % (fc.note, mark)) if fc.note else mark
            except Exception as exc:  # noqa: BLE001
                log.debug("应用学习校准失败 %s: %s", h.key, exc)
        forecasts.append(fc)
    forecasts.sort(key=lambda f: horizon_spec(f.horizon).bars)

    consistency = cross_horizon_consistency(forecasts)
    bundle.indicators["cross_horizon"] = consistency

    result = PredictionResult(
        symbol=bundle.symbol, name=bundle.name, as_of=bundle.as_of,
        price=price, forecasts=forecasts, bundle=bundle,
        meta={
            "volatility": vol, "consistency": consistency, "ml": ml_meta,
            "aggregates": {k: {"score": v.score, "n": v.n_factors,
                               "evidence": v.evidence,
                               "bull": v.n_bull, "bear": v.n_bear}
                           for k, v in aggs.items()},
            "n_factors": len(bundle.factors),
        },
    )
    return result


def _prior_for(bundle, spec: HorizonSpec) -> float:
    """按市场环境与个股阶段给先验上涨概率（略微偏离 50%）。"""
    prior = config.FORECAST["prior_p_up"]
    try:
        regime = (bundle.macro or {}).get("market_regime") or {}
        rs = float(regime.get("trend_score", 0.0) or 0.0)
        # 大盘趋势向上 → 先验略偏多；向下 → 略偏空
        if spec.kind == "long":
            prior += 0.045 * clamp(rs, -1, 1)
        elif spec.kind == "medium":
            prior += 0.025 * clamp(rs, -1, 1)
        else:
            prior += 0.012 * clamp(rs, -1, 1)
    except Exception:  # noqa: BLE001
        pass
    return clamp(prior, 0.30, 0.70)


def train_models_with_rule_feature(df: pd.DataFrame, bundle, horizons=HORIZONS) -> dict:
    """训练 AI 模型，并把**逐根 K 线的规则分历史**作为特征注入。

    这是"理论规则 + 机器学习"结合的关键：模型能学到
    "规则模型在哪些状态下更可靠"，从而动态调整规则分的权重。
    """
    from .mlmodel import build_features, make_targets, train_models
    try:
        x = build_features(df, getattr(bundle, "index_data", None))
        if x.empty:
            return {}
        extra = {}
        for h in horizons:
            extra["rule_score_%s" % h.key] = fast_rule_score_history(df, h.key)
        x = x.copy()
        for k, vv in extra.items():
            x[k] = vv
        # 用带规则分特征的特征矩阵重新训练
        t = make_targets(df, horizons)
        models = _train_with_features(x, t, horizons)
        return models
    except Exception as exc:  # noqa: BLE001
        log.warning("注入规则分特征训练失败，退回标准训练: %s", exc)
        return train_models(df, {"index_df": getattr(bundle, "index_data", None)}, horizons)


def _train_with_features(x: pd.DataFrame, t: pd.DataFrame, horizons) -> dict:
    """在给定特征矩阵上训练（:func:`core.mlmodel.train_models` 的内部复用入口）。"""
    from . import mlmodel
    from .mlmodel import TrainedModel, KNNPatternModel, _model_zoo, _time_decay_weights

    out: dict = {}
    for h in horizons:
        col = "fwd_%s" % h.key
        if col not in t.columns:
            continue
        y_ret = t[col].to_numpy(dtype=float)
        x_all = x.to_numpy(dtype=float)
        feats = list(x.columns)
        valid = np.isfinite(y_ret) & (np.isfinite(x_all).mean(axis=1) >= 0.6)
        idx_tr = np.where(valid)[0]
        if len(idx_tr) < config.RUNTIME["ml_min_samples"]:
            out[h.key] = TrainedModel(horizon=h.key, label=h.label, bars=h.bars,
                                      features=feats, n_train=len(idx_tr), note="样本不足")
            continue
        x_tr = x_all[idx_tr]
        y_tr = (y_ret[idx_tr] > 0).astype(float)
        w = _time_decay_weights(len(idx_tr), 250.0)
        models = {}
        for name, factory in _model_zoo(config.FORECAST.get("ml_model", "auto"), len(idx_tr)).items():
            try:
                mdl = factory()
                mdl.fit(x_tr, y_tr, sample_weight=w)
                models[name] = mdl
            except Exception as exc:  # noqa: BLE001
                log.debug("训练 %s/%s 失败: %s", h.key, name, exc)
        try:
            knn = KNNPatternModel(k=min(40, max(15, len(idx_tr) // 8)))
            knn.fit(x_tr, y_ret[idx_tr], sample_weight=w)
            if knn.fitted_:
                models["knn"] = knn
        except Exception as exc:  # noqa: BLE001
            log.debug("训练 %s/knn 失败: %s", h.key, exc)
        oos = {}
        try:
            ev = mlmodel.walk_forward_eval(
                x_tr, y_tr, lambda: mlmodel.LogisticModel(l2=1.5, lr=0.3, epochs=300),
                n_splits=3)
            oos = ev
        except Exception:  # noqa: BLE001
            pass
        out[h.key] = TrainedModel(horizon=h.key, label=h.label, bars=h.bars, features=feats,
                                  models=models, oos=oos, base_rate=float(y_tr.mean()),
                                  n_train=len(idx_tr),
                                  sigma_hist=float(np.nanstd(y_ret[idx_tr])))
    return out


# =========================================================================== #
# 七、汇总建议
# =========================================================================== #
def ensemble_view(result: PredictionResult) -> dict:
    """把 8 个周期的预测汇总成可执行的投资视角（综合评分 + 操作倾向）。"""
    if not result.forecasts:
        return {}
    F = result.forecasts
    weights = {"d1": 0.05, "d5": 0.08, "d10": 0.10, "d20": 0.17,
               "m1": 0.18, "q1": 0.20, "h1": 0.12, "y1": 0.10}
    tot = sum(weights.get(f.horizon, 0.1) for f in F)
    score = sum(weights.get(f.horizon, 0.1) * f.score for f in F) / max(tot, 1e-9)
    exp_20 = (result.by_key("d20") or F[0]).exp_return
    cons = result.meta.get("consistency", {})
    horizon_summary = [{"周期": f.label, "结论": f.trend,
                        "上涨概率": round(f.p_up * 100, 1),
                        "期望涨跌幅": round(f.exp_return, 2),
                        "置信度": round(f.confidence * 100, 1)} for f in F]

    # 操作倾向
    if score > 0.35:
        stance = "积极看多"
    elif score > 0.15:
        stance = "偏多"
    elif score > 0.05:
        stance = "谨慎偏多"
    elif score > -0.05:
        stance = "中性观望"
    elif score > -0.15:
        stance = "谨慎偏空"
    elif score > -0.35:
        stance = "偏空"
    else:
        stance = "明确看空"

    # 关键价位：用 d20 区间给出参考
    f20 = result.by_key("d20")
    key_levels = {}
    if f20 and result.price == result.price:
        key_levels = {
            "现价": round(result.price, 2),
            "20日乐观目标": round(result.price * (1 + f20.exp_return_high / 100), 2),
            "20日中性目标": round(result.price * (1 + f20.exp_return / 100), 2),
            "20日悲观目标": round(result.price * (1 + f20.exp_return_low / 100), 2),
            "20日区间下沿": round(result.price * (1 + f20.ci_low / 100), 2),
            "20日区间上沿": round(result.price * (1 + f20.ci_high / 100), 2),
        }
    # 最强驱动与最大风险（跨周期合并）
    all_drv: dict = {}
    all_rsk: dict = {}
    for f in F:
        for d in f.drivers:
            all_drv[d["因子"]] = all_drv.get(d["因子"], 0) + abs(d.get("贡献", 0)) * f.confidence
        for d in f.counter:
            all_rsk[d["因子"]] = all_rsk.get(d["因子"], 0) + abs(d.get("贡献", 0)) * f.confidence
    top_drv = sorted(all_drv.items(), key=lambda kv: -kv[1])[:6]
    top_rsk = sorted(all_rsk.items(), key=lambda kv: -kv[1])[:5]

    return {
        "score": round(score, 4),
        "score_pct": round(score * 100, 1),
        "stance": stance,
        "p_up_20d": round((result.by_key("d20") or F[0]).p_up, 4),
        "exp_20d": round(exp_20, 2),
        "consistency": cons,
        "horizon_summary": horizon_summary,
        "key_levels": key_levels,
        "top_drivers": [{"因子": k, "权重分": round(v, 2)} for k, v in top_drv],
        "top_risks": [{"因子": k, "权重分": round(v, 2)} for k, v in top_rsk],
    }


# =========================================================================== #
# 自测
# =========================================================================== #
if __name__ == "__main__":  # pragma: no cover
    from .contracts import AnalysisBundle
    from .selftest_data import make_index, make_ohlcv

    df = make_ohlcv(700, seed=42)
    idx = make_index(700, seed=7)
    b = AnalysisBundle(symbol="600519", name="测试股", as_of=str(df["date"].iloc[-1])[:10],
                       ohlcv=df, index_data=idx)
    # 造一批假因子：技术/资金看多，基本面看空，检验权重矩阵是否按周期生效
    b.add(Factor("ma_alignment", "均线多头排列", Category.TECH, 1, 0.8, 0.9,
                 detail="MA5>MA20>MA60", theory="道氏理论"))
    b.add(Factor("macd_cross", "MACD金叉", Category.TECH, 1, 0.6, 0.85,
                 detail="DIF上穿DEA", theory="MACD"))
    b.add(Factor("capital_main_net", "主力资金净流入", Category.CAPITAL, 1, 0.7, 0.8,
                 detail="近5日净流入3.2亿", theory="资金流向"))
    b.add(Factor("valuation_percentile", "估值处于高位", Category.FUND, -1, 0.6, 0.7,
                 detail="PE分位 88%", theory="估值回归"))
    b.add(Factor("fundamental_growth", "营收增速下滑", Category.FUND, -1, 0.5, 0.65,
                 detail="营收同比 -5%", theory="基本面"))
    b.add(Factor("retail_sentiment_contrarian", "散户情绪过热", Category.SENTIMENT, 1, 0.7, 0.6,
                 detail="股吧人气排名骤升（反向指标）", theory="博弈心理"))
    b.add(Factor("macro_cycle", "PMI 回升", Category.MACRO, 1, 0.4, 0.55,
                 detail="PMI 50.8", theory="经济周期"))

    res = build_forecasts(b, use_ml=False)
    print("现价 %.2f  因子数 %d" % (res.price, len(b.factors)))
    print()
    for f in res.forecasts:
        print("%-16s %s  涨%.1f%%/跌%.1f%%  期望%+.2f%%  区间[%+.1f%%, %+.1f%%]  σ=%.2f%%  conf=%.2f" % (
            f.label, f.trend, f.p_up * 100, f.p_down * 100, f.exp_return,
            f.ci_low, f.ci_high, f.sigma, f.confidence))
    print()
    ens = ensemble_view(res)
    print("综合评分:", ens["score_pct"], "操作倾向:", ens["stance"])
    print("一致性:", ens["consistency"]["detail"])
    print("关键驱动:", [d["因子"] for d in ens["top_drivers"]])
    print("主要风险:", [d["因子"] for d in ens["top_risks"]])

    # ---- 断言 ----
    assert len(res.forecasts) == 8, "应输出 8 个周期"
    for f in res.forecasts:
        assert 0.0 <= f.p_up <= 1.0 and abs(f.p_up + f.p_down - 1.0) < 1e-9
        assert f.ci_low < f.exp_return < f.ci_high, "区间必须包含期望值"
        assert f.sigma > 0
        assert 0 <= f.confidence <= 1
        assert f.p_big_up + f.p_big_down <= 1.0 + 1e-9
    # 短线资金权重 > 长线资金权重 → d1 的分数应比 y1 更偏多
    d1 = res.by_key("d1")
    y1 = res.by_key("y1")
    assert d1.strength_score > 0 and y1.strength_score < d1.strength_score, \
        "短线资金面主导、长线基本面主导，权重矩阵应体现周期差异：d1=%.1f y1=%.1f" % (
            d1.strength_score, y1.strength_score)
    # 反向指标必须被翻转
    agg = aggregate_factors(b.factors, "d1")
    retail = [c for f, w, c in agg.contributions if f.id == "retail_sentiment_contrarian"]
    assert retail and retail[0] < 0, "散户情绪反向指标应贡献负分"
    # 波动率随周期单调
    sigs = [f.sigma for f in res.forecasts]
    assert sigs == sorted(sigs), "波动率应随周期递增: %s" % sigs
    # 快速规则分历史严格因果
    s_full = fast_rule_score_history(df, "d20")
    s_trunc = fast_rule_score_history(df.iloc[:500], "d20")
    assert np.allclose(s_full[:500], s_trunc, atol=1e-9), "规则分历史存在未来函数"
    print("\nforecast self-test OK")

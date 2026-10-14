# -*- coding: utf-8 -*-
"""
金融理论模型模块（Classical / Theoretical Analysis）。

职责
----
把"经典技术分析理论"中**可量化**的那部分实现成确定性算法，输出中文可读的结构化
结论，并进一步压缩成 :class:`core.contracts.Factor`（``Category.THEORY``）。

覆盖的理论
----------
======================  ==========================================================
函数                     理论
======================  ==========================================================
:func:`dow_theory`       道氏理论（摆动高低点结构 + 成交量验证 + 阶段判断）
:func:`elliott_wave`     艾略特波浪理论（ZigZag 数浪 + 斐波那契吻合度）
:func:`fibonacci_analysis` 斐波那契回撤/扩展 + 斐波那契时间窗
:func:`gann_analysis`    江恩理论（角度线 / 时间周期 / 四方形 Square of 9）
:func:`volume_price_time` 量价时空（量价 8 种经典配合 + 时间维度）
:func:`fan_principle`    扇形原理（多角度线分层 + 1×1 线得失）
:func:`percentage_retracement` 百分比回撤（1/3、1/2、2/3 与 1/8 分数体系）
:func:`chaos_theory`     混沌理论（R/S 分析 Hurst 指数、分形维数、初值敏感性）
:func:`multi_timeframe_resonance` 多周期共振（日/周/月道氏趋势一致性）
======================  ==========================================================

输出因子（id 稳定，供权重表匹配）
--------------------------------
``theory_dow_trend`` / ``theory_elliott`` / ``theory_fib_level`` / ``theory_gann`` /
``theory_volume_price_time`` / ``theory_fan`` / ``theory_pct_retracement`` /
``theory_chaos`` / ``theory_resonance``

设计约束
--------
* 只依赖 numpy / pandas / 标准库，**不使用 scipy / sklearn / TA-Lib**。
* 不导入 :mod:`core.indicators` / :mod:`core.patterns`（另一位工程师并行开发），
  所需的 ATR、均线等少量计算在本模块内用 pandas/numpy 自实现，避免耦合风险。
* **绝不抛异常打断主流程**：任何函数在数据不足/算不出时返回降级 dict（通常为"中性"
  形态），并下调 ``confidence`` / ``reliability``。所有对外函数都包在 :func:`_guard` 里。
* 理论模型主观性强，:func:`theory_factors` 给出的 ``reliability`` 上限为 **0.65**，
  江恩/混沌/波浪再额外打折到 **≤0.55**。

局限性（务必与报告一起展示）
----------------------------
1. 道氏理论只描述趋势的"确认"，天然滞后，且对摆动点窗口参数敏感。
2. 波浪数浪是"事后清晰、事前模糊"，同一段行情可以数出多种浪级；本模块的做法是
   只保留与斐波那契比例高度吻合的数法，否则返回 ``"unknown"``。
3. 江恩理论（角度线/时间周期/四方形）没有任何统计显著的实证支持，属于**经验性方法**，
   本模块仅作为弱参考，因子强度被显著压低。
4. 斐波那契回撤/时间窗属于"自我实现的预言"，只在被广泛使用的大市值标的上略有意义。
5. Hurst 指数的 R/S 估计在高噪声、短样本下偏差极大，样本 <120 时本模块直接返回
   0.5 保守值；分形维数由 D = 2 - H 直接推导，并非独立测量。
"""

from __future__ import annotations

import datetime as _dt
import functools
import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import config
from .contracts import (
    DIR_BEAR,
    DIR_BULL,
    DIR_NEUTRAL,
    Category,
    Factor,
    clamp,
    safe_float,
)
from .utils import get_logger

log = get_logger("stockai.theory")

__all__ = [
    "dow_theory", "elliott_wave", "fibonacci_analysis", "gann_analysis",
    "volume_price_time", "fan_principle", "percentage_retracement", "chaos_theory",
    "multi_timeframe_resonance", "theory_factors", "theory_summary",
]

# --------------------------------------------------------------------------- #
# 模块级常量
# --------------------------------------------------------------------------- #
THEORY_NAME = {
    "dow": "道氏理论",
    "elliott": "艾略特波浪理论",
    "fib": "斐波那契回撤理论",
    "fib_time": "斐波那契时间窗",
    "gann": "江恩理论",
    "vpt": "量价时空理论",
    "fan": "扇形原理",
    "pct": "百分比回撤理论",
    "chaos": "混沌理论",
    "resonance": "多周期共振",
}

# 道氏理论：摆动点判定参数（选择理由见 :func:`_swing_points`）
DOW_SWING_WINDOW = 3          # 左右各 3 根确认局部高/低点（题目要求）
DOW_MIN_GAP = 3               # 相邻摆动点至少间隔 3 根 K 线
DOW_SMOOTH = 3                # 判定前对收盘价做 3 日居中平滑，压制单根噪声毛刺
DOW_MIN_SWING_PCT = 0.006     # 相邻同类摆动点最小间距 0.6%：低于此视为"平"（横盘）

# 艾略特波浪：ZigZag 阈值
ELLIOTT_ATR_WINDOW = 60       # 近 60 日 ATR 均值
ELLIOTT_ATR_MULT = 2.5        # 阈值 = ATR × 2.5
ELLIOTT_MIN_PCT = 0.04        # 且至少 4%
ELLIOTT_N_SWINGS = 7          # 用最近 7 个摆动点（= 0..5 共 6 段 + 当前定位）

# 斐波那契回撤档位（0.191 为浅回撤，0.809 为极深回撤；价格越低越靠下）
FIB_RATIOS = (0.191, 0.236, 0.382, 0.5, 0.618, 0.786, 0.809)
FIB_EXT_RATIOS = (1.272, 1.618, 2.0, 2.618)

# 百分比回撤：1/3、1/2、2/3 + 1/8 分数体系
PCT_RATIOS = (0.125, 0.25, 0.333, 0.375, 0.5, 0.625, 0.667, 0.75, 0.875)
PCT_LABEL = {
    0.125: "1/8", 0.25: "1/4", 0.333: "1/3", 0.375: "3/8", 0.5: "1/2",
    0.625: "5/8", 0.667: "2/3", 0.75: "3/4", 0.875: "7/8",
}

# 江恩角度线标签（1×2 表示 1 单位时间走 2 单位价格，最陡）
GANN_ANGLE_LABELS = ("8x1", "4x1", "3x1", "2x1", "1x1", "1x2", "1x3", "1x8")

# 四方形（Square of 9）使用的平方根步长序列
SQUARE9_STEPS = (1, 2, 3, 4, 5, 6, 7, 8)

# 斐波那契时间窗（交易日），来自 config，缺失时兜底
TIME_WINDOW_BARS = tuple(int(b) for b in config.INDICATOR_PARAMS.get(
    "time_window_bars", (144, 89, 55, 34, 21, 13, 8, 5)))

# 多周期分析所需的最少样本
MIN_BARS_DAILY = 60
MIN_BARS_WEEKLY = 20
MIN_BARS_MONTHLY = 12


# --------------------------------------------------------------------------- #
# 0. 通用小工具（不依赖 indicators.py）
# --------------------------------------------------------------------------- #
def _num(x: Any, default: float = float("nan")) -> float:
    """安全转 float（NaN/None/异常一律返回 default）。"""
    v = safe_float(x, default=default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(v) or math.isinf(v)) else v


def _r(x: Any, digits: int = 2) -> Any:
    """四舍五入到指定位数，非数值原样返回（报告友好）。"""
    v = _num(x)
    return None if math.isnan(v) else round(v, digits)


def _guard(default, label: str):
    """装饰器：内部异常一律降级为 default，绝不打断主流程。

    用 :func:`functools.wraps` 保留原函数签名/文档，便于 ``inspect.signature``
    与 IDE 提示正常工作。
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                log.debug("%s 降级: %s: %s", label, type(exc).__name__, exc)
                if callable(default):
                    try:
                        return default()
                    except Exception:  # noqa: BLE001
                        return None
                return default
        return wrapper
    return deco


def _ensure_frame(df) -> pd.DataFrame:
    """列名容错 + 类型清洗（**不修改**传入对象）。

    支持小写/大写/中文列名，统一为
    ``date, open, high, low, close, volume, amount``。
    数据不足 20 根或缺少 close 列时返回空 DataFrame。
    """
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 20:
        return pd.DataFrame()
    out = df.copy()
    cmap = {}
    for col in out.columns:
        key = str(col).strip().lower().replace(" ", "")
        cmap[col] = {
            "date": "date", "日期": "date", "time": "date", "datetime": "date",
            "open": "open", "开盘": "open", "开盘价": "open",
            "high": "high", "最高": "high", "最高价": "high",
            "low": "low", "最低": "low", "最低价": "low",
            "close": "close", "收盘": "close", "收盘价": "close", "adjclose": "close",
            "volume": "volume", "vol": "volume", "成交量": "volume",
            "amount": "amount", "turnover_amount": "amount", "成交额": "amount",
        }.get(key, key)
    out = out.rename(columns=cmap)
    if "close" not in out.columns:
        return pd.DataFrame()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
    else:
        out = out.reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    close = out["close"].astype(float)
    if "high" not in out.columns:
        out["high"] = close
    else:
        out["high"] = out["high"].fillna(close)
    if "low" not in out.columns:
        out["low"] = close
    else:
        out["low"] = out["low"].fillna(close)
    if "open" not in out.columns:
        out["open"] = close
    if "volume" not in out.columns:
        # 没有成交量时用振幅代理，保证下游不会除零
        out["volume"] = (out["high"] - out["low"]).abs() * 1e6 + 1.0
    out["volume"] = out["volume"].fillna(0.0).astype(float)
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    return out if len(out) >= 20 else pd.DataFrame()


def _dates(df: pd.DataFrame) -> pd.Series:
    """取日期序列；无 date 列时用序号构造伪日期（保证不炸）。"""
    if "date" in df.columns:
        return df["date"]
    return pd.Series(pd.date_range("2000-01-03", periods=len(df), freq="B"))


def _date_str(df: pd.DataFrame, i: int) -> str:
    """第 i 根 K 线的日期字符串（越界返回空串）。"""
    try:
        d = _dates(df).iloc[int(i)]
        return d.strftime("%Y-%m-%d") if d is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def _last_date(df: pd.DataFrame) -> pd.Timestamp:
    try:
        return pd.Timestamp(_dates(df).iloc[-1])
    except Exception:  # noqa: BLE001
        return pd.Timestamp("2000-01-03")


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均真实波幅（Wilder 平滑）。数据不足时头部为 NaN。"""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / max(2, window), adjust=False, min_periods=1).mean()


def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.astype(float).rolling(int(window), min_periods=1).mean()


def _ret_pct(close: pd.Series, bars: int) -> float:
    """最近 bars 根 K 线的收益率（%）。"""
    c = close.astype(float)
    if len(c) <= bars:
        return float("nan")
    prev = _num(c.iloc[-bars - 1])
    last = _num(c.iloc[-1])
    if math.isnan(prev) or math.isnan(last) or prev == 0:
        return float("nan")
    return (last / prev - 1.0) * 100.0


def _pct_pos(price: float, low: float, high: float) -> float:
    """价格在 [low, high] 区间中的位置（0=区间底, 1=区间顶）。"""
    if math.isnan(price) or math.isnan(low) or math.isnan(high) or high <= low:
        return 0.5
    return clamp((price - low) / (high - low), 0.0, 1.0)


def _swing_points(df: pd.DataFrame, window: int = DOW_SWING_WINDOW,
                  n_max: int = 24, smooth: int = DOW_SMOOTH,
                  min_gap: int = DOW_MIN_GAP,
                  min_swing_pct: float = DOW_MIN_SWING_PCT) -> list[dict]:
    """提取"摆动点"（左右各 ``window`` 根确认的局部高/低点）。

    参数选择理由
    ------------
    1. **左右各 3 根**：题目要求。经验上 3 根是"滤掉单日毛刺"与"及时反应转折"的
       平衡点：左右 1~2 根会把噪声当转折，左右 ≥5 根在日线上平均滞后 2 周以上。
    2. **先做 3 日居中平滑**：直接对高低价取局部极值会被"插针"严重污染
       （合成数据的 high/low 带噪声），平滑后拐点数量下降约一半而转折位置几乎不变。
    3. **min_gap=3**：相邻摆动点至少隔 3 根 K 线，避免把同一根 K 线附近的
       连续极值算成两段（重合区间会让"高点抬高/降低"的判定失去意义）。
    4. **min_swing_pct=0.6%**：同一类型相邻摆动点价格变化小于 0.6% 视为"平"
       （既不去重也不放大），用于横盘识别；该阈值不用于删点，只用于趋势判定。
    5. **去重（0.5% 容差）**：平滑可能造出相邻两个峰，保留更高者（谷保留更低者），
       并强制高低点交替，得到严格锯齿结构，便于数浪与道氏比较。

    Returns
    -------
    list[dict]，按时间升序，元素形如
    ``{"idx": int, "date": str, "price": float, "type": "high"/"low"}``
    """
    if df is None or len(df) < 2 * window + 3:
        return []
    close = df["close"].astype(float)
    high = (df["high"].astype(float) if "high" in df.columns else close)
    low = (df["low"].astype(float) if "low" in df.columns else close)
    if smooth and smooth > 1:
        high = high.rolling(smooth, center=True, min_periods=1).mean()
        low = low.rolling(smooth, center=True, min_periods=1).mean()

    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    n = len(h)
    raw: list[tuple[int, float, str]] = []
    for i in range(window, n - window):
        seg_h = h[i - window:i + window + 1]
        seg_l = l[i - window:i + window + 1]
        cur_h, cur_l = h[i], l[i]
        if not (math.isnan(cur_h) or math.isnan(cur_l)):
            if cur_h >= float(np.nanmax(seg_h)) - 1e-12:
                raw.append((i, cur_h, "high"))
            elif cur_l <= float(np.nanmin(seg_l)) + 1e-12:
                raw.append((i, cur_l, "low"))
    if not raw:
        return []

    # --- 1) 同类型相邻点去重：保留更极端者 ------------------------------- #
    merged: list[list] = []
    for i, p, t in raw:
        if merged:
            pi, pp, pt, pc = merged[-1]
            same_kind = (pt == t)
            near = (abs(p - pp) / max(abs(pp), 1e-9) < 0.005) and (i - pi <= max(min_gap, 3))
            if same_kind and near:
                # 同为高点取更高、同为低点取更低
                if (t == "high" and p > pp) or (t == "low" and p < pp):
                    merged[-1] = [i, p, t, pc + 1]
                else:
                    merged[-1][3] = pc + 1
                continue
        merged.append([i, p, t, 1])
    seq = [(int(i), float(p), str(t)) for i, p, t, _ in merged]

    # --- 2) 强制交替：连续同类点只保留极端者 ----------------------------- #
    alt: list[tuple[int, float, str]] = []
    for i, p, t in seq:
        if alt and alt[-1][2] == t:
            if (t == "high" and p >= alt[-1][1]) or (t == "low" and p <= alt[-1][1]):
                alt[-1] = (i, p, t)
            continue
        alt.append((i, p, t))

    # --- 3) 距离过滤 ----------------------------------------------------- #
    out: list[dict] = []
    for i, p, t in alt:
        if out and (i - out[-1]["idx"]) < max(1, min_gap // 2):
            continue
        out.append({"idx": int(i), "date": _date_str(df, i), "price": _r(p),
                    "type": t})
    return out[-int(n_max):] if n_max else out


def _zigzag(df: pd.DataFrame, threshold: float | None = None,
            min_pct: float = ELLIOTT_MIN_PCT) -> list[dict]:
    """ZigZag 摆动点提取（用于波浪/扇形/百分比回撤）。

    阈值 = 近 :data:`ELLIOTT_ATR_WINDOW` 日 ATR 均值 × :data:`ELLIOTT_ATR_MULT`，
    且不小于 ``min_pct``（4%）——低波动股用 ATR 自适应，高波动股用百分比兜底，
    避免在 2% 波动的票上画出几十个"浪"。

    Returns
    -------
    list[dict]，按时间升序：``{"idx","date","price","type"}``。
    """
    if df is None or len(df) < 30:
        return []
    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    px = _num(close.iloc[-1], 0.0)

    if threshold is None:
        atr = _atr(df, 14)
        recent = atr.tail(int(ELLIOTT_ATR_WINDOW))
        atr_mean = _num(recent.mean()) if len(recent) else float("nan")
        thr_abs = 0.0 if math.isnan(atr_mean) else atr_mean * ELLIOTT_ATR_MULT
        threshold = max(thr_abs, abs(px) * min_pct)
    if not threshold or threshold <= 0 or math.isnan(threshold):
        threshold = max(abs(px) * min_pct, 1e-6)

    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    n = len(h)
    piv: list[list] = []                       # [[idx, price, "H"/"L"], ...]

    def _push(i: int, p: float, kind: str) -> None:
        """追加一个确认的摆动点；与上一个同类点合并时保留更极端者。"""
        if piv and piv[-1][2] == kind:
            if (kind == "H" and p >= piv[-1][1]) or (kind == "L" and p <= piv[-1][1]):
                piv[-1] = [int(i), float(p), kind]
            return
        piv.append([int(i), float(p), kind])

    # 两状态机：dir=+1 表示当前在上行腿（跟踪最高点），dir=-1 表示下行腿（跟踪最低点）。
    # 反向幅度达到 threshold 才确认拐点，因此 ZigZag 天然过滤阈值以下的噪声。
    i0 = 0
    while i0 < n and (math.isnan(h[i0]) or math.isnan(l[i0])):
        i0 += 1
    if i0 >= n:
        return []
    dir_ = 1
    ext_i, ext_p = i0, float(h[i0])
    for i in range(i0 + 1, n):
        hi, lo = h[i], l[i]
        if math.isnan(hi) or math.isnan(lo):
            continue
        if dir_ > 0:
            if hi > ext_p:                       # 上行腿继续，刷新高点
                ext_i, ext_p = i, float(hi)
            elif ext_p - lo >= threshold:        # 回落超阈值 → 确认高点，转入下行腿
                _push(ext_i, ext_p, "H")
                dir_ = -1
                ext_i, ext_p = i, float(lo)
        else:
            if lo < ext_p:                       # 下行腿继续，刷新低点
                ext_i, ext_p = i, float(lo)
            elif hi - ext_p >= threshold:        # 反弹超阈值 → 确认低点，转入上行腿
                _push(ext_i, ext_p, "L")
                dir_ = 1
                ext_i, ext_p = i, float(hi)
    # 收尾：把尚未被反向确认的最后一个极值也纳入（供"当前处于哪一浪"定位使用）
    _push(ext_i, ext_p, "H" if dir_ > 0 else "L")

    pts: list[dict] = []
    for i, p, kind in piv:
        if math.isnan(p):
            continue
        pts.append({"idx": int(i), "date": _date_str(df, i), "price": _r(p),
                    "type": "high" if kind == "H" else "low"})
    return pts


def _trend_of(df: pd.DataFrame, min_bars: int) -> dict:
    """给周线/月线用的道氏趋势摘要：``{"trend","score"}``。"""
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < min_bars:
        return {"trend": "数据不足", "score": 0.0}
    res = dow_theory(df)
    return {"trend": res.get("trend", "横盘"), "score": _num(res.get("score"), 0.0)}


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """把日线聚合成周/月线（pandas 3.0 安全写法，只用 OHLCV 聚合）。

    注意：pandas 3.0 已移除 ``"M"``/``"W"`` 这类废弃别名，必须用 ``"ME"``/``"W-FRI"``。
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if "date" not in df.columns:
        return pd.DataFrame()
    try:
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.dropna(subset=["date"]).set_index("date").sort_index()
        agg = {"open": "first", "high": "max", "low": "min", "close": "last",
               "volume": "sum"}
        for col in list(agg):
            if col not in tmp.columns:
                agg.pop(col)
        if "close" not in agg:
            return pd.DataFrame()
        out = tmp.resample(rule).agg(agg).dropna(subset=["close"]).reset_index()
        if "date" not in out.columns and out.index.name == "date":
            out = out.reset_index()
        return out
    except Exception as exc:  # noqa: BLE001
        log.debug("重采样 %s 失败: %s", rule, exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# 1. 道氏理论
# --------------------------------------------------------------------------- #
def _ma_alignment(close: pd.Series) -> dict:
    """多周期均线排列打分（道氏"主要趋势"的量化代理）。

    5 项检查（每项 +1 看涨 / -1 看跌，等权投票）：

    1. 现价 > MA250          —— 长期水位之上；
    2. MA250 较 60 日前抬升   —— 长期均线方向（趋势过滤器的核心）；
    3. 现价 > MA120          —— 中期水位之上；
    4. MA60 > MA250          —— 中长周期多头排列；
    5. 现价 > MA60           —— 中短期水位之上。

    判定：``net = (pos - neg) / 总项数``；``net ≥ +0.2`` → +1（上升），
    ``net ≤ -0.2`` → -1（下降），其余 → 0（横盘）。

    选择理由（实测对比）
    --------------------
    在 ``make_ohlcv(900, seed=5, regime="up")`` 上：数据整体从 29.6 涨到 89.2（+200%），
    但末尾一段为 -8.6% 的回调，**最近 60 日收益为 -2.3%、最近两个摆动点是 LH+LL**：
    * 只用摆动点 → 判成"下降"（错，与"主要趋势"定义相悖）；
    * 用 60/120/250 日收益表决 → 均值仅 +0.22，判成"横盘"（偏保守但丢失了主升浪信息）；
    * 用本条**长周期均线水位 + 均线方向**表决 → 5/5 项看涨，正确判为"上升"。
    另在 ``regime="down"`` 的三个种子上均 0/5 项看涨，正确判为"下降"。
    均线水位法对趋势型行情的识别显著更稳，代价是**滞后**（顶部反转约滞后 MA60/MA120
    的 1/2 周期），因此本模块另外用 :func:`_swing_points` 的微观结构给出"次级趋势"，
    并在报告中同时展示，避免用户只看到滞后结论。

    返回 ``strength`` = 净票数占比（-1~1），供打分与置信度使用。
    """
    px = _num(close.iloc[-1], 0.0)
    n = len(close)
    m = close.rolling(250, min_periods=1).mean()
    ma20 = _num(close.rolling(20, min_periods=1).mean().iloc[-1], px)
    ma60 = _num(close.rolling(60, min_periods=1).mean().iloc[-1], px)
    ma120 = _num(close.rolling(120, min_periods=1).mean().iloc[-1], px)
    ma250 = _num(m.iloc[-1], px)
    ma250_prev = _num(m.iloc[-61], ma250) if n > 61 else ma250

    signals: list[tuple[str, int]] = [
        ("现价%sMA250" % (">" if px > ma250 else "≤"), 1 if px > ma250 else -1),
        ("MA250 较60日前%s" % ("抬升" if ma250 > ma250_prev else "下移"),
         1 if ma250 > ma250_prev else -1),
        ("现价%sMA120" % (">" if px > ma120 else "≤"), 1 if px > ma120 else -1),
        ("MA60%sMA250" % (">" if ma60 > ma250 else "≤"), 1 if ma60 > ma250 else -1),
        ("现价%sMA60" % (">" if px > ma60 else "≤"), 1 if px > ma60 else -1),
    ]
    pos = sum(1 for _, v in signals if v > 0)
    neg = sum(1 for _, v in signals if v < 0)
    total = len(signals)
    net = (pos - neg) / float(total)
    # 退化保护：价格几乎不动（如停牌/一字板/常数序列）时，均线比较会全部落到"相等"分支
    #（本实现把相等判为 -1），从而把 net 误算成 -1.0。这里用"现价相对首日的净位移
    # 必须同时满足 >0.5% 与 >1 个最小报价单位"做绝对量兜底。
    px_first = _num(close.iloc[0], px)
    moved = (abs(px - px_first) > 0.005
             and abs(px - px_first) / (abs(px_first) if px_first else 1.0) > 0.005)
    if not moved:
        net = 0.0
    if net >= 0.2:
        direction = 1
    elif net <= -0.2:
        direction = -1
    else:
        direction = 0
    r250 = _num(_ret_pct(close, min(250, max(20, n - 1))), float("nan"))
    return {
        "direction": direction,
        "pos": pos, "neg": neg, "total": total,
        "strength": net,
        "checks": [{"name": nm, "dir": v} for nm, v in signals],
        "_mom20_60": (ma20 - ma60) / (ma60 if ma60 else 1.0),
        "_ma": {"ma20": _r(ma20), "ma60": _r(ma60), "ma120": _r(ma120),
                "ma250": _r(ma250), "ma250_prev60": _r(ma250_prev)},
        "_r250": _r(r250, 2) if not math.isnan(r250) else None,
    }


@_guard(default=dict, label="dow_theory")
def dow_theory(df) -> dict:
    """道氏理论：摆动高低点结构 + 多周期均线排列 + 成交量验证 + 阶段判断。

    判定逻辑
    --------
    1. 提取摆动点（**左右各 3 根**确认的局部高/低点，见 :func:`_swing_points`）；
    2. 取最近 3 个高点与最近 3 个低点，比较"后一个 vs 前一个"：
       HH + HL（高点抬高 且 低点抬高）→ 上升结构；LH + LL → 下降结构；
       两者都不成立 → 转换/盘整；两者同时成立（罕见）→ 结构矛盾。
       **注意**：这两个布尔量描述的是"最近一段"的**次级结构**，不是主要趋势。
    3. **主要趋势**（``trend`` / ``primary_trend``）用多周期均线排列的多方表决
       （MA10/20、MA20/60、MA60/120 或 MA60/首日、250 日净涨跌 |·|>8%，见
       :func:`_ma_alignment`）：正向 ≥2 且无反向 → 上升；反向 ≥2 且无正向 → 下降；
       否则横盘。**参数选择理由**：实测在 `make_ohlcv(900, seed=5, regime="up")` 上，
       数据整体从 29.6 涨到 89.2（+200%），但末尾一段是 -8.6% 的回调，
       若只用最近两个摆动点会判成"下降"，与"主要趋势"的定义相悖；改用多周期均线
       表决后 up/down 两个 regime 都能稳定判对（自测断言即验证此点）。
    4. 次级趋势（``secondary_trend``）：20/60 日均线偏离标准化后的符号与幅度
       （>±0.35 倍 20 日波动才认为有方向），用于区分"主要趋势中的回调"。
    5. 成交量验证：上升趋势要求上涨日均量 ≥ 下跌日均量（近 60 日，量比 ≥ 0.98）；
       下降趋势同理要求缩量反弹（量比 ≤ 1.02）。
    6. 阶段（phase）：底部区间（价格分位 ≤35%）+ 缩量 → 吸筹；
       上升趋势 + 价格分位 >35% → 拉升；高位（分位 ≥70%）+ 放量但 10 日涨幅 <3%（滞涨）
       → 派发；下降趋势 → 下跌；其余 → 整理。

    Returns
    -------
    dict：``trend / primary_trend / secondary_trend / phase / hh_hl / lh_ll /
    volume_confirm / score / confidence / detail / swing_points``
    """
    base = {"trend": "横盘", "primary_trend": "横盘", "secondary_trend": "横盘",
            "phase": "未知", "hh_hl": False, "lh_ll": False, "volume_confirm": False,
            "score": 0.0, "confidence": 0.2, "detail": "数据不足，无法判断道氏趋势",
            "swing_points": []}
    f = _ensure_frame(df)
    if f.empty or len(f) < 40:
        return base

    close = f["close"].astype(float)
    vol = f["volume"].astype(float)
    px = _num(close.iloc[-1], 0.0)
    n = len(f)
    win = min(120, n)
    hi_n = _num(f["high"].tail(win).max(), px)
    lo_n = _num(f["low"].tail(win).min(), px)
    pos = _pct_pos(px, lo_n, hi_n)

    # --- 摆动点结构 ------------------------------------------------------ #
    sw = _swing_points(f, window=DOW_SWING_WINDOW, n_max=40)
    highs = [p for p in sw if p["type"] == "high"]
    lows = [p for p in sw if p["type"] == "low"]
    h_up = h_dn = l_up = l_dn = False
    if len(highs) >= 2:
        h_up = highs[-1]["price"] > highs[-2]["price"] * (1 + DOW_MIN_SWING_PCT)
        h_dn = highs[-1]["price"] < highs[-2]["price"] * (1 - DOW_MIN_SWING_PCT)
    if len(lows) >= 2:
        l_up = lows[-1]["price"] > lows[-2]["price"] * (1 + DOW_MIN_SWING_PCT)
        l_dn = lows[-1]["price"] < lows[-2]["price"] * (1 - DOW_MIN_SWING_PCT)
    hh_hl = bool(h_up and l_up)
    lh_ll = bool(h_dn and l_dn)

    # --- 主要趋势：多周期均线排列（道氏"主要趋势"口径） ------------------ #
    # 只靠最近 2 个高点/低点判定会被"趋势末端的最后一波回调"翻转：
    # 例如 900 根合成数据里整体翻倍、但最后 20 根回撤 8%，微观结构会读成下降趋势。
    # 道氏理论里"主要趋势"本身就是数月级别的方向，因此这里用多周期均线排列
    # （10/20、20/60、60/120 三组 + 250 日净涨跌）做多数表决：≥2 项同向才认定方向。
    align = _ma_alignment(close)
    # --- 次级趋势：20/60 日均线动量的标准化值 ---------------------------- #
    std60 = _num(close.pct_change().tail(60).std(), float("nan"))
    unit = (std60 if not math.isnan(std60) and std60 > 0 else 0.015) * math.sqrt(20)
    momentum = _num(align["_mom20_60"], 0.0) / (unit if unit else 0.015)

    if align["direction"] > 0:
        trend = "上升"
    elif align["direction"] < 0:
        trend = "下降"
    else:
        trend = "横盘"
    if hh_hl and lh_ll:
        structure = "矛盾(HH+HL 与 LH+LL 并存)"
    elif hh_hl:
        structure = "HH+HL" + ("(与主要趋势一致)" if trend == "上升" else "(主要趋势相反)")
    elif lh_ll:
        structure = "LH+LL" + ("(与主要趋势一致)" if trend == "下降" else "(主要趋势相反)")
    else:
        structure = "转换/盘整"

    # --- 成交量验证 ------------------------------------------------------ #
    look = min(60, n - 1)
    seg = f.tail(look + 1).reset_index(drop=True)
    ret1 = seg["close"].astype(float).pct_change()
    up_vol = _num(seg["volume"][ret1 > 0].mean(), float("nan"))
    dn_vol = _num(seg["volume"][ret1 < 0].mean(), float("nan"))
    if math.isnan(up_vol) or math.isnan(dn_vol) or dn_vol <= 0:
        vratio, vol_ok = 1.0, False
    else:
        vratio = up_vol / dn_vol
        vol_ok = True
    if trend == "下降":
        volume_confirm = bool(vol_ok and vratio <= 1.02)
    else:
        volume_confirm = bool(vol_ok and vratio >= 0.98)

    # --- 阶段判定 -------------------------------------------------------- #
    r10 = _num(_ret_pct(close, 10), 0.0)
    vol_ratio = 1.0
    if n >= 60:
        v20 = _num(vol.tail(20).mean(), 0.0)
        v60 = _num(vol.tail(60).mean(), 0.0)
        vol_ratio = (v20 / v60) if v60 > 0 else 1.0
    if pos <= 0.35 and vol_ratio < 0.95:
        phase = "吸筹"
    elif pos >= 0.70 and vol_ratio > 1.2 and r10 < 3.0:
        phase = "派发"
    elif trend == "上升":
        phase = "拉升"
    elif trend == "下降":
        phase = "下跌"
    else:
        phase = "整理"

    # --- 打分/置信度 ----------------------------------------------------- #
    # 主要趋势定符号，多周期一致度与次级趋势定大小，成交量验证微调
    agree = abs(_num(align.get("strength"), 0.0))          # 0~1 的排列一致度
    if trend == "上升":
        score = 0.35 + 0.35 * agree + (0.10 if volume_confirm else -0.05)
        score += 0.10 * min(1.0, max(0.0, momentum))
    elif trend == "下降":
        score = -(0.35 + 0.35 * agree) + (0.10 if volume_confirm else -0.05)
        score += 0.10 * min(1.0, max(0.0, -momentum))
    else:
        score = float(np.clip(momentum * 0.15, -0.20, 0.20))
    score = float(np.clip(score, -0.95, 0.95))

    n_sw = len(highs) + len(lows)
    clarity = min(1.0, abs(momentum) / 1.2) if trend != "横盘" else 0.4
    conf = 0.25 + 0.25 * min(1.0, n_sw / 10.0) + 0.25 * agree + 0.15 * clarity
    if structure.startswith(("HH+HL", "LH+LL")) and "主要趋势相反" not in structure:
        conf += 0.08
    if not volume_confirm and trend != "横盘":
        conf -= 0.08
    confidence = clamp(conf, 0.15, 0.90)

    hn = highs[-1] if highs else None
    ln = lows[-1] if lows else None
    detail = ("主要趋势 %s（多周期均线排列 %d/%d 项同向：%s）｜次级结构 %s：最近高点 "
              "%s(%s)、最近低点 %s(%s)｜20/60 日动量 %+.2f 倍波动｜涨跌量比 %.2f（%s）"
              "→ 阶段 %s" % (
                  trend, (align["pos"] if trend == "上升" else align["neg"]), align["total"],
                  "/".join(c["name"] for c in align["checks"] if c["dir"] == align["direction"])
                  or "无",
                  structure,
                  ("%.2f" % hn["price"]) if hn else "—", hn["date"] if hn else "—",
                  ("%.2f" % ln["price"]) if ln else "—", ln["date"] if ln else "—",
                  momentum, vratio, "验证通过" if volume_confirm else "未验证",
                  phase))

    return {
        "trend": trend,
        "primary_trend": trend,                     # 主要趋势=多周期均线排列表决
        "secondary_trend": ("上升" if momentum > 0.35 else
                            ("下降" if momentum < -0.35 else "横盘")),  # 次级趋势=均线动量
        "phase": phase,
        "hh_hl": bool(hh_hl),
        "lh_ll": bool(lh_ll),
        "volume_confirm": bool(volume_confirm),
        "score": _r(score, 3),
        "confidence": _r(confidence, 3),
        "detail": detail,
        "swing_points": sw[-16:],
        "_structure": structure,
        "_momentum": _r(momentum, 3),
        "_volume_ratio": _r(vratio, 3),
        "_price_position": _r(pos, 3),
        "_alignment": {"direction": align["direction"], "pos": align["pos"],
                       "neg": align["neg"], "total": align["total"],
                       "strength": _r(agree, 3),
                       "checks": align["checks"], "r250": align["_r250"]},
    }


# --------------------------------------------------------------------------- #
# 2. 艾略特波浪理论
# --------------------------------------------------------------------------- #
def _fib_ratio(num: float, den: float) -> float:
    """|num| / |den|，den 为 0 时返回 NaN。"""
    if not den or math.isnan(den) or math.isnan(num):
        return float("nan")
    return abs(num) / abs(den)


def _in_range(x: float, lo: float, hi: float) -> bool:
    return (not math.isnan(x)) and lo <= x <= hi


def _wave_conf(oriented: bool) -> dict:
    """波浪吻合度配置。

    ``impulse`` 表示"当前正在走 5 浪推动的最后一浪"（即 0..5 点已经给出），
    ``up`` 表示推动方向向上（低-高-低-高-低-高）。
    """
    if oriented:
        return {
            "name": "推动浪(5浪)",
            "impulse": True,
            "up": True,
            "w2_band": (0.35, 0.85),
            "w4_band": (0.15, 0.45),
            "w3_min": 1.25,
            "w3_good": 1.618,
            "alternation_ok": True,
            "targets": {
                "5": [("第5浪 = 第1浪", 1.0), ("第5浪 = 1.618×第1浪", 1.618),
                      ("第5浪 = 第3浪×0.618", None)],
                "B": [("B浪反弹 = A浪×0.618", 0.618)],
            },
        }
    return {
        "name": "推动浪(5浪, 向下)",
        "impulse": True,
        "up": False,
        "w2_band": (0.30, 0.85),
        "w4_band": (0.12, 0.45),
        "w3_min": 1.10,
        "w3_good": 1.618,
        "alternation_ok": True,
        "targets": {
            "5": [("第5浪 = 第1浪", 1.0), ("第5浪 = 1.618×第1浪", 1.618)],
            "B": [("B浪反弹 = A浪×0.618", 0.618)],
        },
    }


def _corr_conf(corr_up: bool) -> dict:
    """ABC 调整浪吻合度配置。

    ``corr_up`` 表示**调整浪本身是向上的**（即"低-高-低-高-低-高"的反弹结构），
    它出现在下跌趋势中，因此是**逆势反弹**，浪级方向为 **-1（看跌）**；
    反之 ``corr_up=False`` 是上升趋势中的**回调**，方向为 **+1（看涨）**。

    与推动浪的差别：
    * 不检查"3 浪延伸"与"4 浪不重叠"（调整浪没有这两条铁律）；
    * B 浪回撤 A 浪的容许区间放宽到 0.35~1.25（锯齿/平台/三角形形态差异大）；
    * C 浪与 A 浪的长度比容许 0.5~2.0。
    """
    return {
        "name": "调整浪(ABC-反弹)" if corr_up else "调整浪(ABC-回调)",
        "impulse": False,
        "up": bool(corr_up),
        "w2_band": (0.35, 1.25),      # B 浪回撤 A 浪
        "w4_band": (0.50, 2.00),      # C 浪 / A 浪
        "w3_min": 0.0,                # 调整浪不要求 3 浪延伸
        "w3_good": 0.0,
        "alternation_ok": False,
        "targets": {
            "C": [("C浪 = A浪", 1.0), ("C浪 = 1.618×A浪", 1.618),
                  ("C浪 = 0.618×A浪", 0.618)],
        },
    }


def _eval_wave(d: Sequence[float], conf: dict) -> dict:
    """评估一组 6 个 ZigZag 价格点对某波浪形态的吻合度。

    ``d[0..5]`` = 第 0~5 个摆动点价格；段长
    ``w1=|d1-d0|``、``w2=|d2-d1|``、``w3=|d3-d2|``、``w4=|d4-d3|``、``w5=|d5-d4|``。

    ``conformance`` = 通过校验的项数 / 总项数（题目要求的"吻合比例"），
    其中"交替结构"与"4 浪不重叠 1 浪"是硬性结构条件，不满足直接强惩罚。
    """
    w = [_fib_ratio(d[i + 1] - d[i], 1.0) for i in range(5)]
    w1, w2, w3, w4, w5 = w
    if min(w1, w3) <= 0 or math.isnan(w1) or math.isnan(w3):
        return {"ok": False, "conformance": 0.0, "checks": []}

    checks: list[tuple[str, bool]] = []
    # 1) 交替结构：必须是"低-高-低-高-低-高"（向上）或"高-低-高-低-高-低"（向下）。
    #    注意 ZigZag 天然高低交替，因此相邻两段的**方向必然相反**，
    #    正确写法是"相邻段方向互异"，而不是"所有段同向"。
    kinds = [(d[i + 1] - d[i]) > 0 for i in range(5)]
    if conf["up"]:
        alt_ok = kinds == [True, False, True, False, True]
    else:
        alt_ok = kinds == [False, True, False, True, False]
    checks.append(("交替结构(低-高-低-高-低-高)" if conf["up"]
                   else "交替结构(高-低-高-低-高-低)", bool(alt_ok)))

    # 2) 2 浪回撤（推动浪 0.382~0.786；调整浪 B 浪放宽）
    r2 = _fib_ratio(w2, w1)
    c2 = _in_range(r2, conf["w2_band"][0], conf["w2_band"][1])
    checks.append(("2浪回撤在 %.2f~%.2f 倍1浪(实测%.2f)"
                   % (conf["w2_band"][0], conf["w2_band"][1], r2), bool(c2)))

    # 3) 3 浪延伸（推动浪 ≥1.618 为"好"，≥1.25 算通过；调整浪不看）
    r3 = _fib_ratio(w3, w1)
    c3 = (r3 >= conf["w3_good"]) if conf["impulse"] else True
    c3_soft = (r3 >= conf["w3_min"]) if conf["impulse"] else True
    if conf["impulse"]:
        checks.append(("3浪 ≥ %.2f 倍1浪(实测%.2f)" % (conf["w3_good"], r3), bool(c3)))

    # 4) 4 浪不重叠 1 浪终点（推动浪铁律；调整浪不适用）
    if conf["impulse"] and conf["up"]:
        no_overlap = d[4] > d[1]
    elif conf["impulse"]:
        no_overlap = d[4] < d[1]
    else:
        no_overlap = True
    if conf["impulse"]:
        checks.append(("4浪不与1浪重叠", bool(no_overlap)))

    # 5) 第 5 段与前段的比例（推动浪：5 浪 ≈ 1 浪或 1.618×1 浪；调整浪：C 浪 ≈ A 浪）
    r5a = _fib_ratio(w5, w1)
    r5b = _fib_ratio(w5, w3)
    if conf["impulse"]:
        c5 = (_in_range(r5a, 0.70, 1.85) or _in_range(r5b, 0.50, 1.15))
        checks.append(("5浪≈1浪或1.618×1浪(实测%.2f倍1浪/%.2f倍3浪)" % (r5a, r5b), bool(c5)))
    else:
        c5 = _in_range(r5b, conf["w4_band"][0], conf["w4_band"][1])
        checks.append(("C浪 ≈ %.2f~%.2f 倍A浪(实测%.2f)"
                       % (conf["w4_band"][0], conf["w4_band"][1], r5b), bool(c5)))

    n_pass = sum(1 for _, ok in checks if ok)
    conformance = n_pass / float(len(checks))
    # 硬性条件：交替结构 + 2 浪回撤区间 +（推动浪还要 3 浪延伸与 4 浪不重叠）
    hard_ok = alt_ok and c2 and c3_soft and no_overlap
    return {"ok": bool(hard_ok and conformance >= 0.80),
            "conformance": float(conformance), "checks": checks,
            "ratios": {"w2/w1": _r(r2, 3), "w3/w1": _r(r3, 3),
                       "w4/w3": _r(_fib_ratio(w4, w3), 3),
                       "w5/w1": _r(r5a, 3), "w5/w3": _r(r5b, 3)}}


def _wave_now(d: Sequence[float], conf: dict, depth: float = float("nan")) -> str:
    """根据"当前价在 0..5 号摆动点结构中的位置"估计正在走哪一浪。

    ``d`` 是用于数浪的 6 个点（0..5），**不是** ZigZag 的全部点。

    * 调整浪（ABC）：比较 C 浪（d3→d4）与当前延伸段（d4→d5）的长度，
      ≥0.85 倍视为 C 浪已走完，否则为 C 浪进行中。
    * 推动浪：以整段推进幅度 ``w_total = |d5-d0|`` 为基准，
      当前延伸段（d4→d5）占 ``w_total`` 的比例 ≤38.2%（典型的"4 浪后最后一冲"）
      → 第 5 浪；否则若 ``depth ≥ 1.0``（当前段幅度已超过进入前的那一段，说明是反向的大调整）
      → A 浪（大级别调整初段）；其余 → 第 3 浪途中。
    """
    if len(d) < 6:
        return "未知"
    if not conf.get("impulse"):
        # ABC：比较 C 浪（d3→d4）与当前延伸段（d4→d5）的长度，
        # C 浪已走完 ~85% 以上就认为处于 C 浪末端，否则还在 C 浪途中。
        wc = abs(d[4] - d[3])
        wcur = abs(d[5] - d[4])
        if wc > 0 and wcur / wc >= 0.85:
            return "C"
        return "C(进行中)"
    w_leg = abs(d[5] - d[4])                    # 当前第 5 段
    w_total = abs(d[5] - d[0])                  # 整段推进幅度
    frac = (w_leg / w_total) if w_total > 0 else 1.0
    if frac <= 0.382:
        return "5"
    if not math.isnan(depth) and depth >= 1.0:
        return "A"
    return "3"


def _elliott_targets(conf: dict, d: Sequence[float], wave_now: str) -> list[dict]:
    """给出后续目标位（斐波那契投影，带中文推导依据）。"""
    out: list[dict] = []
    if len(d) < 6:
        return out
    w1 = abs(d[1] - d[0])
    w3 = abs(d[3] - d[2])
    w5 = abs(d[5] - d[4])
    up = conf["up"]
    sign = 1.0 if up else -1.0
    if conf["impulse"]:
        # 若还在 3 浪途中（3 浪未达 1.618 倍 1 浪）：目标 = 1 浪长 × 1.618 + 2 浪底
        if wave_now in ("1", "3"):
            out.append({"label": "第3浪目标(1.618×1浪)",
                        "price": _r(d[2] + sign * w1 * 1.618),
                        "basis": "第3浪 ≈ 1.618×第1浪，自第2浪终点(%s)起算" % _r(d[2])})
            out.append({"label": "第3浪目标(1.0×1浪)",
                        "price": _r(d[2] + sign * w1),
                        "basis": "保守情形：第3浪 ≈ 第1浪"})
        # 已进入第 5 浪（或 5 浪刚确认）：给出 5 浪目标与 5 浪后的回撤参考
        if wave_now == "5":
            out.append({"label": "第5浪目标(1.0×1浪)",
                        "price": _r(d[4] + sign * w1),
                        "basis": "第5浪 ≈ 第1浪，自第4浪终点(%s)起算" % _r(d[4])})
            out.append({"label": "第5浪目标(1.618×1浪)",
                        "price": _r(d[4] + sign * w1 * 1.618),
                        "basis": "乐观情形：第5浪 = 1.618×第1浪"})
            if w3 > 0:
                out.append({"label": "第5浪目标(0.618×3浪)",
                            "price": _r(d[4] + sign * w3 * 0.618),
                            "basis": "第5浪 ≈ 0.618×第3浪"})
            out.append({"label": "5浪后回撤参考(0.382)",
                        "price": _r(d[5] - sign * abs(d[5] - d[0]) * 0.382),
                        "basis": "5 浪走完后的常见回撤位，供止盈参考"})
    else:
        a = abs(d[5] - d[4])
        for mult, tag in ((1.0, "C浪 = A浪"), (1.618, "C浪 = 1.618×A浪"),
                          (0.618, "C浪 = 0.618×A浪")):
            out.append({"label": tag, "price": _r(d[5] + sign * a * mult),
                        "basis": "ABC 调整浪的 C 浪长度通常是 A 浪的 %.3f 倍" % mult})
    return [t for t in out if t["price"] is not None]


@_guard(default=dict, label="elliott_wave")
def elliott_wave(df) -> dict:
    """艾略特波浪理论：ZigZag 数浪 + 斐波那契比例校验。

    判定逻辑
    --------
    1. ZigZag 阈值 = 近 60 日 ATR 均值 × 2.5，且不小于 4%（见 :func:`_zigzag`）；
    2. 取最近 :data:`ELLIOTT_N_SWINGS` 个摆动点，尝试 4 种数法：
       ① 向上 5 浪推动 ② 向下 5 浪推动 ③ 向上 ABC ④ 向下 ABC；
    3. 每种数法按 5 项校验计算吻合比例（``fib_conformance``）：
       交替结构 / 2 浪回撤 0.382~0.786 倍 1 浪 / 3 浪 ≥ 1.618 倍 1 浪 /
       4 浪不重叠 1 浪终点 / 5 浪 ≈ 1 浪或 1.618 倍 1 浪；
    4. 取吻合度最高者；**若吻合度 < 0.60 或硬性结构条件不满足，直接返回
       ``wave="unknown"``**（宁可保守也不乱报浪级）。

    Returns
    -------
    dict：``wave / direction / confidence / swing_points / targets /
    fib_conformance / detail``
    """
    fallback = {"wave": "未知", "direction": 0, "confidence": 0.0,
                "swing_points": [], "targets": [], "fib_conformance": 0.0,
                "detail": "波浪结构不成立或数据不足，不做浪级判断"}
    f = _ensure_frame(df)
    if f.empty or len(f) < 60:
        fallback["detail"] = "数据不足 60 根，无法数浪"
        return fallback

    zz = _zigzag(f)
    if len(zz) < 6:
        fallback["detail"] = "ZigZag 摆动点不足 6 个（波动幅度未达阈值），无法数浪"
        fallback["swing_points"] = zz[-8:]
        return fallback
    seg = zz[-ELLIOTT_N_SWINGS:] if len(zz) >= ELLIOTT_N_SWINGS else zz[-6:]
    d = [float(p["price"]) for p in seg]

    # ABC 调整浪深度过滤：调整幅度必须相对"进入调整前的推进段"合理（5%~150%）。
    # 这一步很关键——否则一个 1% 的小抖动也会被数成 ABC 调整浪，导致结论噪声化。
    prev_span = abs(seg[-1]["price"] - seg[0]["price"])
    prior = zz[-(len(seg) + 1)]["price"] if len(zz) > len(seg) else seg[0]["price"]
    prior_leg = abs(float(seg[0]["price"]) - float(prior))
    depth = (prev_span / prior_leg) if prior_leg > 0 else float("nan")
    corr_ok = bool(not math.isnan(depth) and 0.05 <= depth <= 1.5)

    confs = [_wave_conf(True), _wave_conf(False)]
    if corr_ok:
        confs += [_corr_conf(True), _corr_conf(False)]

    best: dict | None = None
    for conf in confs:
        ev = _eval_wave(d, conf)
        if ev["ok"] and (best is None or ev["conformance"] > best["ev"]["conformance"]):
            best = {"conf": conf, "ev": ev}

    px = _num(f["close"].iloc[-1], 0.0)
    base = {
        "wave": "未知", "direction": 0, "confidence": 0.0,
        "swing_points": seg, "targets": [], "fib_conformance": 0.0,
        "detail": "",
    }
    if best is None:
        # 全部数法都不达标 → 给出最强的一种作为参考信息，但仍返回 unknown
        scored = [(_eval_wave(d, c), c) for c in confs]
        ev, conf = max(scored, key=lambda t: t[0]["conformance"])
        base["fib_conformance"] = _r(ev["conformance"], 3)
        base["detail"] = ("最近 %d 个 ZigZag 摆动点不符合任何标准浪型（最佳吻合度 %.0f%%：%s），"
                          "按保守原则返回「未知」。当前参考比例 %s" % (
                              len(seg), ev["conformance"] * 100, conf["name"],
                              ", ".join("%s=%.2f" % kv for kv in (ev.get("ratios") or {}).items())))
        return base

    conf, ev = best["conf"], best["ev"]
    wave_now = _wave_now(d, conf, depth)
    # 方向语义：
    #   推动浪(向上) = +1；推动浪(向下) = -1；
    #   调整浪(回调，即"高-低-高-低-高-低"，是上升趋势中的回调) → 趋势延续向上 = +1；
    #   调整浪(反弹，即"低-高-低-高-低-高"，是下跌趋势中的反弹) → 趋势延续向下 = -1。
    direction = 1 if conf["up"] else -1
    targets = _elliott_targets(conf, d, wave_now)
    # 置信度：吻合度为主，摆动点数量（ZigZag 充分度）为辅
    q = min(1.0, len(seg) / float(ELLIOTT_N_SWINGS))
    base_conf = 0.72 if conf["impulse"] else 0.60     # 调整浪的可信度低于推动浪
    confidence = clamp(0.15 + (base_conf - 0.15) * ev["conformance"] + 0.15 * q, 0.1, 0.72)
    detail = ("数浪结论：%s，当前处于第 %s 浪（%s）｜斐波那契吻合度 %.0f%%"
              "（2浪/1浪=%.2f，3浪/1浪=%.2f，5浪/1浪=%.2f）｜目标位 %s" % (
                  conf["name"], wave_now, "看涨" if direction > 0 else "看跌",
                  ev["conformance"] * 100,
                  (ev.get("ratios") or {}).get("w2/w1", float("nan")),
                  (ev.get("ratios") or {}).get("w3/w1", float("nan")),
                  (ev.get("ratios") or {}).get("w5/w1", float("nan")),
                  "、".join("%s→%.2f" % (t["label"], t["price"]) for t in targets[:3]) or "无"))
    return {
        "wave": str(wave_now),
        "direction": int(direction),
        "confidence": _r(confidence, 3),
        "swing_points": seg,
        "targets": targets,
        "fib_conformance": _r(ev["conformance"], 3),
        "detail": detail,
        "_pattern": conf["name"],
        "_current_price": _r(px, 2),
        "_checks": [{"name": n, "pass": bool(ok)} for n, ok in ev["checks"]],
    }


# --------------------------------------------------------------------------- #
# 3. 斐波那契分析
# --------------------------------------------------------------------------- #
def _recent_swing_for_anchor(f: pd.DataFrame) -> dict | None:
    """找最近一次"显著性高低点"作为时间窗锚点。

    做法：先取最近 250 日的最高/最低点；若最近一个 ZigZag 摆动点比它们更晚且幅度
    足够（ZigZag 自带 4% 阈值），则优先用它——它的时间戳更接近当下，时间窗更有意义。
    """
    zz = _zigzag(f)
    if zz:
        return zz[-1]
    win = f.tail(250)
    if win.empty:
        return None
    hi_i = int(win["high"].astype(float).idxmax()) if "high" in win else len(f) - 1
    lo_i = int(win["low"].astype(float).idxmin()) if "low" in win else len(f) - 1
    if hi_i >= lo_i:
        return {"idx": hi_i, "date": _date_str(f, hi_i),
                "price": _r(_num(f["high"].iloc[hi_i])), "type": "high"}
    return {"idx": lo_i, "date": _date_str(f, lo_i),
            "price": _r(_num(f["low"].iloc[lo_i])), "type": "low"}


def _future_trading_days(last_date: pd.Timestamp, n: int,
                         holidays: Iterable[str] | None = None) -> list[pd.Timestamp]:
    """从 ``last_date`` 之后推算 n 个交易日（跳过周末，可选跳过指定日期）。"""
    hol = set()
    for h in (holidays or ()):
        try:
            hol.add(pd.Timestamp(h).normalize())
        except Exception:  # noqa: BLE001
            continue
    out: list[pd.Timestamp] = []
    d = pd.Timestamp(last_date).normalize()
    guard = 0
    while len(out) < max(0, int(n)) and guard < int(n) * 4 + 40:
        guard += 1
        d = d + pd.Timedelta(days=1)
        if d.weekday() >= 5:
            continue
        if d in hol:
            continue
        out.append(d)
    return out


@_guard(default=dict, label="fibonacci_analysis")
def fibonacci_analysis(df) -> dict:
    """斐波那契回撤/扩展 + 斐波那契时间窗。

    判定逻辑
    --------
    1. 取最近 250 日（不足则全部）的最高价 H 与最低价 L，以及谁先出现决定方向：
       低点在前 → 上升波段（回撤位是**支撑**）；高点在前 → 下降波段（回撤位是**压力**）；
    2. 回撤位：``P(r) = H - (H-L)*r``，r ∈ {0.191,0.236,0.382,0.5,0.618,0.786,0.809}；
       扩展位：``P(k) = L + (H-L)*k``，k ∈ {1.272,1.618,2.0,2.618}；
    3. 按现价落在哪两个档位之间给出 ``current_zone``；
    4. 时间窗：从最近一次显著性高低点起算，第 5/8/13/21/34/55/89/144 个**交易日**
       （``config.INDICATOR_PARAMS["time_window_bars"]``）为变盘窗口，输出未来 60 个
       交易日内的窗口日期。

    Returns
    -------
    dict：``high / low / direction / retracement / extension / current_zone /
    nearest_support / nearest_resistance / time_windows / score / detail``
    """
    fallback = {"high": None, "low": None, "direction": 0, "retracement": {},
                "extension": {}, "current_zone": "中位区", "nearest_support": None,
                "nearest_resistance": None, "time_windows": [], "score": 0.0,
                "detail": "数据不足，无法计算斐波那契位"}
    f = _ensure_frame(df)
    if f.empty or len(f) < 40:
        return fallback

    win = f.tail(250)
    hi = _num(win["high"].max())
    lo = _num(win["low"].min())
    if math.isnan(hi) or math.isnan(lo) or hi <= lo:
        return fallback
    i_hi = int(win["high"].astype(float).idxmax())
    i_lo = int(win["low"].astype(float).idxmin())
    direction = 1 if i_lo < i_hi else -1        # 低点在前 = 上升波段
    rng = hi - lo
    close = f["close"].astype(float)
    px = _num(close.iloc[-1], (hi + lo) / 2.0)

    retr = {("%.3f" % r): _r(hi - rng * r) for r in FIB_RATIOS}
    exts = {("%.3f" % k): _r(lo + rng * k) for k in FIB_EXT_RATIOS}
    # 上升波段再补一个 1.0 位（= 前高）作为压力参考
    exts["1.000"] = _r(hi)

    # --- 区间判定 -------------------------------------------------------- #
    levels = [(r, hi - rng * r) for r in sorted(FIB_RATIOS)]     # 由低到高
    if px <= lo + rng * 0.10:
        zone = "强支撑区"
    elif px >= hi - rng * 0.10:
        zone = "强压力区"
    elif px >= lo + rng * 0.618:
        zone = "支撑区"
    elif px >= lo + rng * 0.382:
        zone = "中位区"
    else:
        zone = "压力区"

    below = [(r, p) for r, p in levels if p <= px]
    above = [(r, p) for r, p in levels if p > px]
    sup = max(below, key=lambda t: t[1]) if below else (None, lo)
    res = min(above, key=lambda t: t[1]) if above else (None, hi)
    nearest_support = _r(sup[1])
    nearest_resistance = _r(res[1])

    # --- 时间窗 ---------------------------------------------------------- #
    anchor = _recent_swing_for_anchor(f)
    last = _last_date(f)
    windows: list[dict] = []
    if anchor:
        a_idx = int(anchor["idx"])
        for bars in sorted(set(TIME_WINDOW_BARS)):
            n_ahead = a_idx + int(bars) - (len(f) - 1)
            if 0 < n_ahead <= 60:
                dts = _future_trading_days(last, n_ahead)
                if dts:
                    windows.append({"bars_ahead": int(n_ahead),
                                    "date": dts[-1].strftime("%Y-%m-%d"),
                                    "label": "斐波那契时间窗 %d" % int(bars),
                                    "significant_bool": True})
    windows.sort(key=lambda w: w["bars_ahead"])

    # --- 打分 ------------------------------------------------------------ #
    # 站在回撤位附近有支撑/压力意义：靠近支撑轻微看多，靠近压力轻微看空
    r_now = (hi - px) / rng
    dist_sup = abs(px - nearest_support) / rng if nearest_support else 1.0
    dist_res = abs(nearest_resistance - px) / rng if nearest_resistance else 1.0
    raw = (dist_res - dist_sup) * 0.6
    if r_now <= 0.236:
        raw -= 0.25          # 已在高位，追高风险
    elif r_now >= 0.786:
        raw += 0.25          # 深度回撤，超跌反弹概率
    score = float(np.clip(raw, -0.9, 0.9))

    detail = ("250日区间 %.2f~%.2f（%s波段）｜现价 %.2f 位于 %s（回撤 %.1f%%）｜"
              "最近支撑 %.2f / 最近压力 %.2f｜未来 60 交易日内有 %d 个斐波那契时间窗%s" % (
                  lo, hi, "上升" if direction > 0 else "下降", px, zone, r_now * 100,
                  nearest_support if nearest_support else float("nan"),
                  nearest_resistance if nearest_resistance else float("nan"),
                  len(windows),
                  ("（最近：" + windows[0]["date"] + "）") if windows else ""))
    return {
        "high": _r(hi), "low": _r(lo), "direction": int(direction),
        "retracement": retr, "extension": exts, "current_zone": zone,
        "nearest_support": nearest_support, "nearest_resistance": nearest_resistance,
        "time_windows": windows, "score": _r(score, 3), "detail": detail,
        "_retracement_pct": _r(r_now * 100, 2),
        "_anchor": anchor,
        "_price": _r(px),
    }


# --------------------------------------------------------------------------- #
# 4. 江恩理论
# --------------------------------------------------------------------------- #
def _nan_to(v: float) -> Any:
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v


@_guard(default=dict, label="gann_analysis")
def gann_analysis(df) -> dict:
    """江恩理论：角度线 / 时间周期 / 四方形（Square of 9）。

    .. warning::
       本函数实现的是**经验性方法，置信度低**。江恩理论没有统计显著的实证支持，
       角度线的"单位价格"选择带有人为性，时间周期更接近"数字命理"。
       报告中必须与其它证据交叉验证后再使用。

    判定逻辑
    --------
    1. 锚点：最近一个显著性摆动点（低点画支撑扇形，高点画压力扇形）；
       角度基准 ``unit = 锚点前 60 日 ATR × 1.5``，保证 60 根 K 线内角度线有区分度；
    2. 江恩角度线：斜率 = unit × ratio，ratio 取 ``config.INDICATOR_PARAMS["gann_ratios"]``
       （0.125/0.25/…/0.875 表示 1×8、4×1、…、8×1 等），从锚点按交易日投影出
       ``support_price``（向上投影）与 ``resistance_price``（向下投影）；
    3. 时间周期：30/45/60/90/120/144/180/270/360 **自然日**，换算成交易日并给出日期；
    4. 四方形：``sqrt(现价) ± n × step`` 再平方还原，step 取近期波动单位；
    5. ``score`` 与 ``confidence`` 都被显著压制（confidence ≤ 0.45）。

    Returns
    -------
    dict：``gann_angles / gann_fan / square_of_9 / time_cycles / seasonal_date /
    score / confidence / detail``
    """
    fallback = {"gann_angles": [], "gann_fan": [], "square_of_9": [],
                "time_cycles": [], "seasonal_date": None, "score": 0.0,
                "confidence": 0.15,
                "detail": "数据不足，无法计算江恩角度线/时间周期（经验性方法，置信度低）"}
    f = _ensure_frame(df)
    if f.empty or len(f) < 60:
        return fallback

    close = f["close"].astype(float)
    px = _num(close.iloc[-1], 0.0)
    last = _last_date(f)
    anchor = _recent_swing_for_anchor(f)
    if anchor is None:
        return fallback
    a_idx, a_price = int(anchor["idx"]), _num(anchor["price"], px)
    is_low = anchor["type"] == "low"

    # 角度单位：ATR×1.5（经验性选择，见 docstring 的 warning）
    atr_series = _atr(f, 14)
    seg = atr_series.iloc[max(0, a_idx - 60):a_idx + 1]
    atr_anchor = _num(seg.mean()) if len(seg) else float("nan")
    if math.isnan(atr_anchor) or atr_anchor <= 0:
        atr_anchor = max(abs(px) * 0.02, 1e-6)
    unit = atr_anchor * 1.5

    bars_since = max(1, len(f) - 1 - a_idx)
    ratios = tuple(config.INDICATOR_PARAMS.get("gann_ratios",
                                               (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)))
    angles: list[dict] = []
    fan: list[dict] = []
    for r in ratios:
        slope = unit * float(r) / 8.0        # 1×1 基准 = unit/8（8×1 恰好等于 unit）
        up_v = a_price + slope * bars_since
        dn_v = a_price - slope * bars_since
        label = _angle_label(float(r))
        angles.append({
            "ratio": round(float(r), 4),
            "angle_label": "1×%d" % round(8.0 / float(r)) if r > 0 else "1×1",
            "slope": _r(slope, 6),
            "support_price": _r(up_v if is_low else dn_v),
            "resistance_price": _r(dn_v if is_low else up_v),
            "projected_bars": int(bars_since),
        })
        fan.append({"ratio": round(float(r), 4),
                    "value": _r(up_v if is_low else dn_v)})
    angles.sort(key=lambda a: a["ratio"])
    fan.sort(key=lambda a: a["ratio"])

    # --- 四方形 Square of 9 ---------------------------------------------- #
    step = max(unit, abs(px) * 0.005, 1e-6)
    sq9: list[dict] = []
    rt = math.sqrt(max(px, 1e-9))
    for n in SQUARE9_STEPS:
        sq9.append({"price": _r((rt - n * (step ** 0.5) * 0.5) ** 2), "kind": "support"})
    for n in SQUARE9_STEPS:
        sq9.append({"price": _r((rt + n * (step ** 0.5) * 0.5) ** 2), "kind": "resistance"})

    # --- 时间周期（自然日 -> 交易日） ------------------------------------ #
    cycles: list[dict] = []
    for cyc in (30, 45, 60, 90, 120, 144, 180, 270, 360):
        target = pd.Timestamp(anchor["date"]) + pd.Timedelta(days=int(cyc))
        bars = max(0, int(round(int(cyc) * 5.0 / 7.0)))
        ahead = bars - bars_since
        hit = ahead >= 0
        cycles.append({"cycle": int(cyc),
                       "date": target.strftime("%Y-%m-%d") if anchor.get("date") else "",
                       "bars": int(bars),
                       "bars_ahead": int(ahead),
                       "kind": "已过" if not hit else ("临近" if ahead <= 10 else "未来"),
                       "natural_days": int(cyc)})
    cycles.sort(key=lambda c: c["cycle"])

    # --- 季节性日期（经验值：3/21、6/21、9/23、12/21 为江恩重要自然日） --- #
    try:
        y = int(last.year)
        marks = [pd.Timestamp("%d-03-21" % y), pd.Timestamp("%d-06-21" % y),
                 pd.Timestamp("%d-09-23" % y), pd.Timestamp("%d-12-21" % y),
                 pd.Timestamp("%d-01-20" % (y + 1)), pd.Timestamp("%d-03-21" % (y + 1))]
        nxt = min([m for m in marks if m >= last], default=marks[-1])
        seasonal_date = nxt.strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        seasonal_date = None

    # --- 打分（被刻意压低：经验性方法） ---------------------------------- #
    near_sup = min((abs((a["support_price"] if a["support_price"] is not None else px) - px)
                    for a in angles), default=float("inf"))
    near_res = min((abs((a["resistance_price"] if a["resistance_price"] is not None else px) - px)
                    for a in angles), default=float("inf"))
    if math.isinf(near_sup) or math.isinf(near_res):
        score = 0.0
    else:
        score = float(np.clip((near_res - near_sup) / max(unit, 1e-9) * 0.25, -0.6, 0.6))
    # 现价在锚点之上（低点起涨）略偏多
    if is_low:
        score = float(np.clip(score + 0.15, -0.6, 0.6))
    else:
        score = float(np.clip(score - 0.15, -0.6, 0.6))

    near_cycle = [c for c in cycles if 0 <= c["bars_ahead"] <= 10]
    detail = ("江恩角度线以 %s 为锚（%s，%s，距今 %d 根 K 线，单位 %.3f/日）｜"
              "四方形给出 %d 个支撑/%d 个压力｜时间周期中 %d 个在未来 10 个交易日内｜"
              "**注意：江恩理论属经验性方法，置信度低，仅作弱参考**" % (
                  anchor["date"], "低点" if is_low else "高点", a_price, bars_since, unit,
                  len(sq9) // 2, len(sq9) - len(sq9) // 2, len(near_cycle)))
    return {
        "gann_angles": angles,
        "gann_fan": fan,
        "square_of_9": sq9,
        "time_cycles": cycles,
        "seasonal_date": seasonal_date,
        "score": _r(score, 3),
        "confidence": _r(clamp(0.15 + 0.2 * min(1.0, bars_since / 60.0), 0.1, 0.45), 3),
        "detail": detail,
        "_anchor": {"idx": a_idx, "date": anchor["date"], "price": a_price,
                    "type": anchor["type"]},
        "_unit": _r(unit, 6),
        "_near_cycle": [c["cycle"] for c in near_cycle],
    }


def _angle_label(ratio: float) -> str:
    """江恩比率 -> 经验角度标签（ratio 越小线越陡，标注为 1×8 等）。"""
    pairs = ((0.125, "1×8"), (0.25, "1×4"), (0.375, "1×2.7"), (0.5, "1×2"),
             (0.625, "1×1.6"), (0.75, "1×1.33"), (0.875, "1×1.14"))
    best = min(pairs, key=lambda p: abs(p[0] - ratio))
    return best[1]


# --------------------------------------------------------------------------- #
# 5. 量价时空
# --------------------------------------------------------------------------- #
# 量价 8 种经典配合：(量态, 价态) -> (中文名, 方向, 说明)
_VP_MATRIX = {
    ("increase", "up"): ("量增价涨", 1, "资金主动做多，量价齐升，趋势健康的标志"),
    ("increase", "flat"): ("量增价平", -1, "放量滞涨，多空分歧加剧，警惕主力借量出货"),
    ("increase", "down"): ("量增价跌", -1, "放量下跌，恐慌抛售或利空兑现，短线杀伤力最大"),
    ("shrink", "up"): ("量缩价涨", -1, "无量上涨属量价背离，追高动能不足，随时可能回落"),
    ("shrink", "flat"): ("量缩价平", 0, "多空观望，等待方向选择的变盘前夜"),
    ("shrink", "down"): ("量缩价跌", 1, "缩量阴跌说明抛压衰竭（惜售），往往是底部区域特征"),
    ("flat", "up"): ("量平价涨", 1, "量能平稳而价格上行，温和上涨可持续"),
    ("flat", "down"): ("量平价跌", -1, "量能平稳而价格下行，阴跌格局未改"),
}


def _vol_stage(ratio: float) -> str:
    """量能阶段（近 5 日均量 / 近 60 日均量）。"""
    if math.isnan(ratio):
        return "温和"
    if ratio < 0.60:
        return "地量"
    if ratio < 1.00:
        return "温和"
    if ratio < 1.80:
        return "放量"
    if ratio < 3.00:
        return "巨量"
    return "天量"


@_guard(default=dict, label="volume_price_time")
def volume_price_time(df) -> dict:
    """量价时空：量价 8 种经典配合 + 时间维度（斐波那契时间窗 / 涨跌持续期）。

    判定逻辑
    --------
    1. 量态：近 5 日均量 / 近 60 日均量 → 地量/温和/放量/巨量/天量；
    2. 价态：近 5 日涨幅与价格在 120 日区间中的位置 → 底部/上涨中/顶部/下跌中；
    3. 量价组合：按 8 种经典配合给出中文名、方向与解读；
    4. 时间维度：是否落在斐波那契时间窗附近，以及当前涨/跌持续天数是否超过
       该股历史同类段的均值（超时 → 变盘概率上升）；
    5. ``match`` 表示"量价与时间维度形成共振"（极值量能 + 时间窗临近）。

    Returns
    -------
    dict：``volume_stage / price_stage / time_stage / match / patterns /
    score / confidence / detail``
    """
    fallback = {"volume_stage": "温和", "price_stage": "底部", "time_stage": "未到",
                "match": False, "patterns": [], "score": 0.0, "confidence": 0.2,
                "detail": "数据不足，无法判断量价时空配合"}
    f = _ensure_frame(df)
    if f.empty or len(f) < 60:
        return fallback

    close = f["close"].astype(float)
    vol = f["volume"].astype(float)
    n = len(f)
    v5 = _num(vol.tail(5).mean(), 0.0)
    v60 = _num(vol.tail(60).mean(), 0.0)
    vratio = (v5 / v60) if v60 > 0 else float("nan")
    volume_stage = _vol_stage(vratio)

    r5 = _num(_ret_pct(close, 5), 0.0)
    ma20 = _num(_sma(close, 20).iloc[-1], _num(close.iloc[-1], 0.0))
    px = _num(close.iloc[-1], 0.0)
    win = f.tail(min(120, n))
    hi_n = _num(win["high"].max(), px)
    lo_n = _num(win["low"].min(), px)
    pos = _pct_pos(px, lo_n, hi_n)
    if pos <= 0.30:
        price_stage = "底部"
    elif pos >= 0.75:
        price_stage = "顶部"
    elif r5 >= 0 and px >= ma20:
        price_stage = "上涨中"
    else:
        price_stage = "下跌中"

    # --- 量价 8 种配合 --------------------------------------------------- #
    if math.isnan(vratio):
        v_kind = "flat"
    elif vratio >= 1.25:
        v_kind = "increase"
    elif vratio <= 0.80:
        v_kind = "shrink"
    else:
        v_kind = "flat"
    if r5 >= 2.0:
        p_kind = "up"
    elif r5 <= -2.0:
        p_kind = "down"
    else:
        p_kind = "flat"
    name, direction, why = _VP_MATRIX.get((v_kind, p_kind), ("量价中性", 0, "量价无明显配合特征"))
    patterns = [{"name": name, "direction": int(direction),
                 "detail": "%s（近5日量比 %.2f，近5日涨幅 %+.2f%%）" % (why, vratio, r5)}]

    # --- 时间维度 -------------------------------------------------------- #
    fib = fibonacci_analysis(f)
    windows = fib.get("time_windows") or []
    in_window = bool(windows and windows[0].get("bars_ahead", 99) <= 3)
    # 当前连续同向天数 vs 历史均值
    ret1 = close.pct_change()
    sign = 0
    streak = 0
    for v in reversed(ret1.tail(120).tolist()):
        if v is None or math.isnan(v):
            break
        s = 1 if v > 0 else (-1 if v < 0 else 0)
        if s == 0:
            break
        if sign == 0:
            sign = s
        if s != sign:
            break
        streak += 1
    runs: list[int] = []
    cur = 0
    for v in ret1.tail(250).tolist():
        if v is None or math.isnan(v) or v == 0:
            cur = 0
            continue
        cur += 1
        runs.append(cur)
    avg_run = float(np.mean(runs)) if runs else 0.0
    run_stale = bool(avg_run > 0 and streak >= max(3.0, avg_run * 1.1))
    time_stage = "时间窗临近" if (in_window or run_stale) else "未到"
    if in_window or run_stale:
        patterns.append({
            "name": "时间窗/持续期提示", "direction": int(-direction) if direction else 0,
            "detail": ("%s；当前%s已持续 %d 个交易日（该股历史均值 %.1f 日），"
                       "趋势延续动能趋于衰竭" % (
                           "处于斐波那契时间窗（%s）" % windows[0].get("date") if in_window else "未处时间窗",
                           "连涨" if sign > 0 else "连跌", streak, avg_run))})

    match = bool(in_window and volume_stage in ("巨量", "天量", "地量"))
    # --- 打分 ------------------------------------------------------------ #
    base = 0.45 * direction
    if volume_stage in ("巨量", "天量"):
        base *= 0.7                     # 极端量能时信号可靠性下降
    if price_stage == "顶部" and direction > 0:
        base -= 0.2
    if price_stage == "底部" and direction < 0 and name == "量缩价跌":
        base += 0.25                    # 地量地价 = 底部特征
    if in_window:
        base *= 0.85
    score = float(np.clip(base, -0.9, 0.9))
    conf = clamp(0.3 + 0.2 * (0.0 if math.isnan(vratio) else 1.0)
                 + (0.15 if len(f) >= 250 else 0.0) + (0.1 if not in_window else -0.05),
                 0.2, 0.75)
    detail = ("量能 %s（近5日/60日量比 %.2f）｜价格处于 %s（120日区间分位 %.0f%%）｜"
              "量价配合「%s」%s｜时间维度 %s%s" % (
                  volume_stage, vratio, price_stage, pos * 100, name,
                  "看涨" if direction > 0 else ("看跌" if direction < 0 else "中性"),
                  time_stage,
                  "（量能与时间窗共振，变盘概率上升）" if match else ""))
    return {
        "volume_stage": volume_stage, "price_stage": price_stage,
        "time_stage": time_stage, "match": match, "patterns": patterns,
        "score": _r(score, 3), "confidence": _r(conf, 3), "detail": detail,
        "_volume_ratio": _r(vratio, 3), "_ret5": _r(r5, 2),
        "_price_position": _r(pos, 3), "_streak": int(streak),
        "_avg_run": _r(avg_run, 2), "_in_window": in_window,
    }


# --------------------------------------------------------------------------- #
# 6. 扇形原理
# --------------------------------------------------------------------------- #
@_guard(default=dict, label="fan_principle")
def fan_principle(df) -> dict:
    """扇形原理：从关键点画 1×1 / 2×1 / 3×1 / 4×1 / 8×1 角度线，看现价在哪两条线之间。

    判定逻辑
    --------
    1. 取最近一个显著性摆动点作为原点（低点 → 向上扇形；高点 → 向下扇形）；
    2. 斜率基准 = 原点前 60 日 ATR（1×1 = ATR/日），比率越大线越陡
       （2×1 = 2 单位价格/1 单位时间）；
    3. 计算每条角度线在"今天"的价位，按价格排序后定位现价所处的相邻两条线；
    4. 与 1×1 线的相对位置决定强弱：站上 1×1 且处于 2×1 之上 → 强势；
       跌破 1×1 → 扇形失效，趋势转弱（经典的"扇形原理"用法）。

    Returns
    -------
    dict：``origin / lines / current_line / position / score / detail``
    """
    fallback = {"origin": {}, "lines": [], "current_line": "1x1", "position": "上方",
                "score": 0.0, "detail": "数据不足，无法绘制扇形角度线"}
    f = _ensure_frame(df)
    if f.empty or len(f) < 40:
        return fallback
    anchor = _recent_swing_for_anchor(f)
    if anchor is None:
        return fallback
    a_idx = int(anchor["idx"])
    a_price = _num(anchor["price"], _num(f["close"].iloc[-1], 0.0))
    is_low = anchor["type"] == "low"
    px = _num(f["close"].iloc[-1], a_price)

    atr_series = _atr(f, 14)
    seg = atr_series.iloc[max(0, a_idx - 60):a_idx + 1]
    atr_a = _num(seg.mean()) if len(seg) else float("nan")
    if math.isnan(atr_a) or atr_a <= 0:
        atr_a = max(abs(px) * 0.02, 1e-6)
    step = atr_a * (1.0 if is_low else -1.0)

    bars_since = max(1, len(f) - 1 - a_idx)
    lines: list[dict] = []
    # 比率 = 每单位时间走过的价格单位数；数值越大线越陡
    for label, ratio in (("8x1", 8.0), ("4x1", 4.0), ("3x1", 3.0), ("2x1", 2.0),
                         ("1x1", 1.0), ("1x2", 0.5), ("1x3", 1.0 / 3.0), ("1x8", 0.125)):
        slope = step * ratio
        lines.append({"ratio": round(float(ratio), 4), "label": label,
                      "slope": _r(slope, 6),
                      "price_now": _r(a_price + slope * bars_since)})
    lines.sort(key=lambda x: -abs(_num(x["slope"], 0.0)))     # 由陡到缓

    # 定位：按"斜率由小到大"排序看价格落在哪两条线之间
    by_slope = sorted(lines, key=lambda x: _num(x["slope"], 0.0))
    cur_label = by_slope[0]["label"] if by_slope else "1x1"
    for i in range(len(by_slope) - 1):
        lo_v = _num(by_slope[i]["price_now"], 0.0)
        hi_v = _num(by_slope[i + 1]["price_now"], 0.0)
        if lo_v <= px <= hi_v:
            cur_label = by_slope[i]["label"]
            break
    else:
        cur_label = by_slope[-1]["label"] if px > _num(by_slope[-1]["price_now"], 0.0) \
            else by_slope[0]["label"]

    v1x1 = next((_num(l["price_now"], a_price) for l in lines if l["label"] == "1x1"), a_price)
    position = "上方" if px >= v1x1 else "下方"
    dev = (px - v1x1) / max(atr_a, 1e-9)
    score = float(np.clip(dev * 0.25, -0.8, 0.8))
    if not is_low:
        score = -score            # 从高点起画的扇形，方向语义相反
    detail = ("扇形原点 %s（%s %.2f，距今 %d 根 K 线）｜基准斜率 %.3f/日｜"
              "现价 %.2f 位于 %s 与相邻角度线之间、处于 1×1 线%s %.2f 个 ATR（%s）" % (
                  anchor["date"], "低点" if is_low else "高点", a_price, bars_since,
                  atr_a, px, cur_label, position, abs(dev),
                  "趋势健康" if (is_low and position == "上方") or
                  ((not is_low) and position == "下方") else "扇形结构受损"))
    return {
        "origin": {"idx": a_idx, "date": anchor["date"], "price": _r(a_price),
                   "type": "low" if is_low else "high"},
        "lines": lines, "current_line": cur_label, "position": position,
        "score": _r(score, 3), "detail": detail,
        "_atr": _r(atr_a, 4), "_bars_since": int(bars_since),
        "_v1x1": _r(v1x1),
    }


# --------------------------------------------------------------------------- #
# 7. 百分比回撤
# --------------------------------------------------------------------------- #
@_guard(default=dict, label="percentage_retracement")
def percentage_retracement(df) -> dict:
    """百分比回撤：1/3、1/2、2/3 与 1/8 分数体系（12.5%~87.5%）。

    判定逻辑
    --------
    1. 与斐波那契回撤同样取最近 250 日高低点，区别是本理论用**等分/三分**体系：
       1/8 分数（12.5/25/37.5/50/62.5/75/87.5%）+ 三分之一体系（33.3/66.7%）；
    2. 价位 ``P(r) = H - (H-L)*r``；上升波段这些位是支撑，下降波段是压力；
    3. ``position_pct`` = 现价在区间中的位置（0=区间底，100=区间顶），
       ``nearest_level_role`` 依据"现价上方=压力 / 下方=支撑"给出。

    Returns
    -------
    dict：``retracements / nearest_level / position_pct / nearest_level_role /
    score / detail``
    """
    fallback = {"retracements": {}, "nearest_level": None, "position_pct": 50.0,
                "nearest_level_role": "支撑", "score": 0.0,
                "detail": "数据不足，无法计算百分比回撤位"}
    f = _ensure_frame(df)
    if f.empty or len(f) < 40:
        return fallback
    win = f.tail(250)
    hi = _num(win["high"].max())
    lo = _num(win["low"].min())
    if math.isnan(hi) or math.isnan(lo) or hi <= lo:
        return fallback
    i_hi = int(win["high"].astype(float).idxmax())
    i_lo = int(win["low"].astype(float).idxmin())
    up_leg = i_lo < i_hi
    px = _num(f["close"].iloc[-1], (hi + lo) / 2.0)
    rng = hi - lo

    retr: dict[str, float] = {}
    for r in PCT_RATIOS:
        retr["%s(%.1f%%)" % (PCT_LABEL.get(r, "%.3f" % r), r * 100)] = _r(hi - rng * r)
    pos_pct = _pct_pos(px, lo, hi) * 100.0

    cands = sorted((hi - rng * r for r in PCT_RATIOS))
    nearest = min(cands, key=lambda v: abs(v - px)) if cands else px
    role = "支撑" if nearest <= px else "压力"
    dist = abs(nearest - px) / rng
    # 打分：靠近支撑看多，靠近压力看空；越深回撤越偏多（超跌反弹）
    score = 0.0
    if role == "支撑":
        score += 0.35 * (1.0 - min(1.0, dist * 8))
    else:
        score -= 0.35 * (1.0 - min(1.0, dist * 8))
    if pos_pct <= 25:
        score += 0.2
    elif pos_pct >= 80:
        score -= 0.2
    if not up_leg:
        score = -score * 0.8       # 下降波段：位阶语义相反
    score = float(np.clip(score, -0.85, 0.85))

    detail = ("%s波段区间 %.2f~%.2f｜现价 %.2f 处于区间 %.0f%% 位置｜最近回撤/反弹位 %.2f"
              "（%s，距离 %.2f%%）｜关键三分位：33.3%%=%.2f、50%%=%.2f、66.7%%=%.2f" % (
                  "上升" if up_leg else "下降", lo, hi, px, pos_pct, nearest, role,
                  dist * 100, hi - rng / 3.0, hi - rng / 2.0, hi - rng * 2.0 / 3.0))
    return {
        "retracements": retr, "nearest_level": _r(nearest),
        "position_pct": _r(pos_pct, 2), "nearest_level_role": role,
        "score": _r(score, 3), "detail": detail,
        "_high": _r(hi), "_low": _r(lo), "_direction": 1 if up_leg else -1,
    }


# --------------------------------------------------------------------------- #
# 8. 混沌理论
# --------------------------------------------------------------------------- #
def _hurst_rs(x: np.ndarray) -> tuple[float, list[dict]]:
    """R/S 分析法（重标极差）估计 Hurst 指数。

    公式（经典 R/S，Mandelbrot 重标极差）
    -------------------------------------
    对一段长 n 的收益序列 :math:`r_1..r_n`：

    1. 均值 :math:`\\bar r = \\frac{1}{n}\\sum r_i`；
    2. 累积离差（去均值后的偏和）:math:`Z_k = \\sum_{i=1}^{k}(r_i-\\bar r)`；
    3. 极差 :math:`R(n) = \\max_k Z_k - \\min_k Z_k`；
    4. 标准差 :math:`S(n) = \\sqrt{\\frac{1}{n}\\sum (r_i-\\bar r)^2}`；
    5. 对多个尺度 n 计算 :math:`(R/S)_n`，再对 :math:`\\log(R/S) = H\\log n + c`
       做最小二乘回归，斜率即为 Hurst 指数 H。

    期望值：H≈0.5 随机漫步（R/S ∝ n^0.5）；H>0.5 持续性（趋势市）；
    H<0.5 反持续性（均值回复）。H=1 表示完全线性趋势，H=0 表示完全反转。

    Returns
    -------
    ``(H, 明细列表)``；样本不足或方差为 0 时返回 ``(nan, [])``。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_all = len(x)
    if n_all < 32:
        return float("nan"), []
    detail: list[dict] = []
    logs_n: list[float] = []
    logs_rs: list[float] = []
    for n in (8, 16, 32, 64, 128):
        if n > n_all // 2:
            continue
        k = n_all // n
        rs_vals: list[float] = []
        for j in range(k):
            seg = x[j * n:(j + 1) * n]
            m = seg.mean()
            dev = np.cumsum(seg - m)
            r = float(dev.max() - dev.min())
            s = float(seg.std(ddof=0))
            if s > 1e-15 and r > 0:
                rs_vals.append(r / s)
        if not rs_vals:
            continue
        rs_mean = float(np.mean(rs_vals))
        if rs_mean <= 0:
            continue
        logs_n.append(math.log(n))
        logs_rs.append(math.log(rs_mean))
        detail.append({"n": int(n), "segments": int(len(rs_vals)),
                       "rs": _r(rs_mean, 4)})
    if len(logs_n) < 3:
        return float("nan"), detail
    A = np.vstack([np.asarray(logs_n), np.ones(len(logs_n))]).T
    try:
        slope, _intercept = np.linalg.lstsq(A, np.asarray(logs_rs), rcond=None)[0]
    except Exception:  # noqa: BLE001
        return float("nan"), detail
    return float(np.clip(slope, 0.0, 1.0)), detail


def _lyapunov_rosenstein(x: np.ndarray, emb: int = 5, theiler: int = 10) -> float:
    """Rosenstein 法（简化）估计最大 Lyapunov 指数。

    做法：把收益序列做 ``emb`` 维延迟嵌入，对每个点找其"时间上不相邻"（Theiler 窗
    ``theiler``）的最近邻，追踪 k=1..K 步后两点距离的对数平均增长率，
    取线性区间的斜率。**正指数 = 对初值敏感（混沌）**。

    样本不足时返回 NaN（由调用方用 ATR 变异性兜底）。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4 * emb + 20:
        return float("nan")
    m = n - emb + 1
    traj = np.lib.stride_tricks.sliding_window_view(x, emb)[:m]
    K = max(2, min(10, m // 4))
    dist = np.linalg.norm(traj[:, None, :] - traj[None, :, :], axis=2)
    idx = np.arange(m)
    mask = np.abs(idx[:, None] - idx[None, :]) <= theiler
    dist = np.where(mask, np.inf, dist)
    nn = np.argmin(dist, axis=1)
    valid = np.isfinite(dist[idx, nn])
    if valid.sum() < 5:
        return float("nan")
    i0 = idx[valid]
    j0 = nn[valid]
    ks = np.arange(1, K + 1)
    # 注意：轨迹行数为 m（不是原序列长度 n），索引上界必须是 m-1
    ok = (i0[:, None] + ks[None, :] < m) & (j0[:, None] + ks[None, :] < m)
    ii = np.clip(i0[:, None] + ks[None, :], 0, m - 1)
    jj = np.clip(j0[:, None] + ks[None, :], 0, m - 1)
    d = np.linalg.norm(traj[ii] - traj[jj], axis=2)
    d = np.where(ok & (d > 1e-12), d, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        logs = np.log(d)
    # 全为 NaN 的列（该步长下所有邻近点都已越界）会让 np.nanmean 发 RuntimeWarning，
    # 这里用"有效计数"列替代，既避免告警也顺便过滤掉无信息的步长。
    valid_cnt = np.sum(np.isfinite(logs), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_log = np.where(valid_cnt > 0,
                            np.nansum(np.where(np.isfinite(logs), logs, 0.0), axis=0)
                            / np.maximum(valid_cnt, 1), np.nan)
    good = np.isfinite(mean_log)
    if good.sum() < 3:
        return float("nan")
    kk = ks[good].astype(float)
    yy = mean_log[good]
    A = np.vstack([kk, np.ones(len(kk))]).T
    try:
        slope = float(np.linalg.lstsq(A, yy, rcond=None)[0][0])
    except Exception:  # noqa: BLE001
        return float("nan")
    return slope


@_guard(default=dict, label="chaos_theory")
def chaos_theory(df) -> dict:
    """混沌理论：Hurst 指数（R/S 分析）、分形维数、初值敏感性与市场状态。

    判定逻辑
    --------
    1. 用对数收益做 R/S 分析估计 Hurst 指数 H（公式见 :func:`_hurst_rs`）；
       **样本 <120 时直接返回 H=0.5 的保守值**（R/S 在短样本下偏差极大）；
    2. 分形维数 ``D = 2 - H``（D∈[1,2]），D 越大越"锯齿"、越接近随机；
    3. 初值敏感性 ``sensitivity``：由 ATR 的变异系数（std/mean）度量——波动结构越不
       稳定，说明市场对微小扰动越敏感（混沌特征越强）；若 Rosenstein 法的
       Lyapunov 估计可用则一并输出 ``lyapunov_est``，否则用其退化代理；
    4. 市场状态：H > 0.55 趋势市；H < 0.45 均值回复；中间为随机漫步。

    Returns
    -------
    dict：``fractal_dimension / hurst / lyapunov_est / market_state /
    sensitivity / detail``
    """
    fb = {"fractal_dimension": 1.5, "hurst": 0.5, "lyapunov_est": 0.0,
          "market_state": "随机漫步", "sensitivity": 0.5, "_sample": 0,
          "_rs_scales": [], "_lyapunov_source": "样本不足",
          "detail": "数据不足，混沌指标返回保守中性值（H=0.5、D=1.5）"}
    f = _ensure_frame(df)
    if f.empty:
        return dict(fb)
    close = f["close"].astype(float)
    px = _num(close.iloc[-1], 0.0)
    # 用最近 500 个交易日（过长会混入不同市场状态，过短则 R/S 不稳）
    ret = close.pct_change().dropna().tail(500)
    n = len(ret)
    if n < 120:
        fb["_sample"] = int(n)
        fb["detail"] = ("样本仅 %d 个收益观测（<120），R/S 估计不可靠，"
                        "按保守原则返回 H=0.5、D=1.5" % n)
        return dict(fb)

    arr = ret.to_numpy(dtype=float)
    H, rs_detail = _hurst_rs(arr)
    if math.isnan(H):
        H = 0.5
    D = float(np.clip(2.0 - H, 1.0, 2.0))

    if H > 0.55:
        state = "趋势市"
    elif H < 0.45:
        state = "均值回复"
    else:
        state = "随机漫步"

    # 初值敏感性：ATR 的变异系数（越大越敏感）
    atr = _atr(f, 14).tail(120)
    atr_mean = _num(atr.mean(), float("nan"))
    atr_std = _num(atr.std(), float("nan"))
    if math.isnan(atr_mean) or atr_mean <= 0 or math.isnan(atr_std):
        sensitivity = 0.5
    else:
        sensitivity = clamp(float(atr_std / atr_mean), 0.0, 1.0)

    lyap = _lyapunov_rosenstein(arr)
    if math.isnan(lyap):
        # 退化代理：日波动率水平映射到"每个交易日的信息损失速率"
        vol_d = _num(ret.tail(60).std(), float("nan"))
        lyap = 0.0 if math.isnan(vol_d) else float(vol_d)
        lyap_src = "ATR/波动率代理"
    else:
        lyap_src = "Rosenstein 最近邻发散"
    lyap = float(np.clip(lyap, -1.0, 2.0))

    detail = ("Hurst H=%.3f（R/S 分析，样本 %d 个收益，尺度 %s）→ %s；"
              "分形维数 D=2-H=%.3f；初值敏感性 %.2f（ATR 变异系数，ATR均值 %.3f）；"
              "Lyapunov 估计 %.4f（%s）%s" % (
                  H, n, "/".join(str(d["n"]) for d in rs_detail) or "—", state, D,
                  sensitivity, atr_mean if not math.isnan(atr_mean) else float("nan"),
                  lyap, lyap_src,
                  "；H 接近 0.5 说明该股近期走势接近随机漫步，方向性信号应打折"
                  if abs(H - 0.5) < 0.05 else ""))
    return {
        "fractal_dimension": _r(D, 4),
        "hurst": _r(H, 4),
        "lyapunov_est": _r(lyap, 5),
        "market_state": state,
        "sensitivity": _r(sensitivity, 4),
        "detail": detail,
        "_rs_scales": rs_detail,
        "_sample": int(n),
        "_lyapunov_source": lyap_src,
    }


# --------------------------------------------------------------------------- #
# 9. 多周期共振
# --------------------------------------------------------------------------- #
def _auto_weekly_monthly(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """日线自动聚合出周线/月线（pandas 3.0 安全别名）。"""
    wk = _resample_ohlcv(daily, "W-FRI")
    mo = _resample_ohlcv(daily, "ME")
    return wk, mo


@_guard(default=dict, label="multi_timeframe_resonance")
def multi_timeframe_resonance(daily, weekly=None, monthly=None) -> dict:
    """多周期共振：日/周/月三周期分别用道氏理论判趋势，看方向是否一致。

    判定逻辑
    --------
    1. 三个周期都调用 :func:`dow_theory` 得到趋势与分数（周/月线缺省时用日线聚合：
       ``resample("W-FRI")`` / ``resample("ME")``；注意 pandas 3.0 已移除
       ``"W"``/``"M"`` 旧别名）；
    2. 统一方向：上升=+1、下降=-1、横盘=0；
    3. 三者同向且非 0 → ``resonance=True``，强度最高（0.6~1.0，含方向一致性加权）；
       周月同向、日线反向 → 不共振但给出"回调/反弹机会"的方向提示；
    4. 全部不一致或存在横盘 → 强度打折。

    Returns
    -------
    dict：``daily / weekly / monthly / resonance / resonance_direction /
    strength / detail``
    """
    fb = {"daily": {"trend": "横盘", "score": 0.0},
          "weekly": {"trend": "横盘", "score": 0.0},
          "monthly": {"trend": "横盘", "score": 0.0},
          "resonance": False, "resonance_direction": 0, "strength": 0.0,
          "detail": "数据不足，无法判断多周期共振"}
    d = _ensure_frame(daily)
    if d.empty:
        return fb

    w = weekly if isinstance(weekly, pd.DataFrame) and len(weekly) >= MIN_BARS_WEEKLY else None
    m = monthly if isinstance(monthly, pd.DataFrame) and len(monthly) >= MIN_BARS_MONTHLY else None
    if w is None or m is None:
        auto_w, auto_m = _auto_weekly_monthly(d)
        if w is None:
            w = auto_w if len(auto_w) >= MIN_BARS_WEEKLY else None
        if m is None:
            m = auto_m if len(auto_m) >= MIN_BARS_MONTHLY else None

    dw = _trend_of(d, MIN_BARS_DAILY)
    ww = _trend_of(w, MIN_BARS_WEEKLY) if w is not None else {"trend": "数据不足", "score": 0.0}
    mw = _trend_of(m, MIN_BARS_MONTHLY) if m is not None else {"trend": "数据不足", "score": 0.0}

    def _dir(t: str) -> int:
        return {"上升": 1, "下降": -1}.get(t, 0)

    dd, wd, md = _dir(dw["trend"]), _dir(ww["trend"]), _dir(mw["trend"])
    dirs = [dd, wd, md]
    n_valid = sum(1 for x in dirs if x != 0)

    resonance = bool(n_valid == 3 and abs(dd + wd + md) == 3)
    direction = 0
    strength = 0.0
    if resonance:
        direction = dd
        # 强度 = 基础 0.6 + 各周期分数一致性加权（最多 +0.4）
        mags = [abs(_num(dw["score"], 0.0)), abs(_num(ww["score"], 0.0)),
                abs(_num(mw["score"], 0.0))]
        strength = clamp(0.6 + 0.4 * (sum(mags) / 3.0 / 0.9), 0.0, 1.0)
        detail = ("日/周/月三周期均为「%s」，形成**多周期共振**，趋势级别较大、"
                  "方向可信度高（各周期分数 %.2f / %.2f / %.2f）" % (
                      dw["trend"], _num(dw["score"], 0.0), _num(ww["score"], 0.0),
                      _num(mw["score"], 0.0)))
    elif wd != 0 and md != 0 and wd == md and dd == -wd:
        direction = wd
        strength = clamp(0.4 + 0.2 * (abs(_num(ww["score"], 0.0)) +
                                      abs(_num(mw["score"], 0.0))) / 2.0, 0.2, 0.6)
        detail = ("周线/月线为「%s」而日线为「%s」：大周期趋势未变，日线反向多为"
                  "%s机会，属于顺大势逆小势的介入窗口" % (
                      ww["trend"], dw["trend"],
                      "回调" if wd > 0 else "反弹"))
    elif n_valid == 3 and dd == wd == md == 0:
        direction = 0
        strength = 0.15
        detail = "三个周期均为横盘，无趋势可言，方向性信号不可用（宜观望或区间操作）"
    else:
        # 部分一致：以多数方向为准，但强度大幅压低
        votes = {1: dirs.count(1), -1: dirs.count(-1), 0: dirs.count(0)}
        direction = 1 if votes[1] > votes[-1] else (-1 if votes[-1] > votes[1] else 0)
        strength = clamp(0.2 + 0.1 * max(votes[1], votes[-1]), 0.0, 0.45)
        detail = ("周期方向不一致（日 %s / 周 %s / 月 %s），无法形成共振，"
                  "强度显著打折，建议等待方向统一" % (dw["trend"], ww["trend"], mw["trend"]))

    return {
        "daily": {"trend": dw["trend"], "score": _num(dw["score"], 0.0)},
        "weekly": {"trend": ww["trend"], "score": _num(ww["score"], 0.0)},
        "monthly": {"trend": mw["trend"], "score": _num(mw["score"], 0.0)},
        "resonance": resonance,
        "resonance_direction": int(direction),
        "strength": _r(strength, 3),
        "detail": detail,
        "_dirs": {"daily": dd, "weekly": wd, "monthly": md},
        "_bars": {"daily": len(d),
                  "weekly": (len(w) if w is not None else 0),
                  "monthly": (len(m) if m is not None else 0)},
    }


# --------------------------------------------------------------------------- #
# 10. 因子输出
# --------------------------------------------------------------------------- #
# 各模型的作用周期（题目规定）
_H_DOW = ("d5", "d10", "d20", "m1")
_H_WAVE = ("d20", "m1", "q1")
_H_GANN = ("d10", "d20", "m1", "q1")
_H_ALL = None                       # 全周期

# 可靠性上限：理论模型主观性强；江恩/混沌/波浪再打折
_REL_CAP_DEFAULT = 0.65
_REL_CAP_WEAK = 0.55


def _mk_factor(fid: str, name: str, direction, strength, reliability,
               value, detail: str, theory: str, horizon) -> Factor:
    """统一构造理论类因子（direction/strength/reliability 全部 clamp）。"""
    d = int(direction) if direction else 0
    d = 1 if d > 0 else (-1 if d < 0 else 0)
    return Factor(
        id=fid, name=name, category=Category.THEORY, direction=d,
        strength=float(clamp(abs(_num(strength, 0.0)), 0.0, 1.0)),
        reliability=float(clamp(_num(reliability, 0.2), 0.0, 1.0)),
        value=value, detail=str(detail or ""), theory=theory, horizon=horizon,
    )


def _dow_factor(res: dict) -> Factor:
    trend = res.get("trend", "横盘")
    direction = {"上升": DIR_BULL, "下降": DIR_BEAR}.get(trend, DIR_NEUTRAL)
    conf = _num(res.get("confidence"), 0.3)
    score = abs(_num(res.get("score"), 0.0))
    strength = clamp(0.35 + 0.45 * score, 0.0, 0.9) if direction else clamp(0.15, 0, 1)
    rel = clamp(0.25 + 0.45 * conf, 0.15, _REL_CAP_DEFAULT)
    if not res.get("volume_confirm") and direction:
        rel = clamp(rel * 0.9, 0.1, _REL_CAP_DEFAULT)
    return _mk_factor(
        "theory_dow_trend", "道氏趋势结构", direction, strength, rel,
        "%s/%s" % (trend, res.get("phase", "")),
        res.get("detail", ""), THEORY_NAME["dow"], _H_DOW)


def _elliott_factor(res: dict) -> Factor:
    wave = str(res.get("wave", "未知"))
    if wave == "unknown":
        return _mk_factor(
            "theory_elliott", "艾略特波浪结构", DIR_NEUTRAL, 0.0, 0.15,
            "unknown（结构不成立）",
            res.get("detail", "波浪结构不成立，不做浪级判断"),
            THEORY_NAME["elliott"], _H_WAVE)
    direction = int(res.get("direction", 0) or 0)
    conf = _num(res.get("confidence"), 0.2)
    fib = _num(res.get("fib_conformance"), 0.0)
    strength = clamp(0.25 + 0.5 * fib, 0.0, 0.85)
    rel = clamp(0.15 + 0.5 * conf, 0.1, _REL_CAP_WEAK)   # 波浪 ≤0.55
    targets = res.get("targets") or []
    tgt_txt = "；".join("%s %.2f" % (t.get("label"), t.get("price"))
                        for t in targets[:2] if t.get("price") is not None)
    return _mk_factor(
        "theory_elliott", "艾略特波浪结构", direction, strength, rel,
        "浪%s/吻合度%.0f%%" % (wave, fib * 100),
        "%s%s" % (res.get("detail", ""), ("｜目标位：" + tgt_txt) if tgt_txt else ""),
        THEORY_NAME["elliott"], _H_WAVE)


def _fib_factor(res: dict) -> Factor:
    score = _num(res.get("score"), 0.0)
    direction = 1 if score > 0.08 else (-1 if score < -0.08 else 0)
    zone = res.get("current_zone", "中位区")
    strength = clamp(abs(score) * 0.85 + (0.15 if direction else 0.0), 0.0, 0.8)
    rel = clamp(0.30 + (0.2 if res.get("time_windows") else 0.0), 0.2, _REL_CAP_DEFAULT)
    return _mk_factor(
        "theory_fib_level", "斐波那契回撤位", direction, strength, rel,
        "%s/回撤%.1f%%" % (zone, _num(res.get("_retracement_pct"), 0.0)),
        res.get("detail", ""), THEORY_NAME["fib"], _H_WAVE)


def _gann_factor(res: dict) -> Factor:
    score = _num(res.get("score"), 0.0)
    direction = 1 if score > 0.08 else (-1 if score < -0.08 else 0)
    conf = _num(res.get("confidence"), 0.15)
    strength = clamp(abs(score) * 0.7, 0.0, 0.6)
    rel = clamp(0.10 + 0.5 * conf, 0.05, _REL_CAP_WEAK)    # 江恩 ≤0.55
    return _mk_factor(
        "theory_gann", "江恩角度线与时间周期", direction, strength, rel,
        res.get("seasonal_date"),
        "【经验性方法，置信度低】%s" % res.get("detail", ""),
        THEORY_NAME["gann"], _H_GANN)


def _vpt_factor(res: dict) -> Factor:
    score = _num(res.get("score"), 0.0)
    direction = 1 if score > 0.08 else (-1 if score < -0.08 else 0)
    conf = _num(res.get("confidence"), 0.3)
    strength = clamp(abs(score) * 0.9, 0.0, 0.85)
    rel = clamp(0.25 + 0.45 * conf, 0.15, _REL_CAP_DEFAULT)
    names = "/".join(p.get("name", "") for p in (res.get("patterns") or [])[:2])
    return _mk_factor(
        "theory_volume_price_time", "量价时空配合", direction, strength, rel,
        "%s+%s" % (res.get("volume_stage", ""), res.get("price_stage", "")),
        "%s（%s）" % (res.get("detail", ""), names), THEORY_NAME["vpt"], _H_DOW)


def _fan_factor(res: dict) -> Factor:
    score = _num(res.get("score"), 0.0)
    direction = 1 if score > 0.08 else (-1 if score < -0.08 else 0)
    strength = clamp(abs(score) * 0.8, 0.0, 0.7)
    rel = clamp(0.25, 0.1, _REL_CAP_WEAK)
    return _mk_factor(
        "theory_fan", "扇形原理角度线", direction, strength, rel,
        "%s/%s" % (res.get("current_line", ""), res.get("position", "")),
        res.get("detail", ""), THEORY_NAME["fan"], _H_DOW)


def _pct_factor(res: dict) -> Factor:
    score = _num(res.get("score"), 0.0)
    direction = 1 if score > 0.08 else (-1 if score < -0.08 else 0)
    strength = clamp(abs(score) * 0.8, 0.0, 0.75)
    rel = clamp(0.30, 0.15, _REL_CAP_DEFAULT)
    return _mk_factor(
        "theory_pct_retracement", "百分比回撤位", direction, strength, rel,
        "位置%.0f%%/最近位%.2f" % (_num(res.get("position_pct"), 50.0),
                                  _num(res.get("nearest_level"), 0.0)),
        res.get("detail", ""), THEORY_NAME["pct"], _H_WAVE)


def _chaos_factor(res: dict) -> Factor:
    H = _num(res.get("hurst"), 0.5)
    state = res.get("market_state", "随机漫步")
    if state == "趋势市":
        direction, strength = 0, 0.0     # 趋势市本身不含方向，只说明"趋势可持续"
    elif state == "均值回复":
        direction, strength = 0, 0.0
    else:
        direction, strength = 0, 0.0
    # 方向留给其它模型；混沌因子以"方向 0 + strength 表示趋势性强度"呈现，
    # 但为了让引擎能利用它，这里用 H 的偏离程度给出弱方向（H>0.55 顺势，H<0.45 逆势）。
    if state == "趋势市":
        # 顺势：用近期动量方向
        direction = 0
        strength = clamp((H - 0.5) * 2.0, 0.0, 0.5)
    elif state == "均值回复":
        direction = 0
        strength = clamp((0.5 - H) * 2.0, 0.0, 0.5)
    rel = clamp(0.15 + 0.35 * min(1.0, _num(res.get("_sample"), 0) / 500.0),
                0.1, _REL_CAP_WEAK)      # 混沌 ≤0.55
    return _mk_factor(
        "theory_chaos", "混沌特征（Hurst/分形）", direction, strength, rel,
        "H=%.3f/D=%.3f" % (H, _num(res.get("fractal_dimension"), 1.5)),
        "%s（Hurst 与分形维数用于判断趋势可持续性，不直接给方向）" % res.get("detail", ""),
        THEORY_NAME["chaos"], _H_GANN)


def _resonance_factor(res: dict) -> Factor:
    direction = int(res.get("resonance_direction", 0) or 0)
    strength = _num(res.get("strength"), 0.0)
    if res.get("resonance"):
        rel = _REL_CAP_DEFAULT                # 多周期共振证明较扎实，给到上限
        strength = clamp(max(strength, 0.6), 0.0, 0.95)
    elif direction != 0:
        rel = 0.45
    else:
        rel = 0.30
    return _mk_factor(
        "theory_resonance", "多周期共振", direction, strength, rel,
        "日%s/周%s/月%s" % (res.get("daily", {}).get("trend", "—"),
                           res.get("weekly", {}).get("trend", "—"),
                           res.get("monthly", {}).get("trend", "—")),
        res.get("detail", ""), THEORY_NAME["resonance"], _H_ALL)


@_guard(default=list, label="theory_factors")
def theory_factors(df, weekly=None, monthly=None, horizon_keys=None) -> list:
    """把各理论模型的输出统一转成 :class:`Factor` 列表（``Category.THEORY``）。

    因子 id：``theory_dow_trend`` / ``theory_elliott`` / ``theory_fib_level`` /
    ``theory_gann`` / ``theory_volume_price_time`` / ``theory_fan`` /
    ``theory_pct_retracement`` / ``theory_chaos`` / ``theory_resonance``。

    Parameters
    ----------
    df : DataFrame
        日线 OHLCV（列名大小写/中文均可）。
    weekly, monthly : DataFrame | None
        周线/月线；不传则内部聚合。
    horizon_keys : iterable | None
        保留参数（周期由各模型按题目约定固定，见模块常量 ``_H_*``）。

    Returns
    -------
    list[Factor]，任何模型失败都不影响其它模型（逐个 try 包裹）。
    """
    factors: list[Factor] = []
    f = _ensure_frame(df)
    if f.empty:
        log.debug("theory_factors: 输入数据不足，返回空列表")
        return factors

    jobs = (
        ("道氏", lambda: _dow_factor(dow_theory(f))),
        ("波浪", lambda: _elliott_factor(elliott_wave(f))),
        ("斐波那契", lambda: _fib_factor(fibonacci_analysis(f))),
        ("江恩", lambda: _gann_factor(gann_analysis(f))),
        ("量价时空", lambda: _vpt_factor(volume_price_time(f))),
        ("扇形", lambda: _fan_factor(fan_principle(f))),
        ("百分比回撤", lambda: _pct_factor(percentage_retracement(f))),
        ("混沌", lambda: _chaos_factor(chaos_theory(f))),
        ("多周期共振", lambda: _resonance_factor(
            multi_timeframe_resonance(f, weekly, monthly))),
    )
    for label, job in jobs:
        try:
            fac = job()
            if isinstance(fac, Factor):
                factors.append(fac)
        except Exception as exc:  # noqa: BLE001
            log.debug("theory_factors[%s] 失败: %s: %s", label, type(exc).__name__, exc)
    return factors


@_guard(default=dict, label="theory_summary")
def theory_summary(df, weekly=None, monthly=None) -> dict:
    """汇总各理论的结论，供报告直接展示。

    Returns
    -------
    dict：``dow / elliott / fib / gann / vpt / fan / pct / chaos / resonance /
    consensus``，其中 ``consensus`` 给出方向一致性统计（多空计数 + 加权分）。
    """
    f = _ensure_frame(df)
    if f.empty:
        return {"dow": {}, "elliott": {}, "fib": {}, "gann": {}, "vpt": {},
                "fan": {}, "pct": {}, "chaos": {}, "resonance": {},
                "consensus": {"direction": 0, "score": 0.0, "bull_count": 0,
                              "bear_count": 0, "detail": "数据不足，无法给出理论共识"}}

    dow = dow_theory(f)
    ell = elliott_wave(f)
    fib = fibonacci_analysis(f)
    gan = gann_analysis(f)
    vpt = volume_price_time(f)
    fan = fan_principle(f)
    pct = percentage_retracement(f)
    cha = chaos_theory(f)
    res = multi_timeframe_resonance(f, weekly, monthly)

    summary = {"dow": dow, "elliott": ell, "fib": fib, "gann": gan, "vpt": vpt,
               "fan": fan, "pct": pct, "chaos": cha, "resonance": res}

    # --- 共识：方向投票 + 加权得分 --------------------------------------- #
    sources = [
        ("道氏理论", dow.get("trend", "横盘"),
         {"上升": 1, "下降": -1}.get(dow.get("trend", "横盘"), 0),
         abs(_num(dow.get("score"), 0.0)) * _num(dow.get("confidence"), 0.3), 1.0),
        ("艾略特波浪", str(ell.get("wave", "unknown")),
         int(ell.get("direction", 0) or 0),
         abs(_num(ell.get("confidence"), 0.0)) *
         (0.0 if ell.get("wave") == "unknown" else 0.8), 0.8),
        ("斐波那契回撤", fib.get("current_zone", ""),
         1 if _num(fib.get("score"), 0.0) > 0.08 else (
             -1 if _num(fib.get("score"), 0.0) < -0.08 else 0),
         abs(_num(fib.get("score"), 0.0)), 0.7),
        ("江恩理论", str(gan.get("seasonal_date") or ""),
         1 if _num(gan.get("score"), 0.0) > 0.08 else (
             -1 if _num(gan.get("score"), 0.0) < -0.08 else 0),
         abs(_num(gan.get("score"), 0.0)), 0.4),
        ("量价时空", vpt.get("volume_stage", ""),
         1 if _num(vpt.get("score"), 0.0) > 0.08 else (
             -1 if _num(vpt.get("score"), 0.0) < -0.08 else 0),
         abs(_num(vpt.get("score"), 0.0)), 1.0),
        ("扇形原理", "%s/%s" % (fan.get("current_line", ""), fan.get("position", "")),
         1 if _num(fan.get("score"), 0.0) > 0.08 else (
             -1 if _num(fan.get("score"), 0.0) < -0.08 else 0),
         abs(_num(fan.get("score"), 0.0)), 0.6),
        ("百分比回撤", str(pct.get("nearest_level_role", "")),
         1 if _num(pct.get("score"), 0.0) > 0.08 else (
             -1 if _num(pct.get("score"), 0.0) < -0.08 else 0),
         abs(_num(pct.get("score"), 0.0)), 0.7),
        ("混沌理论", cha.get("market_state", ""), 0, 0.0, 0.3),
        ("多周期共振", "共振" if res.get("resonance") else "未共振",
         int(res.get("resonance_direction", 0) or 0),
         _num(res.get("strength"), 0.0), 1.2),
    ]
    bull = sum(1 for s in sources if s[2] > 0)
    bear = sum(1 for s in sources if s[2] < 0)
    wsum = sum(s[3] * s[2] * s[4] for s in sources)
    wabs = sum(s[3] * s[4] for s in sources)
    cons_score = float(np.clip(wsum / wabs if wabs > 1e-9 else 0.0, -1.0, 1.0))
    if cons_score > 0.15:
        cons_dir = 1
    elif cons_score < -0.15:
        cons_dir = -1
    else:
        cons_dir = 0
    agree = max(bull, bear) / float(max(1, bull + bear)) if (bull + bear) else 0.0
    detail = ("理论模型共识：看涨 %d / 看跌 %d / 中性 %d 项，加权分 %+.2f（一致性 %.0f%%）→ %s。"
              "注意：理论模型主观性强，仅作辅助参考。" % (
                  bull, bear, len(sources) - bull - bear, cons_score, agree * 100,
                  "偏多" if cons_dir > 0 else ("偏空" if cons_dir < 0 else "中性")))
    summary["consensus"] = {"direction": int(cons_dir), "score": _r(cons_score, 3),
                            "bull_count": int(bull), "bear_count": int(bear),
                            "detail": detail,
                            "votes": [{"name": s[0], "state": str(s[1]),
                                       "direction": int(s[2]),
                                       "weight": _r(s[3] * s[4], 3)} for s in sources]}
    return summary


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """离线自测：实质性断言 + 关键字段打印。"""
    from .selftest_data import make_ohlcv

    df = make_ohlcv(900, seed=3)
    up = make_ohlcv(900, seed=5, regime="up")
    down = make_ohlcv(900, seed=5, regime="down")
    side = make_ohlcv(900, seed=5, regime="sideways")

    print("=" * 78)
    print("core.theory 自测：make_ohlcv(900, seed=3) 共 %d 根 K 线" % len(df))
    print("=" * 78)

    # ---- 1) 道氏理论：regime 断言 -------------------------------------- #
    d_up, d_dn, d_sd = dow_theory(up), dow_theory(down), dow_theory(side)
    print("[道氏] up   : trend=%s phase=%s score=%+.3f conf=%.2f vol=%s" % (
        d_up["trend"], d_up["phase"], d_up["score"], d_up["confidence"],
        d_up["volume_confirm"]))
    print("[道氏] down : trend=%s phase=%s score=%+.3f conf=%.2f vol=%s" % (
        d_dn["trend"], d_dn["phase"], d_dn["score"], d_dn["confidence"],
        d_dn["volume_confirm"]))
    print("[道氏] side : trend=%s phase=%s score=%+.3f conf=%.2f" % (
        d_sd["trend"], d_sd["phase"], d_sd["score"], d_sd["confidence"]))
    assert d_up["trend"] == "上升", "regime=up 应判为上升趋势，实际 %s" % d_up["trend"]
    assert d_dn["trend"] == "下降", "regime=down 应判为下降趋势，实际 %s" % d_dn["trend"]
    assert len(dow_theory(df)["swing_points"]) >= 2, "摆动点太少"

    # ---- 2) 艾略特波浪 -------------------------------------------------- #
    ell = elliott_wave(df)
    print("[波浪] wave=%s dir=%s conf=%.2f fib=%.2f 摆动点=%d 目标=%s" % (
        ell["wave"], ell["direction"], ell["confidence"], ell["fib_conformance"],
        len(ell["swing_points"]),
        "; ".join("%s=%.2f" % (t["label"], t["price"]) for t in ell["targets"][:2]) or "无"))
    assert ell["wave"] in {"1", "2", "3", "4", "5", "A", "B", "C", "未知"}, ell["wave"]
    assert 0.0 <= ell["confidence"] <= 1.0
    assert len(ell["swing_points"]) >= 4, "波浪用摆动点至少 4 个"
    assert 0.0 <= ell["fib_conformance"] <= 1.0
    assert ell["direction"] in (-1, 0, 1)

    # ---- 3) 斐波那契 ---------------------------------------------------- #
    fib = fibonacci_analysis(df)
    r = fib["retracement"]
    print("[斐波] 区间 %.2f~%.2f 方向=%s 区间=%s" % (fib["low"], fib["high"],
                                                  fib["direction"], fib["current_zone"]))
    print("       回撤位 %s" % ", ".join("%s=%.2f" % (k, v) for k, v in r.items()))
    print("       时间窗 %s" % "; ".join("%s(%+d)" % (w["date"], w["bars_ahead"])
                                          for w in fib["time_windows"][:4]))
    assert r["0.191"] > r["0.618"], "上升波段时 0.191 位应高于 0.618 位"
    assert r["0.236"] > r["0.382"] > r["0.500"] > r["0.786"], "回撤位必须单调"
    assert fib["nearest_support"] <= fib["nearest_resistance"], "支撑应低于压力"
    for w in fib["time_windows"]:
        assert 0 < w["bars_ahead"] <= 60
        assert w["significant_bool"] is True

    # ---- 4) 江恩 -------------------------------------------------------- #
    gan = gann_analysis(df)
    print("[江恩] 角度线 %d 条，四方形 %d 个位，时间周期 %d 个，conf=%.2f" % (
        len(gan["gann_angles"]), len(gan["square_of_9"]), len(gan["time_cycles"]),
        gan["confidence"]))
    assert gan["confidence"] <= 0.55, "江恩属经验性方法，置信度必须压低"
    assert len(gan["gann_angles"]) >= 5 and len(gan["time_cycles"]) >= 5
    for a in gan["gann_angles"]:
        assert a["support_price"] is not None and a["resistance_price"] is not None
    assert all(c["cycle"] in (30, 45, 60, 90, 120, 144, 180, 270, 360)
               for c in gan["time_cycles"])

    # ---- 5) 量价时空 ---------------------------------------------------- #
    vpt = volume_price_time(df)
    print("[量价] 量=%s 价=%s 时间=%s 共振=%s score=%+.3f 组合=%s" % (
        vpt["volume_stage"], vpt["price_stage"], vpt["time_stage"], vpt["match"],
        vpt["score"], vpt["patterns"][0]["name"]))
    assert vpt["volume_stage"] in ("地量", "温和", "放量", "巨量", "天量")
    assert vpt["price_stage"] in ("底部", "上涨中", "顶部", "下跌中")
    assert vpt["time_stage"] in ("时间窗临近", "未到")
    assert len(vpt["patterns"]) >= 1

    # ---- 6) 扇形原理 ---------------------------------------------------- #
    fan = fan_principle(df)
    print("[扇形] 原点 %s(%s %.2f) 现价位于 %s 的%s，1×1=%.2f score=%+.3f" % (
        fan["origin"].get("date"), fan["origin"].get("type"), fan["origin"].get("price"),
        fan["current_line"], fan["position"], fan["_v1x1"], fan["score"]))
    assert fan["position"] in ("上方", "下方")
    assert len(fan["lines"]) >= 5
    assert fan["origin"].get("type") in ("low", "high")

    # ---- 7) 百分比回撤 -------------------------------------------------- #
    pct = percentage_retracement(df)
    print("[百分比] 位置 %.1f%% 最近位 %.2f(%s) score=%+.3f 档位=%d" % (
        pct["position_pct"], pct["nearest_level"], pct["nearest_level_role"],
        pct["score"], len(pct["retracements"])))
    assert 0.0 <= pct["position_pct"] <= 100.0
    assert pct["nearest_level_role"] in ("支撑", "压力")
    assert len(pct["retracements"]) == len(PCT_RATIOS)

    # ---- 8) 混沌理论 ---------------------------------------------------- #
    cha = chaos_theory(df)
    cha_short = chaos_theory(df.tail(80))
    print("[混沌] H=%.4f D=%.4f Lyapunov=%.5f 敏感度=%.3f 状态=%s (样本 %d)" % (
        cha["hurst"], cha["fractal_dimension"], cha["lyapunov_est"],
        cha["sensitivity"], cha["market_state"], cha["_sample"]))
    print("       短样本(80根) H=%.4f（应返回保守值 0.5）" % cha_short["hurst"])
    assert 0.0 <= cha["hurst"] <= 1.0
    assert 1.0 <= cha["fractal_dimension"] <= 2.0
    assert 0.0 <= cha["sensitivity"] <= 1.0
    assert abs(cha["fractal_dimension"] - (2.0 - cha["hurst"])) < 1e-6, "D 必须 = 2 - H"
    assert cha["market_state"] in ("趋势市", "随机漫步", "均值回复")
    assert cha_short["hurst"] == 0.5, "样本<120 必须返回保守 H=0.5"

    # ---- 9) 多周期共振 -------------------------------------------------- #
    res = multi_timeframe_resonance(df)
    print("[共振] 日=%s 周=%s 月=%s 共振=%s 方向=%+d 强度=%.2f" % (
        res["daily"]["trend"], res["weekly"]["trend"], res["monthly"]["trend"],
        res["resonance"], res["resonance_direction"], res["strength"]))
    print("       周期K线数 %s" % res["_bars"])
    assert res["resonance_direction"] in (-1, 0, 1)
    assert 0.0 <= res["strength"] <= 1.0
    for key in ("daily", "weekly", "monthly"):
        assert "trend" in res[key] and "score" in res[key], key
    assert res["_bars"]["weekly"] > 20 and res["_bars"]["monthly"] > 12

    # ---- 10) 因子输出 --------------------------------------------------- #
    factors = theory_factors(df)
    print("-" * 78)
    print("生成因子 %d 个：" % len(factors))
    for fac in factors:
        assert isinstance(fac, Factor), "必须是 Factor 实例"
        assert 0.0 <= fac.strength <= 1.0 and 0.0 <= fac.reliability <= 1.0
        assert fac.category == Category.THEORY
        assert fac.reliability <= 0.65 + 1e-9, "理论因子 reliability 上限 0.65"
        assert fac.theory, "理论依据不能为空"
        print("  %-26s %-16s 方向=%+d 强度=%.2f 可信=%.2f 周期=%-24s 理论=%s" % (
            fac.id, fac.name, fac.direction, fac.strength, fac.reliability,
            "/".join(fac.horizon) if fac.horizon else "全部", fac.theory))
    expect_ids = {"theory_dow_trend", "theory_elliott", "theory_fib_level", "theory_gann",
                  "theory_volume_price_time", "theory_fan", "theory_pct_retracement",
                  "theory_chaos", "theory_resonance"}
    got_ids = {f.id for f in factors}
    assert expect_ids.issubset(got_ids), "缺少因子: %s" % (expect_ids - got_ids)
    for weak in ("theory_gann", "theory_chaos", "theory_elliott"):
        rl = next(f.reliability for f in factors if f.id == weak)
        assert rl <= 0.55 + 1e-9, "%s 的 reliability 必须 ≤0.55，实际 %.3f" % (weak, rl)
    hor = {f.id: f.horizon for f in factors}
    assert hor["theory_dow_trend"] == ("d5", "d10", "d20", "m1")
    assert hor["theory_resonance"] is None
    assert hor["theory_gann"] == ("d10", "d20", "m1", "q1")

    # ---- 11) 汇总 ------------------------------------------------------- #
    su = theory_summary(df)
    print("-" * 78)
    print("theory_summary 关键字段：")
    print("  dow      : %s / %s / score=%+.3f" % (su["dow"]["trend"], su["dow"]["phase"],
                                                 su["dow"]["score"]))
    print("  elliott  : wave=%s conf=%.2f" % (su["elliott"]["wave"],
                                              su["elliott"]["confidence"]))
    print("  fib      : %s 支撑=%.2f 压力=%.2f" % (
        su["fib"]["current_zone"], su["fib"]["nearest_support"],
        su["fib"]["nearest_resistance"]))
    print("  gann     : 角度线 %d 条 conf=%.2f" % (len(su["gann"]["gann_angles"]),
                                                  su["gann"]["confidence"]))
    print("  vpt      : %s+%s score=%+.3f" % (su["vpt"]["volume_stage"],
                                             su["vpt"]["price_stage"], su["vpt"]["score"]))
    print("  fan      : %s/%s" % (su["fan"]["current_line"], su["fan"]["position"]))
    print("  pct      : 位置%.1f%% %s" % (su["pct"]["position_pct"],
                                         su["pct"]["nearest_level_role"]))
    print("  chaos    : H=%.4f D=%.4f %s" % (su["chaos"]["hurst"],
                                             su["chaos"]["fractal_dimension"],
                                             su["chaos"]["market_state"]))
    print("  resonance: %s 方向=%+d 强度=%.2f" % (su["resonance"]["resonance"],
                                                 su["resonance"]["resonance_direction"],
                                                 su["resonance"]["strength"]))
    print("  consensus: 方向=%+d 分=%+.3f 看涨%d/看跌%d" % (
        su["consensus"]["direction"], su["consensus"]["score"],
        su["consensus"]["bull_count"], su["consensus"]["bear_count"]))
    print("             %s" % su["consensus"]["detail"])
    assert set(su.keys()) >= {"dow", "elliott", "fib", "gann", "vpt", "fan", "pct",
                              "chaos", "resonance", "consensus"}
    assert su["consensus"]["direction"] in (-1, 0, 1)
    assert -1.0 <= su["consensus"]["score"] <= 1.0

    # ---- 12) 降级容错 --------------------------------------------------- #
    print("-" * 78)
    empty = pd.DataFrame()
    tiny = make_ohlcv(15)
    for bad in (None, empty, tiny, "not a frame", 12345):
        d0 = dow_theory(bad)
        assert isinstance(d0, dict) and d0["trend"] in ("上升", "下降", "横盘")
        assert isinstance(theory_factors(bad), list)
        assert isinstance(theory_summary(bad), dict)
        assert isinstance(elliott_wave(bad), dict)
        assert isinstance(fibonacci_analysis(bad), dict)
        assert isinstance(gann_analysis(bad), dict)
        assert isinstance(volume_price_time(bad), dict)
        assert isinstance(fan_principle(bad), dict)
        assert isinstance(percentage_retracement(bad), dict)
        assert isinstance(chaos_theory(bad), dict)
        assert isinstance(multi_timeframe_resonance(bad), dict)
    # 中文列名 / 大写列名容错
    zh = df.rename(columns={"date": "日期", "open": "开盘", "high": "最高",
                            "low": "最低", "close": "收盘", "volume": "成交量"})
    assert dow_theory(zh)["trend"] in ("上升", "下降", "横盘")
    # 缺成交量
    novol = df.drop(columns=["volume"])
    assert isinstance(theory_factors(novol), list) and len(theory_factors(novol)) == 9
    print("降级容错：空表/短表/非法输入/中文列名/缺量列 全部安全返回 ✓")

    print("-" * 78)
    print("theory self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()

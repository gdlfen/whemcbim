# -*- coding: utf-8 -*-
"""
时间周期与季节性规律模块（Calendar Effects & Seasonality）。

职责
----
把"A 股特有的日历效应"与"经典时间周期工具"实现成确定性算法，输出中文可读的结构化
结论，并压缩成 :class:`core.contracts.Factor`（``Category.CYCLE``）。

覆盖的内容
----------
============================  ==========================================================
函数                           内容
============================  ==========================================================
:func:`calendar_effects`       日历效应综合判定（财报窗口/两会/长假/月末/季末年末）
:func:`month_effect`           月份效应（个股自身历史 + ``config.MONTH_EFFECT_PRIOR`` 先验融合）
:func:`weekday_effect`         星期效应（周一~周五平均收益/胜率）
:func:`holiday_effects`        长假前后（节前 5/10 日、节后 5/10 日）统计
:func:`earnings_window`        财报披露窗口与业绩地雷/超预期机会
:func:`two_sessions`           两会行情窗口（3 月初~3 月中，**近似规则**）
:func:`month_end_effect`       月末最后 3 日 / 次月前 3 日效应
:func:`fib_time_window`        斐波那契时间窗（5/8/13/21/34/55/89/144 交易日）
:func:`gann_time_cycle`        江恩时间周期（30~360 自然日）
:func:`solar_terms`            二十四节气（**近似公式**，误差 ±1 天）
============================  ==========================================================

输出因子（id 稳定，供权重表匹配，``Category.CYCLE``）
----------------------------------------------------
``calendar_effect`` / ``month_effect`` / ``weekday_effect`` / ``holiday_effect`` /
``earnings_window`` / ``two_sessions`` / ``month_end`` / ``fib_time_window`` /
``gann_time_cycle`` / ``solar_term``

设计约束
--------
* 只依赖 numpy / pandas / 标准库；需要摆动点/趋势判断时复用 :mod:`core.theory`
  （本模块被 theory 依赖的情况不存在，无循环导入）。
* 日历效应是**弱先验**：因子 ``strength`` 上限 **0.35**，``reliability`` 按历史样本量给
  （样本充分 → 0.60，样本很少 → 0.30），绝不允许单靠日历效应下重注。
* 所有交易日推算**自己算**（跳过周末 + 已知长假），不依赖外部交易日历。
* **绝不抛异常打断主流程**：所有对外函数都用 :func:`_guard` 包裹，降级返回中性结果。
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
from .theory import (
    _atr,
    _date_str,
    _ensure_frame,
    _future_trading_days,
    _last_date,
    _num,
    _r,
    _recent_swing_for_anchor,
    dow_theory,
)
from .utils import get_logger

log = get_logger("stockai.seasonality")

__all__ = [
    "calendar_effects", "month_effect", "weekday_effect", "holiday_effects",
    "earnings_window", "two_sessions", "month_end_effect", "fib_time_window",
    "gann_time_cycle", "solar_terms", "seasonality_factors", "seasonality_report",
]

# --------------------------------------------------------------------------- #
# 模块级常量
# --------------------------------------------------------------------------- #
CYCLE_THEORY = "日历效应/时间周期"

#: 因子强度上限：日历类信号只是弱先验
MAX_CALENDAR_STRENGTH = 0.35

#: 二十四节气近似公式常数表（C 值，21 世纪取值；来源：通用"寿星公式"近似表）。
#: 公式为 ``日期 = int(Y*0.2422 + C) - int(Y/4)``，Y 为年份后两位。
#: **这是近似算法，误差约 ±1 天**，仅用于"节气临近"这类粗粒度判断。
SOLAR_TERM_C: tuple[tuple[str, float], ...] = (
    ("小寒", 5.4055), ("大寒", 20.12), ("立春", 3.87), ("雨水", 18.73),
    ("惊蛰", 5.63), ("春分", 20.646), ("清明", 4.81), ("谷雨", 20.1),
    ("立夏", 5.52), ("小满", 21.04), ("芒种", 5.678), ("夏至", 21.37),
    ("小暑", 7.108), ("大暑", 22.83), ("立秋", 7.5), ("处暑", 23.13),
    ("白露", 7.646), ("秋分", 23.042), ("寒露", 8.318), ("霜降", 23.438),
    ("立冬", 7.438), ("小雪", 22.36), ("大雪", 7.18), ("冬至", 21.94),
)
SOLAR_TERM_NAMES = tuple(n for n, _ in SOLAR_TERM_C)

#: 江恩时间周期（自然日）
GANN_CYCLES = (30, 45, 60, 90, 120, 144, 180, 270, 360)

#: 斐波那契时间窗（交易日），来自 config
TIME_WINDOW_BARS = tuple(int(b) for b in config.INDICATOR_PARAMS.get(
    "time_window_bars", (144, 89, 55, 34, 21, 13, 8, 5)))

#: A 股定期报告披露截止日（月, 日）与窗口标签
REPORT_DEADLINES = (
    {"period": "年报", "deadline": (4, 30), "window": "年报窗口",
     "start": (1, 1)},                       # 年报预告 1 月起，正式披露 3~4 月
    {"period": "一季报", "deadline": (4, 30), "window": "一季报窗口",
     "start": (3, 20)},
    {"period": "半年报", "deadline": (8, 31), "window": "半年报窗口",
     "start": (7, 1)},
    {"period": "三季报", "deadline": (10, 31), "window": "三季报窗口",
     "start": (10, 1)},
)

#: 财报窗口判定阈值（自然日）
EARNINGS_NEAR_DAYS = 30          # 距截止日 ≤30 天算"进入窗口"
EARNINGS_HIGH_RISK_DAYS = 14     # 距截止日 ≤14 天算"高风险披露期"

#: 判定"长假"的交易日缺口（自然日）
LONG_HOLIDAY_GAP_DAYS = 3

#: 交易日历缓存：数据里出现过的非交易日（用于交易日推算时跳过）
_CALENDAR_CACHE: dict[str, set] = {}


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


# --------------------------------------------------------------------------- #
# 0. 通用工具
# --------------------------------------------------------------------------- #
def _frame(df) -> pd.DataFrame:
    """:func:`core.theory._ensure_frame` 的薄封装（保持本模块自解释）。"""
    return _ensure_frame(df)


def _as_ts(as_of, df: pd.DataFrame | None = None) -> pd.Timestamp:
    """把 ``as_of`` 归一化成 Timestamp；为空时取数据最后一根 K 线的日期。"""
    if as_of is None or (isinstance(as_of, float) and math.isnan(as_of)):
        if df is not None and len(df):
            return _last_date(df)
        return pd.Timestamp(_dt.date.today())
    try:
        ts = pd.Timestamp(as_of)
        if pd.isna(ts):
            raise ValueError("NaT")
        return ts.normalize()
    except Exception:  # noqa: BLE001
        if df is not None and len(df):
            return _last_date(df)
        return pd.Timestamp(_dt.date.today())


def _holiday_set(df: pd.DataFrame | None) -> set:
    """从 K 线日期里推断"非交易日"（工作日的缺口 = 法定休市）。

    以数据首尾之间的工作日为全集，减去实际有 K 线的日期，即得休市日集合。
    这样在**不联网**的情况下也能跳过春节/国庆等长假，使交易日推算更贴近现实。
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "date" not in df.columns:
        return set()
    try:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if len(dates) < 2:
            return set()
        key = "%s|%s" % (dates.iloc[0].date(), dates.iloc[-1].date())
        if key in _CALENDAR_CACHE:
            return _CALENDAR_CACHE[key]
        have = {d.normalize() for d in dates}
        full = {d.normalize() for d in pd.date_range(dates.iloc[0], dates.iloc[-1], freq="B")}
        hol = full - have
        _CALENDAR_CACHE[key] = hol
        return hol
    except Exception:  # noqa: BLE001
        return set()


def _next_trading_days(last_date, n: int, holidays: Iterable | None = None) -> list[pd.Timestamp]:
    """从 ``last_date`` 之后推算 n 个交易日（跳过周末 + 已知长假）。"""
    return _future_trading_days(last_date, n, holidays)


def _trading_days_between(df: pd.DataFrame, d0, d1) -> int:
    """统计 (d0, d1] 之间的交易日根数（用 K 线日期直接数，最可靠）。"""
    if df is None or df.empty or "date" not in df.columns:
        return 0
    try:
        dates = pd.to_datetime(df["date"], errors="coerce")
        m = (dates > pd.Timestamp(d0)) & (dates <= pd.Timestamp(d1))
        return int(m.sum())
    except Exception:  # noqa: BLE001
        return 0


def _monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """按月聚合收益：``DataFrame[year, month, ret_pct, n_bars]``（已剔除不完整月份）。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["year", "month", "ret_pct", "n_bars"])
    try:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date")
        if len(d) < 2:
            return pd.DataFrame(columns=["year", "month", "ret_pct", "n_bars"])
        d["ym"] = d["date"].dt.to_period("M")
        rows = []
        for ym, g in d.groupby("ym", sort=True):
            g = g.sort_values("date")
            if len(g) < 2:
                continue
            first, last = _num(g["close"].iloc[0]), _num(g["close"].iloc[-1])
            if math.isnan(first) or first == 0 or math.isnan(last):
                continue
            rows.append({"year": int(ym.year), "month": int(ym.month),
                         "ret_pct": (last / first - 1.0) * 100.0, "n_bars": int(len(g))})
        out = pd.DataFrame(rows)
        # 剔除样本过少（<10 个交易日）的月份（多为上市/退市/停牌的残缺月）
        if not out.empty:
            out = out[out["n_bars"] >= 10].reset_index(drop=True)
        return out
    except Exception as exc:  # noqa: BLE001
        log.debug("月度收益聚合失败: %s", exc)
        return pd.DataFrame(columns=["year", "month", "ret_pct", "n_bars"])


def _detect_holidays(df: pd.DataFrame) -> list[dict]:
    """从 K 线日期缺口识别长假（缺口 ≥ :data:`LONG_HOLIDAY_GAP_DAYS` 自然日）。

    Returns
    -------
    list[dict]：``{"start","end","gap_days","pre_idx","post_idx","name"}``，
    按时间升序；``pre_idx``/``post_idx`` 为长假前后那根 K 线在 df 中的位置。
    """
    out: list[dict] = []
    if df is None or df.empty or "date" not in df.columns or len(df) < 3:
        return out
    try:
        dates = pd.to_datetime(df["date"], errors="coerce")
        for i in range(1, len(dates)):
            d0, d1 = dates.iloc[i - 1], dates.iloc[i]
            if pd.isna(d0) or pd.isna(d1):
                continue
            gap = (d1 - d0).days
            if gap < LONG_HOLIDAY_GAP_DAYS:
                continue
            # 关键过滤：普通的"周末三天缺口"（周五休市 → 下周一）不算长假。
            # 只有当缺口里存在**既非周末、又没有 K 线**的日期时，才说明真的放假了
            #（例如国庆 10/1~10/7、春节 7 天假都含有多个工作日）。
            # 这比硬编码节假日表更稳健（各年安排不同），也不依赖联网。
            weekdays = [d for d in pd.date_range(
                d0 + pd.Timedelta(days=1), d1 - pd.Timedelta(days=1))
                if d.weekday() < 5]
            if not weekdays:
                continue
            # 用"缺失的工作日数"衡量假期长短：1 天多为元旦，≥3 天为春节/国庆级长假
            if len(weekdays) < 2:
                continue
            # 按缺口所在月份给个粗略的中文名（仅用于展示）
            if d0.month == 12 or d1.month == 1:
                name = "元旦/春节"
            elif d0.month == 1 or d1.month == 2:
                name = "春节"
            elif d1.month == 5:
                name = "五一"
            elif d1.month == 10:
                name = "十一"
            elif d1.month == 4:
                name = "清明"
            elif d1.month == 6:
                name = "端午"
            elif d1.month == 9:
                name = "中秋"
            else:
                name = "长假"
            out.append({"start": d0.strftime("%Y-%m-%d"),
                        "end": d1.strftime("%Y-%m-%d"),
                        "gap_days": int(gap), "pre_idx": int(i - 1),
                        "post_idx": int(i), "name": name,
                        "missed_weekdays": int(len(weekdays))})
    except Exception as exc:  # noqa: BLE001
        log.debug("长假识别失败: %s", exc)
    return out


def _window_return(close: pd.Series, idx: int, n: int, forward: bool) -> float:
    """长假前/后 n 个交易日的累计收益（%）。

    ``forward=True``  → 从 idx 起往后 n 根（节后行情）；
    ``forward=False`` → 到 idx 为止往前 n 根（节前行情）。
    """
    try:
        if forward:
            a, b = idx, min(len(close) - 1, idx + n)
            if b <= a:
                return float("nan")
            p0, p1 = _num(close.iloc[a]), _num(close.iloc[b])
        else:
            a, b = max(0, idx - n), idx
            if b <= a:
                return float("nan")
            p0, p1 = _num(close.iloc[a]), _num(close.iloc[b])
        if math.isnan(p0) or math.isnan(p1) or p0 == 0:
            return float("nan")
        return (p1 / p0 - 1.0) * 100.0
    except Exception:  # noqa: BLE001
        return float("nan")


def _mean(xs: Sequence[float]) -> float:
    vals = [_num(x) for x in xs]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _win_rate(xs: Sequence[float]) -> float:
    vals = [_num(x) for x in xs]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan")
    return sum(1 for v in vals if v > 0) / float(len(vals))


def _two_sessions_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """两会窗口的**近似规则**：3 月第一个周一（若为 3/1 则顺延一周）至 3 月 15 日。

    真实的两会日期由全国人大常委会/政协主席会议决定并公布（通常 3 月 3 日政协开幕、
    3 月 5 日人大开幕），历史上偶有调整（如 2020 年推迟到 5 月）。本函数只做近似，
    并在 detail 里明确标注，不用于精确事件驱动。
    """
    d = pd.Timestamp("%d-03-01" % int(year))
    # 3 月第一个周一
    offset = (7 - d.weekday()) % 7
    first_mon = d + pd.Timedelta(days=int(offset))
    if first_mon.day == 1:                    # 3/1 就是周一 → 顺延一周（大约 3/8 开幕不合史实）
        first_mon = first_mon + pd.Timedelta(days=7)
    return first_mon, pd.Timestamp("%d-03-15" % int(year))


# --------------------------------------------------------------------------- #
# 1. 月份效应
# --------------------------------------------------------------------------- #
@_guard(default=dict, label="month_effect")
def month_effect(df, as_of=None) -> dict:
    """月份效应：用**该股自身历史**统计各月平均收益/胜率，再与经验先验按样本量融合。

    判定逻辑
    --------
    1. 按月聚合月度收益（首收盘 → 末收盘），剔除不足 10 个交易日的残缺月；
    2. ``hist_month_ret_pct`` / ``hist_month_win_rate``：当前月份的历史均值与胜率；
    3. 与 :data:`core.config.MONTH_EFFECT_PRIOR`（A 股口径的经验月均收益）加权融合：

       ``w_prior = clip(n_years / 10.0, 0.25, 0.75)``，``blended = prior*w_prior +
       hist*(1-w_prior)``；即**样本 <3 年时先验权重 ≥0.7**（个股自身历史太短不可信），
       样本 ≥8 年时个股历史开始占主导；
    4. 方向：``|blended| < 0.5%`` → 中性；否则按符号给出 ±1；强度 = 幅度映射后
       **再乘样本量折扣**，上限 :data:`MAX_CALENDAR_STRENGTH`。

    Returns
    -------
    dict：``current_month / hist_month_ret_pct / hist_month_win_rate / n_years /
    prior_ret_pct / blended_ret_pct / direction / strength / detail``
    """
    fb = {"current_month": int(_as_ts(as_of).month), "hist_month_ret_pct": None,
          "hist_month_win_rate": None, "n_years": 0,
          "prior_ret_pct": _num(config.MONTH_EFFECT_PRIOR.get(int(_as_ts(as_of).month), 0.0)),
          "blended_ret_pct": _num(config.MONTH_EFFECT_PRIOR.get(int(_as_ts(as_of).month), 0.0)),
          "direction": DIR_NEUTRAL, "strength": 0.0,
          "detail": "无历史样本，仅给出经验先验（弱先验，可靠度低）", "_sample": 0}
    f = _frame(df)
    eff = _as_ts(as_of, f if not f.empty else None)
    month = int(eff.month)
    prior = _num(config.MONTH_EFFECT_PRIOR.get(month, 0.0), 0.0)
    fb["current_month"] = month
    fb["prior_ret_pct"] = _r(prior, 3)

    if f.empty:
        return fb
    mr = _monthly_returns(f)
    if mr.empty:
        return fb
    sub = mr[mr["month"] == month]
    n_years = int(sub["year"].nunique())
    hist_ret = _mean(sub["ret_pct"].tolist())
    hist_win = _win_rate(sub["ret_pct"].tolist())
    if math.isnan(hist_ret):
        fb["n_years"] = n_years
        return fb
    w_prior = clamp(n_years / 10.0, 0.25, 0.75)
    blended = prior * w_prior + hist_ret * (1.0 - w_prior)
    if blended > 0.5:
        direction = DIR_BULL
    elif blended < -0.5:
        direction = DIR_BEAR
    else:
        direction = DIR_NEUTRAL
    strength = clamp(abs(blended) / 6.0, 0.0, MAX_CALENDAR_STRENGTH) * (
        0.5 + 0.5 * min(1.0, n_years / 6.0))
    detail = ("%d 月历史（该股 %d 年样本）：平均 %+.2f%%、胜率 %.0f%%；"
              "经验先验 %+.2f%%（先验权重 %.0f%%）→ 融合预期 %+.2f%%，%s。"
              "月份效应是弱先验，样本不足时请以其它证据为主。" % (
                  month, n_years, hist_ret, (hist_win * 100 if not math.isnan(hist_win) else float("nan")),
                  prior, w_prior * 100, blended,
                  "偏多" if direction > 0 else ("偏空" if direction < 0 else "中性")))
    return {"current_month": month,
            "hist_month_ret_pct": _r(hist_ret, 3),
            "hist_month_win_rate": _r(hist_win, 4) if not math.isnan(hist_win) else None,
            "n_years": n_years, "prior_ret_pct": _r(prior, 3),
            "blended_ret_pct": _r(blended, 3), "direction": int(direction),
            "strength": _r(strength, 3), "detail": detail, "_sample": int(len(sub))}


# --------------------------------------------------------------------------- #
# 2. 星期效应
# --------------------------------------------------------------------------- #
@_guard(default=dict, label="weekday_effect")
def weekday_effect(df) -> dict:
    """星期效应：周一~周五各自的平均日收益与胜率（用该股自身历史）。

    判定逻辑：对每根 K 线计算日收益（首日无收益），按 ``date.weekday()`` 分组统计。
    样本 <40 个观测的星期给出 ``None`` 并降低可靠度（体现在 detail 中）。
    A 股经验上"周一偏弱、周五偏强"（周末效应），但个股差异大，仅作弱先验。
    """
    fb = {"by_weekday": {i: {"n": 0, "mean_ret_pct": None, "win_rate": None,
                             "name": n}
                         for i, n in enumerate(("周一", "周二", "周三", "周四", "周五"))},
          "best_day": None, "worst_day": None, "detail": "数据不足，无法统计星期效应"}
    f = _frame(df)
    if f.empty or len(f) < 30:
        return fb
    try:
        ret = f["close"].astype(float).pct_change() * 100.0
        wd = pd.to_datetime(f["date"], errors="coerce").dt.weekday
        by: dict[int, dict] = {}
        stats: list[tuple[int, float]] = []
        for i in range(5):
            vals = ret[(wd == i)].dropna().tolist()
            m = _mean(vals)
            w = _win_rate(vals)
            by[i] = {"n": int(len(vals)), "name": ("周一", "周二", "周三", "周四", "周五")[i],
                     "mean_ret_pct": _r(m, 4) if not math.isnan(m) else None,
                     "win_rate": _r(w, 4) if not math.isnan(w) else None}
            if not math.isnan(m) and len(vals) >= 20:
                stats.append((i, m))
        best = max(stats, key=lambda t: t[1])[0] if stats else None
        worst = min(stats, key=lambda t: t[1])[0] if stats else None
        if best is None:
            detail = "各星期样本不足 20 个，暂不给出星期效应结论"
        else:
            names = ("周一", "周二", "周三", "周四", "周五")
            detail = ("星期效应（该股历史）：最强 %s（均值 %+.3f%%）、最弱 %s（均值 %+.3f%%）"
                      "｜各日样本 %s。注意：星期效应统计显著性普遍很弱。" % (
                          names[best], by[best]["mean_ret_pct"], names[worst],
                          by[worst]["mean_ret_pct"],
                          "/".join("%s:%d" % (by[i]["name"], by[i]["n"]) for i in range(5))))
        return {"by_weekday": by,
                "best_day": best, "worst_day": worst,
                "best_day_name": (("周一", "周二", "周三", "周四", "周五")[best]
                                  if best is not None else None),
                "worst_day_name": (("周一", "周二", "周三", "周四", "周五")[worst]
                                   if worst is not None else None),
                "detail": detail}
    except Exception as exc:  # noqa: BLE001
        log.debug("星期效应统计失败: %s", exc)
        return fb


# --------------------------------------------------------------------------- #
# 3. 长假效应
# --------------------------------------------------------------------------- #
@_guard(default=dict, label="holiday_effects")
def holiday_effects(df, as_of=None) -> dict:
    """长假效应：春节/国庆/五一等长假前后 5/10 个交易日的平均收益统计。

    判定逻辑
    --------
    1. 用 K 线日期的缺口识别长假（缺口 ≥ :data:`LONG_HOLIDAY_GAP_DAYS` 自然日，
       即"用历史年份的开窗日期近似"——不依赖外部节假日表）；
    2. 对每个长假统计：节前 5/10 日累计收益、节后 5/10 日累计收益；
    3. 取所有样本的均值，方向由"节后 5 日均值"主导（节后开门红/补跌），
       节前均值作为强度修正；
    4. **没有历史样本时返回保守中性结果**（``direction=0``、``strength=0``）。

    Returns
    -------
    dict：``pre5_ret / post5_ret / pre10_ret / post10_ret / n_holidays /
    direction / strength / detail``
    """
    fb = {"pre5_ret": None, "post5_ret": None, "pre10_ret": None,
          "post10_ret": None, "n_holidays": 0, "direction": DIR_NEUTRAL,
          "strength": 0.0,
          "detail": "历史样本中没有识别到长假缺口（或数据不足），长假效应返回中性"}
    f = _frame(df)
    if f.empty or len(f) < 60:
        return fb
    hols = _detect_holidays(f)
    if not hols:
        return fb
    close = f["close"].astype(float)
    pre5, post5, pre10, post10 = [], [], [], []
    used = []
    for h in hols:
        i_pre, i_post = h["pre_idx"], h["post_idx"]
        # 前后样本不足 10 根的（数据首尾）跳过，避免截断偏差
        if i_pre < 10 or i_post + 10 >= len(f):
            continue
        a = _window_return(close, i_pre, 5, forward=False)
        b = _window_return(close, i_post - 1, 5, forward=True)
        c = _window_return(close, i_pre, 10, forward=False)
        d = _window_return(close, i_post - 1, 10, forward=True)
        if not math.isnan(a):
            pre5.append(a)
        if not math.isnan(b):
            post5.append(b)
        if not math.isnan(c):
            pre10.append(c)
        if not math.isnan(d):
            post10.append(d)
        used.append(h)
    n = len(used)
    if n == 0:
        return fb
    m_pre5, m_post5 = _mean(pre5), _mean(post5)
    m_pre10, m_post10 = _mean(pre10), _mean(post10)
    post = m_post5 if not math.isnan(m_post5) else (m_post10 if not math.isnan(m_post10) else 0.0)
    pre = m_pre5 if not math.isnan(m_pre5) else (m_pre10 if not math.isnan(m_pre10) else 0.0)
    if post > 0.8:
        direction = DIR_BULL
    elif post < -0.8:
        direction = DIR_BEAR
    else:
        direction = DIR_NEUTRAL
    strength = clamp(abs(post) / 6.0, 0.0, MAX_CALENDAR_STRENGTH) * (
        0.4 + 0.6 * min(1.0, n / 6.0))
    # 节前缩量与节后方向常相反，作为强度修正
    if direction and pre * post < 0:
        strength = clamp(strength * 1.1, 0.0, MAX_CALENDAR_STRENGTH)
    detail = ("共识别 %d 个长假（%s）：节前 5 日均值 %+.2f%%、节后 5 日均值 %+.2f%%；"
              "节前 10 日 %+.2f%%、节后 10 日 %+.2f%% → %s。"
              "A 股经验：节前缩量观望、节后常现开门红，但统计显著性弱。" % (
                  n, "、".join(sorted({h["name"] for h in used})[:4]),
                  m_pre5 if not math.isnan(m_pre5) else 0.0,
                  m_post5 if not math.isnan(m_post5) else 0.0,
                  m_pre10 if not math.isnan(m_pre10) else 0.0,
                  m_post10 if not math.isnan(m_post10) else 0.0,
                  "节后偏多" if direction > 0 else ("节后偏空" if direction < 0 else "无明显节后倾向")))
    return {"pre5_ret": _r(m_pre5, 3) if not math.isnan(m_pre5) else None,
            "post5_ret": _r(m_post5, 3) if not math.isnan(m_post5) else None,
            "pre10_ret": _r(m_pre10, 3) if not math.isnan(m_pre10) else None,
            "post10_ret": _r(m_post10, 3) if not math.isnan(m_post10) else None,
            "n_holidays": int(n), "direction": int(direction),
            "strength": _r(strength, 3), "detail": detail,
            "_holidays": used[-8:]}


# --------------------------------------------------------------------------- #
# 4. 月末效应
# --------------------------------------------------------------------------- #
def _is_last_n_trading_days(df: pd.DataFrame, eff: pd.Timestamp, n: int = 3) -> tuple[bool, int]:
    """判定 eff 是否落在当月最后 n 个交易日内；返回 ``(是否在窗口, 之后还有几个交易日)``。

    优先用 K 线日期（最可靠）；当 ``eff`` 超出数据范围（例如预测未来日期）时，
    退化为"当月最后 n 个工作日"的近似判断，保证函数在任何日期都有确定行为。
    """
    if df is not None and not df.empty and "date" in df.columns:
        try:
            dates = pd.to_datetime(df["date"], errors="coerce").dropna()
            same = dates[(dates.dt.year == eff.year) & (dates.dt.month == eff.month)]
            if not same.empty and bool((same <= eff).any()):
                later = int((same > eff).sum())
                return bool((same >= eff).sum() <= n), later
        except Exception:  # noqa: BLE001
            pass
    # 近似：当月最后 n 个工作日
    try:
        last_day = (eff.replace(day=1) + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
        days = []
        d = last_day
        while len(days) < n:
            if d.weekday() < 5:
                days.append(d)
            d = d - pd.Timedelta(days=1)
        start = min(days)
        if eff >= start:
            return True, int(sum(1 for x in days if x > eff))
        return False, int(sum(1 for x in days if x >= eff))
    except Exception:  # noqa: BLE001
        return False, 99


def _is_first_n_trading_days(df: pd.DataFrame, eff: pd.Timestamp, n: int = 3) -> bool:
    """判定 eff 是否落在当月前 n 个交易日内（超出数据范围时按前 n 个工作日近似）。"""
    if df is not None and not df.empty and "date" in df.columns:
        try:
            dates = pd.to_datetime(df["date"], errors="coerce").dropna()
            same = dates[(dates.dt.year == eff.year) & (dates.dt.month == eff.month)]
            if not same.empty and bool((same >= eff).any()):
                return bool((same <= eff).sum() <= n)
        except Exception:  # noqa: BLE001
            pass
    try:
        d = eff.replace(day=1)
        days = []
        while len(days) < n:
            if d.weekday() < 5:
                days.append(d)
            d = d + pd.Timedelta(days=1)
        return bool(eff <= max(days))
    except Exception:  # noqa: BLE001
        return False


@_guard(default=dict, label="month_end_effect")
def month_end_effect(df, as_of=None) -> dict:
    """月末效应：当月最后 3 个交易日与次月前 3 个交易日的收益规律。

    判定逻辑
    --------
    1. 用 K 线日期定位"每月最后 3 个 / 前 3 个交易日"（不依赖日历表，直接数日期）；
    2. 统计历史各月这两段的累计收益均值（月末段常用于判断资金面紧张与机构做净值，
       次月首段反映"月初资金回流"）；
    3. 方向以"次月前 3 日"为主（月初效应通常更稳定），月末段做强度修正。

    Returns
    -------
    dict：``is_month_end / days_to_month_end / hist_last3_ret / hist_first3_ret /
    direction / strength / detail``
    """
    fb = {"is_month_end": False, "days_to_month_end": 99, "hist_last3_ret": None,
          "hist_first3_ret": None, "direction": DIR_NEUTRAL, "strength": 0.0,
          "detail": "数据不足，月末效应返回中性"}
    f = _frame(df)
    eff = _as_ts(as_of, f if not f.empty else None)
    if f.empty or len(f) < 60:
        return fb
    try:
        dates = pd.to_datetime(f["date"], errors="coerce")
        close = f["close"].astype(float)
        d2 = pd.DataFrame({"date": dates, "close": close}).dropna()
        d2["ym"] = d2["date"].dt.to_period("M")
        d2["pos_in_month"] = d2.groupby("ym").cumcount() + 1
        d2["n_in_month"] = d2.groupby("ym")["close"].transform("size")
        last3, first3 = [], []
        for ym, g in d2.groupby("ym", sort=True):
            g = g.sort_values("date")
            if len(g) < 8:                       # 残缺月跳过
                continue
            seg_l = g.tail(3)
            r_l = (_num(seg_l["close"].iloc[-1]) / _num(seg_l["close"].iloc[0]) - 1.0) * 100.0
            if not math.isnan(r_l):
                last3.append(r_l)
            nxt = g.head(3)
            r_f = (_num(nxt["close"].iloc[-1]) / _num(nxt["close"].iloc[0]) - 1.0) * 100.0
            if not math.isnan(r_f):
                first3.append(r_f)
        m_last, m_first = _mean(last3), _mean(first3)
        n = min(len(last3), len(first3))
        if n == 0:
            return fb
        is_me, days_left = _is_last_n_trading_days(f, eff, 3)
        ref = m_first if not math.isnan(m_first) else 0.0
        if ref > 0.4:
            direction = DIR_BULL
        elif ref < -0.4:
            direction = DIR_BEAR
        else:
            direction = DIR_NEUTRAL
        strength = clamp(abs(ref) / 4.0, 0.0, MAX_CALENDAR_STRENGTH) * (
            0.4 + 0.6 * min(1.0, n / 12.0))
        detail = ("历史 %d 个月统计：月末最后 3 个交易日平均 %+.2f%%、次月前 3 日平均 %+.2f%%"
                  "｜当前%s（距月末还有 %d 个交易日）→ %s。月末资金面偏紧、次月初常有回流，"
                  "但个股差异大。" % (
                      n, m_last if not math.isnan(m_last) else 0.0,
                      m_first if not math.isnan(m_first) else 0.0,
                      "处于月末最后 3 个交易日内" if is_me else
                      ("处于月初前 3 个交易日内" if _is_first_n_trading_days(f, eff, 3) else "不在月末/月初窗口"),
                      days_left,
                      "月初偏多" if direction > 0 else ("月初偏空" if direction < 0 else "无明显倾向")))
        return {"is_month_end": bool(is_me), "days_to_month_end": int(days_left),
                "hist_last3_ret": _r(m_last, 3) if not math.isnan(m_last) else None,
                "hist_first3_ret": _r(m_first, 3) if not math.isnan(m_first) else None,
                "direction": int(direction), "strength": _r(strength, 3),
                "detail": detail, "_n_months": int(n)}
    except Exception as exc:  # noqa: BLE001
        log.debug("月末效应统计失败: %s", exc)
        return fb


# --------------------------------------------------------------------------- #
# 5. 财报窗口
# --------------------------------------------------------------------------- #
def _financial_trend(frame) -> dict:
    """从财务摘要 DataFrame 判断最近的业绩趋势（用于财报窗口的"机会/风险"）。"""
    out = {"rev_yoy": None, "profit_yoy": None, "trend": "无", "direction": 0,
           "detail": "未提供财务数据，无法判断业绩趋势"}
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return out
    try:
        cols = {str(c).strip(): c for c in frame.columns}
        def _pick(*names):
            for n in names:
                for key, real in cols.items():
                    if n in key:
                        return real
            return None
        c_rev = _pick("营收同比", "营收增长", "营业收入同比")
        c_pft = _pick("净利同比", "净利润同比", "净利增长")
        tail = frame.tail(4)
        rev = _mean([safe_float(v, float("nan")) for v in tail[c_rev].tolist()]) \
            if c_rev is not None else float("nan")
        pft = _mean([safe_float(v, float("nan")) for v in tail[c_pft].tolist()]) \
            if c_pft is not None else float("nan")
        out["rev_yoy"] = None if math.isnan(rev) else _r(rev, 2)
        out["profit_yoy"] = None if math.isnan(pft) else _r(pft, 2)
        if math.isnan(pft):
            return out
        if pft > 20:
            out.update({"trend": "加速增长", "direction": 1,
                        "detail": "近 %d 期净利同比均值 %+.1f%%（营收 %s）→ 业绩超预期机会较大" % (
                            len(tail), pft,
                            ("%+.1f%%" % rev) if not math.isnan(rev) else "—")})
        elif pft < -10:
            out.update({"trend": "下滑", "direction": -1,
                        "detail": "近 %d 期净利同比均值 %+.1f%% → 业绩地雷风险偏高" % (
                            len(tail), pft)})
        else:
            out.update({"trend": "平稳", "direction": 0,
                        "detail": "近 %d 期净利同比均值 %+.1f%% → 业绩弹性有限" % (
                            len(tail), pft)})
    except Exception as exc:  # noqa: BLE001
        log.debug("财务趋势判断失败: %s", exc)
    return out


def _next_report_period(eff: pd.Timestamp) -> tuple[str, pd.Timestamp, str]:
    """返回 ``(下一个披露期标识, 截止日, 对应的 window 标签)``。

    报告期标识形如 ``2026Q3`` / ``2025A``。A 股披露节奏：
    年报（次年 1~4 月，截止 4/30）、一季报（4 月，截止 4/30）、
    半年报（7~8 月，截止 8/31）、三季报（10 月，截止 10/31）。
    """
    y = int(eff.year)
    candidates: list[tuple[pd.Timestamp, str, str]] = []
    for yy in (y - 1, y, y + 1):
        candidates.append((pd.Timestamp("%d-04-30" % yy), "%dA" % (yy - 1), "年报窗口"))
        candidates.append((pd.Timestamp("%d-04-30" % yy), "%dQ1" % yy, "一季报窗口"))
        candidates.append((pd.Timestamp("%d-08-31" % yy), "%dQ2" % yy, "半年报窗口"))
        candidates.append((pd.Timestamp("%d-10-31" % yy), "%dQ3" % yy, "三季报窗口"))
    candidates.sort(key=lambda t: t[0])
    for dl, period, label in candidates:
        if dl >= eff:
            return period, dl, label
    return "%dQ3" % y, pd.Timestamp("%d-10-31" % y), "三季报窗口"


@_guard(default=dict, label="earnings_window")
def earnings_window(symbol, as_of=None, frame=None) -> dict:
    """财报披露窗口判断（A 股法定截止日：年报/一季报 4/30、半年报 8/31、三季报 10/31）。

    判定逻辑
    --------
    1. 找出"当前日期之后最近的一个披露截止日"，得到报告期标识与窗口标签；
    2. ``days_to_deadline`` 为自然日差；``≤30`` 天视为**进入窗口**，``≤14`` 天为
       高风险披露期（历史上业绩暴雷/超预期多集中在截止日前两周）；
    3. 若传入 ``frame``（结构同 :func:`core.selftest_data.make_financials_frame`），
       用最近 4 期"净利同比/营收同比"均值判断业绩趋势，给出方向：

       * 净利同比均值 > +20% → 业绩超预期机会（+1）；
       * < -10% → 业绩地雷风险（-1）；
       * 其余 → 平稳（0）。

    Returns
    -------
    dict：``is_window / window / days_to_deadline / next_report_period / risk /
    direction / detail``
    """
    eff = _as_ts(as_of)
    period, deadline, label = _next_report_period(eff)
    days = int((deadline.normalize() - eff.normalize()).days)
    is_window = bool(0 <= days <= EARNINGS_NEAR_DAYS)
    high_risk = bool(0 <= days <= EARNINGS_HIGH_RISK_DAYS)
    fin = _financial_trend(frame)
    if fin["direction"] > 0:
        risk = "业绩超预期机会"
    elif fin["direction"] < 0:
        risk = "业绩地雷风险"
    else:
        # 无财务数据时按"窗口期本身就是不确定性"处理，方向中性
        risk = "无" if not is_window else "无（未提供财务数据，仅提示披露窗口）"
    direction = int(fin["direction"]) if is_window else DIR_NEUTRAL
    detail = ("最近披露截止日 %s（%s，报告期 %s），距今 %d 个自然日 → %s｜%s" % (
        deadline.strftime("%Y-%m-%d"), label, period, days,
        "已进入披露窗口" + ("（高风险期，历史暴雷/超预期集中在此）" if high_risk else "")
        if is_window else "未进入披露窗口",
        fin.get("detail", "")))
    return {"is_window": is_window, "window": label if is_window else "无",
            "days_to_deadline": int(days), "next_report_period": period,
            "risk": risk, "direction": direction, "detail": detail,
            "_deadline": deadline.strftime("%Y-%m-%d"),
            "_financial_trend": fin.get("trend", "无"),
            "_rev_yoy": fin.get("rev_yoy"), "_profit_yoy": fin.get("profit_yoy")}


# --------------------------------------------------------------------------- #
# 6. 两会窗口
# --------------------------------------------------------------------------- #
def _two_sessions_history(df: pd.DataFrame) -> tuple[list[float], list[float], list[int]]:
    """统计历史各年两会窗口（近似日期）内的收益与前后收益。"""
    inside: list[float] = []
    years: list[int] = []
    if df is None or df.empty or "date" not in df.columns:
        return inside, [], years
    try:
        dates = pd.to_datetime(df["date"], errors="coerce")
        close = df["close"].astype(float)
        first_y = int(dates.dt.year.min())
        last_y = int(dates.dt.year.max())
        for y in range(first_y, last_y + 1):
            start, end = _two_sessions_window(y)
            # 数据必须完整覆盖该窗口
            m = (dates >= start) & (dates <= end)
            if int(m.sum()) < 5:
                continue
            idx = np.where(m.to_numpy())[0]
            p0, p1 = _num(close.iloc[idx[0]]), _num(close.iloc[idx[-1]])
            if math.isnan(p0) or p0 == 0 or math.isnan(p1):
                continue
            inside.append((p1 / p0 - 1.0) * 100.0)
            years.append(int(y))
    except Exception as exc:  # noqa: BLE001
        log.debug("两会历史统计失败: %s", exc)
    return inside, [], years


@_guard(default=dict, label="two_sessions")
def two_sessions(df, as_of=None) -> dict:
    """两会行情窗口判断（全国人大/政协，通常 3 月 3 日~3 月 15 日）。

    .. note::
       两会日期用**近似规则**：3 月第一个周一（若为 3/1 则顺延一周）至 3 月 15 日。
       真实会期由官方公布（历史上如 2020 年推迟至 5 月），本函数不追求精确，
       只在"是否临近/处于两会窗口"这一粗粒度上使用。

    判定逻辑
    --------
    1. 定位当前年份的两会窗口；``in_session`` 表示今天在窗口内，
       ``days_to`` 表示距窗口开始的自然日数（负数表示窗口已结束的天数）；
    2. 用该股历史数据统计历次两会窗口内的收益均值 ``hist_ret_pct``；
    3. 方向：窗口内历史均值 >0.5% → 偏多，<-0.5% → 偏空，否则中性；强度上限 0.35。

    Returns
    -------
    dict：``in_session / days_to / hist_ret_pct / direction / strength / detail``
    """
    f = _frame(df)
    eff = _as_ts(as_of, f if not f.empty else None)
    start, end = _two_sessions_window(eff.year)
    in_session = bool(start <= eff <= end)
    days_to = int((start - eff).days)
    inside, _, years = _two_sessions_history(f) if not f.empty else ([], [], [])
    hist = _mean(inside)
    n = len(inside)
    if math.isnan(hist) or n == 0:
        direction, strength = DIR_NEUTRAL, 0.0
    elif hist > 0.5:
        direction = DIR_BULL
        strength = clamp(abs(hist) / 6.0, 0.0, MAX_CALENDAR_STRENGTH) * (
            0.4 + 0.6 * min(1.0, n / 4.0))
    elif hist < -0.5:
        direction = DIR_BEAR
        strength = clamp(abs(hist) / 6.0, 0.0, MAX_CALENDAR_STRENGTH) * (
            0.4 + 0.6 * min(1.0, n / 4.0))
    else:
        direction, strength = DIR_NEUTRAL, 0.0
    detail = ("两会窗口（**近似规则**：%s ~ %s，3 月第一个周一前后）｜当前%s"
              "（距窗口开始 %+d 天）｜历史 %d 次窗口内平均收益 %s → %s。"
              "两会期间政策预期主导，题材轮动快，对个股方向指示性弱。" % (
                  start.strftime("%m-%d"), end.strftime("%m-%d"),
                  "处于窗口内" if in_session else "不在窗口内", days_to, n,
                  ("%+.2f%%" % hist) if not math.isnan(hist) else "—",
                  "偏多" if direction > 0 else ("偏空" if direction < 0 else "中性")))
    return {"in_session": in_session, "days_to": int(days_to),
            "hist_ret_pct": _r(hist, 3) if not math.isnan(hist) else None,
            "direction": int(direction), "strength": _r(strength, 3),
            "detail": detail,
            "_window": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
            "_years": years}


# --------------------------------------------------------------------------- #
# 7. 斐波那契时间窗
# --------------------------------------------------------------------------- #
@_guard(default=dict, label="fib_time_window")
def fib_time_window(df, as_of=None) -> dict:
    """斐波那契时间窗：从最近的显著性高低点起算 5/8/13/21/34/55/89/144 个交易日。

    判定逻辑
    --------
    1. 锚点 = 最近一个显著性摆动点（ZigZag 阈值 ATR×2.5 且 ≥4%，见
       :func:`core.theory._recent_swing_for_anchor`）；
    2. ``bars_since`` = 锚点距今的交易日根数；
    3. 对每个斐波那契数字 ``b``，若 ``b - bars_since`` 落在未来 60 个交易日内，
       就算作一个变盘窗口，日期用**交易日推算**（跳过周末 + 数据里识别出的长假）；
    4. ``in_window``：距最近窗口 ≤1 个交易日；方向依据锚点类型（低点起算偏多、
       高点起算偏空）与当前是否处在上行腿。

    Returns
    -------
    dict：``anchor_date / anchor_price / anchor_type / bars_since / next_windows /
    in_window / direction / strength / detail``
    """
    fb = {"anchor_date": None, "anchor_price": None, "anchor_type": None,
          "bars_since": 0, "next_windows": [], "in_window": False,
          "direction": DIR_NEUTRAL, "strength": 0.0,
          "detail": "数据不足，无法计算斐波那契时间窗"}
    f = _frame(df)
    if f.empty or len(f) < 40:
        return fb
    eff = _as_ts(as_of, f)
    anchor = _recent_swing_for_anchor(f)
    if anchor is None:
        return fb
    a_idx = int(anchor["idx"])
    bars_since = max(0, (len(f) - 1) - a_idx)
    hol = _holiday_set(f)
    windows: list[dict] = []
    for b in sorted(set(TIME_WINDOW_BARS)):
        ahead = int(b) - bars_since
        if 0 <= ahead <= 60:
            dts = _next_trading_days(eff, max(1, ahead), hol) if ahead > 0 else [eff]
            if not dts:
                continue
            windows.append({"bars_ahead": int(ahead),
                            "date": dts[-1].strftime("%Y-%m-%d"),
                            "label": "第 %d 个交易日（斐波那契）" % int(b)})
    windows.sort(key=lambda w: (w["bars_ahead"], w["date"]))
    in_window = bool(windows and windows[0]["bars_ahead"] <= 1)
    is_low = anchor["type"] == "low"
    if is_low:
        direction = DIR_BULL
    else:
        direction = DIR_BEAR
    # 窗口临近本身不构成方向，方向主要来自锚点性质；强度压得很低
    strength = clamp(0.10 + (0.10 if in_window else 0.0), 0.0, MAX_CALENDAR_STRENGTH)
    detail = ("斐波那契时间窗锚点：%s（%s %.2f），距今 %d 个交易日｜未来 60 个交易日内有 "
              "%d 个窗口%s→ %s。时间窗只提示「可能变盘」，不指示方向。" % (
                  anchor["date"], "低点" if is_low else "高点", anchor["price"],
                  bars_since, len(windows),
                  ("（最近 %s，%s）" % (windows[0]["date"], windows[0]["label"])) if windows else "",
                  "偏多" if direction > 0 else "偏空"))
    return {"anchor_date": anchor["date"], "anchor_price": _r(anchor["price"]),
            "anchor_type": anchor["type"], "bars_since": int(bars_since),
            "next_windows": windows, "in_window": in_window,
            "direction": int(direction), "strength": _r(strength, 3),
            "detail": detail, "_anchor_idx": a_idx}


# --------------------------------------------------------------------------- #
# 8. 江恩时间周期
# --------------------------------------------------------------------------- #
@_guard(default=dict, label="gann_time_cycle")
def gann_time_cycle(df, as_of=None) -> dict:
    """江恩时间周期：从最近显著性高低点起算 30/45/60/90/120/144/180/270/360 **自然日**。

    .. warning::
       江恩时间周期属**经验性方法，置信度低**，本函数只给出"周期到期日"与
       "是否临近"，不据此给方向（``direction`` 恒为 0）。

    判定逻辑
    --------
    1. 锚点同 :func:`fib_time_window`（最近显著性摆动点）；
    2. 对每个周期 ``c``（自然日）：``date = 锚点日 + c 天``，交易日根数用
       ``c * 5/7`` 近似（A 股每周 5 个交易日）；
    3. ``near_cycle``：任一周期到期日在未来 5 个自然日内。

    Returns
    -------
    dict：``anchor_date / cycles / near_cycle / direction / detail``
    """
    fb = {"anchor_date": None, "cycles": [], "near_cycle": False,
          "direction": DIR_NEUTRAL,
          "detail": "数据不足，无法计算江恩时间周期（经验性方法，置信度低）"}
    f = _frame(df)
    if f.empty or len(f) < 40:
        return fb
    eff = _as_ts(as_of, f)
    anchor = _recent_swing_for_anchor(f)
    if anchor is None:
        return fb
    a_ts = pd.Timestamp(anchor["date"]) if anchor.get("date") else eff
    cycles: list[dict] = []
    for c in GANN_CYCLES:
        target = a_ts + pd.Timedelta(days=int(c))
        bars = int(round(c * 5.0 / 7.0))
        days_ahead = int((target.normalize() - eff.normalize()).days)
        kind = "已过" if days_ahead < 0 else ("临近" if days_ahead <= 5 else "未来")
        cycles.append({"cycle": int(c), "date": target.strftime("%Y-%m-%d"),
                       "bars": int(bars), "days_ahead": int(days_ahead),
                       "kind": kind, "label": "江恩 %d 日周期" % int(c)})
    cycles.sort(key=lambda x: x["cycle"])
    near = [c for c in cycles if c["kind"] == "临近"] or \
           [c for c in cycles if c["kind"] == "未来" and c["days_ahead"] <= 5]
    detail = ("江恩时间周期以 %s（%s %.2f）为锚，未来最近到期日 %s（周期 %d 自然日）｜"
              "**注意：江恩时间周期属经验性方法，置信度低，仅供参考，不给方向**" % (
                  anchor["date"], "低点" if anchor["type"] == "low" else "高点",
                  anchor["price"],
                  (near[0]["date"] if near else "—"),
                  (near[0]["cycle"] if near else 0)))
    return {"anchor_date": anchor["date"], "cycles": cycles,
            "near_cycle": bool(near), "direction": DIR_NEUTRAL, "detail": detail,
            "_anchor_price": _r(anchor["price"]),
            "_anchor_type": anchor["type"]}


# --------------------------------------------------------------------------- #
# 9. 二十四节气
# --------------------------------------------------------------------------- #
@_guard(default=list, label="solar_terms")
def solar_terms(as_of=None, n: int = 6) -> list:
    """未来 n 个二十四节气。

    **近似算法**（误差约 ±1 天，仅用于粗粒度判断）：::

        日期 = int(Y * 0.2422 + C) - int(Y / 4)

    其中 ``Y`` 为年份后两位、``C`` 为该节气的常数（见 :data:`SOLAR_TERM_C`）。
    该式对 20/21 世纪有效；对少数年份需要 +1 天修正，本实现不做修正，
    因此在"是否临近节气"这种 ±1 天无关紧要的场景使用。

    Returns
    -------
    list[dict]：``[{"name": "立春", "date": "2026-02-04", "days_ahead": 37}, ...]``
    按日期升序，全部在未来（含当天）。
    """
    eff = _as_ts(as_of)
    y = int(eff.year)
    cands: list[tuple[pd.Timestamp, str]] = []
    for yy in (y - 1, y, y + 1):
        Y = yy % 100
        for name, C in SOLAR_TERM_C:
            try:
                day = int(Y * 0.2422 + C) - int(Y / 4)
                if day < 1 or day > 31:
                    continue
                ts = _solar_term_timestamp(yy, name, day)
                if ts is None:
                    continue
                cands.append((ts, name))
            except Exception:  # noqa: BLE001
                continue
    cands = [c for c in cands if c[0].normalize() >= eff.normalize()]
    cands.sort(key=lambda t: t[0])
    out: list[dict] = []
    seen = set()
    for ts, name in cands:
        key = (name, ts.strftime("%Y-%m-%d"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "date": ts.strftime("%Y-%m-%d"),
                    "days_ahead": int((ts.normalize() - eff.normalize()).days)})
        if len(out) >= max(1, int(n)):
            break
    return out


#: 节气名称顺序（用于把"日"修正到正确的月份）
_SOLAR_TERM_ORDER = {name: i for i, (name, _) in enumerate(SOLAR_TERM_C)}


def _solar_term_timestamp(year: int, name: str, day: int):
    """把"年 + 节气名 + 日"组装成 Timestamp（自动推断月份）。

    节气的月份由它在 24 节气中的序号决定：序号 // 2 即月份（小寒=1 月，大寒=1 月，
    立春=2 月，雨水=2 月，……）。这样公式只需给出"日"，不必单独维护月份表。
    """
    idx = _SOLAR_TERM_ORDER.get(name)
    if idx is None:
        return None
    month = idx // 2 + 1
    try:
        return pd.Timestamp(year=int(year), month=int(month), day=int(day))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# 10. 日历效应综合
# --------------------------------------------------------------------------- #
def _season_effect(name: str, direction: int, strength: float, detail: str) -> dict:
    return {"name": name, "direction": int(direction),
            "strength": float(clamp(strength, 0.0, 1.0)), "detail": detail}


@_guard(default=dict, label="calendar_effects")
def calendar_effects(symbol, df, as_of=None) -> dict:
    """日历效应综合判定：把 A 股主要的季节性/事件性窗口汇总成一个方向分。

    综合的效应
    ----------
    1. **财报窗口**：年报/一季报（1~4 月，截止 4/30）、半年报（7~8 月，截止 8/31）、
       三季报（10 月，截止 10/31）——窗口内业绩不确定性上升；
    2. **两会行情**：2 月底~3 月中（近似 3 月第一个周一 ~ 3/15），政策预期驱动；
    3. **春节前后**：节前缩量（资金离场）、节后开门红（历史胜率偏高）；
    4. **月末资金紧张**：每月最后 3 个交易日（资金面考核/缴税）；
    5. **季末/年末调仓**：3/6/9/12 月的最后 10 个交易日（机构考核、排名博弈）；
    6. **十一长假前后**、**五一小长假**；（清明/端午/中秋同样识别）
    7. 斐波那契时间窗与节气窗口（弱提示）。

    返回值中的 ``effects`` 逐条列出命中的效应，``score`` 为其强度加权和（-1~1），
    全部强度都被压到 ≤ :data:`MAX_CALENDAR_STRENGTH`。

    Returns
    -------
    dict：``date / month / weekday / days_to_month_end / is_earnings_window /
    is_two_sessions / is_pre_holiday / is_post_holiday / effects / score``
    """
    f = _frame(df)
    eff = _as_ts(as_of, f if not f.empty else None)
    weekday = int(eff.weekday())
    effects: list[dict] = []

    # --- 1) 财报窗口 ----------------------------------------------------- #
    earn = earnings_window(symbol, eff, None)
    days_dl = int(earn.get("days_to_deadline", 999))
    if earn.get("is_window"):
        effects.append(_season_effect(
            "%s" % earn.get("window", "财报窗口"),
            0,                                  # 窗口内方向由业绩决定，这里不给方向
            clamp(0.18 + (0.12 if days_dl <= EARNINGS_HIGH_RISK_DAYS else 0.0), 0.0, 0.35),
            "距 %s 披露截止日 %d 天（报告期 %s），业绩不确定性上升" % (
                earn.get("_deadline", ""), days_dl, earn.get("next_report_period", ""))))

    # --- 2) 两会 --------------------------------------------------------- #
    ts = two_sessions(f, eff)
    if ts.get("in_session") or 0 <= int(ts.get("days_to", 99)) <= 10:
        effects.append(_season_effect(
            "两会行情窗口", int(ts.get("direction", 0) or 0),
            clamp(float(ts.get("strength", 0.1) or 0.1), 0.05, 0.35),
            ts.get("detail", "")))

    # --- 3) 长假前后 ----------------------------------------------------- #
    hols = _detect_holidays(f) if not f.empty else []
    is_pre = is_post = False
    hol_dir = 0
    hol_strength = 0.15
    hol_detail = ""
    if hols and not f.empty:
        close = f["close"].astype(float)
        # 用数据最后一根 K 线位置判断"当前"处于节前还是节后（as_of 一般等于最后一根）
        dates = pd.to_datetime(f["date"], errors="coerce")
        try:
            pos = int(np.where(dates <= eff)[0][-1]) if int((dates <= eff).sum()) else len(f) - 1
        except Exception:  # noqa: BLE001
            pos = len(f) - 1
        for h in hols:
            if 0 < h["post_idx"] - pos <= 3 or 0 <= pos - h["post_idx"] < 3:
                is_post = True
                hol_detail = "处于 %s 长假后 3 个交易日内（%s）" % (h["name"], h["end"])
                break
            if 0 < h["pre_idx"] - pos <= 3 or 0 <= pos - h["pre_idx"] < 3:
                is_pre = True
                hol_detail = "处于 %s 长假前 3 个交易日内（%s）" % (h["name"], h["start"])
                break
    if is_pre or is_post:
        hist = holiday_effects(f, eff)
        hol_dir = int(hist.get("direction", 0) or 0) if is_post else 0
        hol_strength = clamp(float(hist.get("strength", 0.12) or 0.12), 0.05, 0.35)
        effects.append(_season_effect(
            "长假前缩量" if is_pre else "长假后开门红",
            hol_dir, hol_strength,
            "%s；历史%s" % (hol_detail, hist.get("detail", ""))))

    # --- 4) 月末 / 月初资金面 -------------------------------------------- #
    me = month_end_effect(f, eff)
    days_left = int(me.get("days_to_month_end", 99))
    if me.get("is_month_end") or days_left <= 1:
        effects.append(_season_effect(
            "月末资金紧张", int(me.get("direction", 0) or 0),
            clamp(float(me.get("strength", 0.12) or 0.12), 0.05, 0.35),
            "已进入当月最后 3 个交易日（距月末 %d 个交易日）：资金面考核/缴税通常偏紧"
            % days_left))

    # --- 5) 季末/年末机构调仓 -------------------------------------------- #
    if eff.month in (3, 6, 9, 12):
        last10 = _is_last_n_trading_days(f, eff, 10)
        if last10[0]:
            tag = "年末" if eff.month == 12 else "季末"
            effects.append(_season_effect(
                "%s机构调仓" % tag, -1 if eff.month == 12 else 0, 0.15,
                "%d 月最后 10 个交易日：机构考核/排名博弈，%s" % (
                    eff.month,
                    "年末常见获利了结与调仓换股（博弈性强）" if eff.month == 12
                    else "季末资金面与调仓行为叠加，波动放大")))

    # --- 6) 月效应（弱） -------------------------------------------------- #
    mo = month_effect(f, eff)
    if int(mo.get("direction", 0) or 0) != 0:
        effects.append(_season_effect(
            "%d 月季节效应" % eff.month, int(mo["direction"]),
            clamp(float(mo.get("strength", 0.1) or 0.1), 0.0, 0.35),
            mo.get("detail", "")))

    # --- 7) 星期效应（弱） ------------------------------------------------ #
    wd = weekday_effect(f) if not f.empty else {}
    best, worst = wd.get("best_day"), wd.get("worst_day")
    if best is not None and worst is not None and best != worst:
        names = ("周一", "周二", "周三", "周四", "周五")
        if weekday == best:
            effects.append(_season_effect(
                "星期效应(%s偏强)" % names[weekday], 1, 0.10,
                "该股历史上 %s 平均收益最高" % names[weekday]))
        elif weekday == worst:
            effects.append(_season_effect(
                "星期效应(%s偏弱)" % names[weekday], -1, 0.10,
                "该股历史上 %s 平均收益最低" % names[weekday]))

    # --- 8) 斐波那契时间窗 / 节气（提示性，不给方向） --------------------- #
    ftw = fib_time_window(f, eff) if not f.empty else {}
    if ftw.get("in_window"):
        effects.append(_season_effect("斐波那契时间窗", 0, 0.15,
                                      ftw.get("detail", "")))
    st = solar_terms(eff, 2)
    if st and st[0]["days_ahead"] <= 2:
        effects.append(_season_effect(
            "节气窗口(%s)" % st[0]["name"], 0, 0.08,
            "距二十四节气「%s」还有 %d 天（近似算法，误差 ±1 天）" % (
                st[0]["name"], st[0]["days_ahead"])))

    # --- 汇总 ------------------------------------------------------------ #
    num = sum(e["direction"] * e["strength"] for e in effects)
    den = sum(e["strength"] for e in effects if e["direction"] != 0)
    score = float(np.clip(num / den if den > 1e-9 else 0.0, -1.0, 1.0))
    return {
        "date": eff.strftime("%Y-%m-%d"),
        "month": int(eff.month),
        "weekday": weekday,
        "days_to_month_end": int(days_left),
        "is_earnings_window": bool(earn.get("is_window")),
        "is_two_sessions": bool(ts.get("in_session")),
        "is_pre_holiday": bool(is_pre),
        "is_post_holiday": bool(is_post),
        "effects": effects,
        "score": _r(score, 3),
        "_n_effects": len(effects),
        "_weekday_name": ("周一", "周二", "周三", "周四", "周五", "周六", "周日")[weekday],
    }


# --------------------------------------------------------------------------- #
# 11. 因子输出
# --------------------------------------------------------------------------- #
# 各因子的作用周期（题目规定）
_H_CAL = ("d20", "m1", "q1")            # 日历/月效应
_H_EARN = ("d5", "d10", "d20", "m1")    # 财报窗口
_H_TIMEW = ("d5", "d10", "d20")         # 斐波那契/江恩时间窗
_H_HOLIDAY = ("d5", "d10")              # 节假日效应

_REL_SAMPLE_HIGH = 0.60                 # 历史样本充分
_REL_SAMPLE_MID = 0.45
_REL_SAMPLE_LOW = 0.30                  # 样本很少/无样本


def _rel_by_sample(n: float, hi: float = _REL_SAMPLE_HIGH) -> float:
    """按样本量给 reliability：多 → hi，少 → 0.30。"""
    n = _num(n, 0.0)
    if n >= 60:
        return hi
    if n >= 24:
        return (hi + _REL_SAMPLE_MID) / 2.0
    if n >= 8:
        return _REL_SAMPLE_MID
    return _REL_SAMPLE_LOW


def _cyc_factor(fid: str, name: str, direction, strength, reliability,
                value, detail: str, horizon) -> Factor:
    """构造日历经期因子：强度统一被压到 :data:`MAX_CALENDAR_STRENGTH` 以内。"""
    d = int(direction) if direction else 0
    d = 1 if d > 0 else (-1 if d < 0 else 0)
    return Factor(
        id=fid, name=name, category=Category.CYCLE, direction=d,
        strength=float(clamp(abs(_num(strength, 0.0)), 0.0, MAX_CALENDAR_STRENGTH)),
        reliability=float(clamp(_num(reliability, _REL_SAMPLE_LOW), 0.0, 1.0)),
        value=value, detail=str(detail or ""), theory=CYCLE_THEORY, horizon=horizon,
    )


def _cal_factor(res: dict) -> Factor:
    score = _num(res.get("score"), 0.0)
    direction = 1 if score > 0.05 else (-1 if score < -0.05 else 0)
    n_eff = int(res.get("_n_effects", 0) or 0)
    strength = clamp(abs(score) * 0.30 + 0.05 * n_eff, 0.0, MAX_CALENDAR_STRENGTH)
    names = "、".join(e["name"] for e in (res.get("effects") or [])[:3])
    return _cyc_factor(
        "calendar_effect", "日历效应综合", direction, strength,
        _rel_by_sample(60.0 if n_eff else 0),   # 日历规则是"制度性"的，样本量按命中项算
        "%s 命中%d项" % (res.get("date", ""), n_eff),
        "命中日历效应：%s（综合分 %+.2f）。日历效应是弱先验，请结合其它证据。" % (
            names or "无", score),
        _H_CAL)


def _month_factor(res: dict) -> Factor:
    n = _num(res.get("n_years"), 0.0)
    return _cyc_factor(
        "month_effect", "月份效应", res.get("direction", 0),
        res.get("strength", 0.0), _rel_by_sample(n * 12.0),
        "%d月/%+.2f%%" % (int(res.get("current_month", 0) or 0),
                         _num(res.get("blended_ret_pct"), 0.0)),
        res.get("detail", ""), _H_CAL)


def _weekday_factor(res: dict) -> Factor:
    by = res.get("by_weekday") or {}
    n = sum(int((by.get(i) or {}).get("n", 0) or 0) for i in range(5))
    # 星期效应极弱，方向取自"当日历史均值"的符号
    cur = by.get(int(_dt.date.today().weekday()), {}) if by else {}
    m = _num(cur.get("mean_ret_pct"), float("nan"))
    direction = 0
    strength = 0.0
    if not math.isnan(m) and abs(m) > 0.05:
        direction = 1 if m > 0 else -1
        strength = clamp(abs(m) / 1.0, 0.0, 0.15)
    names = ("周一", "周二", "周三", "周四", "周五")
    best = res.get("best_day")
    worst = res.get("worst_day")
    return _cyc_factor(
        "weekday_effect", "星期效应", direction, strength, _rel_by_sample(n / 5.0),
        "最强%s/最弱%s" % (names[best] if best is not None else "—",
                          names[worst] if worst is not None else "—"),
        res.get("detail", ""), _H_CAL)


def _holiday_factor(res: dict) -> Factor:
    n = _num(res.get("n_holidays"), 0.0)
    return _cyc_factor(
        "holiday_effect", "长假前后效应", res.get("direction", 0),
        res.get("strength", 0.0), _rel_by_sample(n * 2.0),
        "节后5日%+.2f%%" % _num(res.get("post5_ret"), 0.0) if res.get("post5_ret") is not None
        else "无样本",
        res.get("detail", ""), _H_HOLIDAY)


def _earnings_factor(res: dict) -> Factor:
    days = _num(res.get("days_to_deadline"), 999.0)
    # 只有"已进入披露窗口"时业绩方向才生效（窗口外风险尚未释放，方向置 0）
    direction = int(res.get("direction", 0) or 0) if res.get("is_window") else 0
    strength = 0.0
    if direction:
        strength = 0.30 if days <= EARNINGS_HIGH_RISK_DAYS else 0.20
    elif res.get("is_window"):
        strength = 0.12                      # 仅提示不确定性（方向 0）
    return _cyc_factor(
        "earnings_window", "财报披露窗口", direction, strength,
        _REL_SAMPLE_HIGH if direction else _REL_SAMPLE_LOW,
        "%s/%d天" % (res.get("next_report_period", ""), int(days)),
        res.get("detail", ""), _H_EARN)


def _two_sessions_factor(res: dict) -> Factor:
    n = 4.0 if res.get("hist_ret_pct") is not None else 0.0
    return _cyc_factor(
        "two_sessions", "两会行情窗口", res.get("direction", 0),
        res.get("strength", 0.0), _rel_by_sample(n if n else 6.0),
        "窗口内" if res.get("in_session") else "距%d天" % int(_num(res.get("days_to"), 0)),
        res.get("detail", ""), _H_CAL)


def _month_end_factor(res: dict) -> Factor:
    n = _num(res.get("_n_months"), 0.0)
    return _cyc_factor(
        "month_end", "月末/月初效应", res.get("direction", 0),
        res.get("strength", 0.0), _rel_by_sample(n),
        "月末%s/距%d日" % ("是" if res.get("is_month_end") else "否",
                          int(_num(res.get("days_to_month_end"), 0))),
        res.get("detail", ""), _H_HOLIDAY)


def _fib_window_factor(res: dict) -> Factor:
    windows = res.get("next_windows") or []
    in_win = bool(res.get("in_window"))
    direction = int(res.get("direction", 0) or 0) if (in_win or windows) else 0
    strength = clamp(0.10 if in_win else (0.05 if windows else 0.0), 0.0,
                     MAX_CALENDAR_STRENGTH)
    return _cyc_factor(
        "fib_time_window", "斐波那契时间窗", direction, strength,
        _REL_SAMPLE_MID if windows else _REL_SAMPLE_LOW,
        "%s/距%d窗" % (res.get("anchor_date") or "—",
                      int((windows[0]["bars_ahead"] if windows else -1))),
        res.get("detail", ""), _H_TIMEW)


def _gann_cycle_factor(res: dict) -> Factor:
    cycles = res.get("cycles") or []
    near = bool(res.get("near_cycle"))
    return _cyc_factor(
        "gann_time_cycle", "江恩时间周期", 0,            # 经验性方法，不方向化
        clamp(0.10 if near else 0.05, 0.0, MAX_CALENDAR_STRENGTH),
        _REL_SAMPLE_LOW,
        "%s/临近%s" % (res.get("anchor_date") or "—", "是" if near else "否"),
        "【经验性方法，置信度低】%s" % res.get("detail", ""), _H_TIMEW)


def _solar_term_factor(terms: list) -> Factor:
    if not terms:
        return _cyc_factor("solar_term", "二十四节气", 0, 0.0, _REL_SAMPLE_LOW,
                           "无", "未取得节气信息（近似算法，误差 ±1 天）", _H_CAL)
    nxt = terms[0]
    days = _num(nxt.get("days_ahead"), 99.0)
    strength = clamp(0.12 if days <= 2 else (0.06 if days <= 7 else 0.0),
                     0.0, MAX_CALENDAR_STRENGTH)
    return _cyc_factor(
        "solar_term", "二十四节气", 0, strength, _REL_SAMPLE_LOW,
        "%s/%d天" % (nxt.get("name", ""), int(days)),
        "下一个节气「%s」在 %s（还有 %d 天）。节气日期由近似公式 int(Y*0.2422+C)-int(Y/4) "
        "计算，误差 ±1 天，仅作弱提示。" % (
            nxt.get("name", ""), nxt.get("date", ""), int(days)),
        _H_CAL)


@_guard(default=list, label="seasonality_factors")
def seasonality_factors(symbol, df, as_of=None, frame=None) -> list:
    """把季节性与时间周期结论统一转成 :class:`Factor` 列表（``Category.CYCLE``）。

    因子 id：``calendar_effect`` / ``month_effect`` / ``weekday_effect`` /
    ``holiday_effect`` / ``earnings_window`` / ``two_sessions`` / ``month_end`` /
    ``fib_time_window`` / ``gann_time_cycle`` / ``solar_term``。

    Parameters
    ----------
    symbol : str
        股票代码（仅用于日志/展示）。
    df : DataFrame
        日线 OHLCV。
    as_of : 任意可转 Timestamp 的值
        分析基准日；默认取数据最后一根 K 线。
    frame : DataFrame | None
        财务摘要（结构同 :func:`core.selftest_data.make_financials_frame`），
        用于财报窗口的"地雷/超预期"方向判断。

    Returns
    -------
    list[Factor]；单个子模型失败不影响其它因子。
    """
    factors: list[Factor] = []
    f = _frame(df)
    eff = _as_ts(as_of, f if not f.empty else None)

    jobs = (
        ("日历综合", lambda: _cal_factor(calendar_effects(symbol, f, eff))),
        ("月份效应", lambda: _month_factor(month_effect(f, eff))),
        ("星期效应", lambda: _weekday_factor(weekday_effect(f))),
        ("长假效应", lambda: _holiday_factor(holiday_effects(f, eff))),
        ("财报窗口", lambda: _earnings_factor(earnings_window(symbol, eff, frame))),
        ("两会", lambda: _two_sessions_factor(two_sessions(f, eff))),
        ("月末", lambda: _month_end_factor(month_end_effect(f, eff))),
        ("斐波那契时间窗", lambda: _fib_window_factor(fib_time_window(f, eff))),
        ("江恩周期", lambda: _gann_cycle_factor(gann_time_cycle(f, eff))),
        ("节气", lambda: _solar_term_factor(solar_terms(eff, 3))),
    )
    for label, job in jobs:
        try:
            fac = job()
            if isinstance(fac, Factor):
                factors.append(fac)
        except Exception as exc:  # noqa: BLE001
            log.debug("seasonality_factors[%s] 失败: %s: %s", label, type(exc).__name__, exc)
    return factors


# --------------------------------------------------------------------------- #
# 12. 汇总报告
# --------------------------------------------------------------------------- #
@_guard(default=dict, label="seasonality_report")
def seasonality_report(symbol, df, as_of=None) -> dict:
    """季节性汇总：把各结论整理成便于报告展示的 dict。

    Returns
    -------
    dict：``symbol / as_of / calendar / month / weekday / holiday / earnings /
    two_sessions / month_end / fib_window / gann_cycle / solar_terms /
    consensus``，``consensus`` 给出方向一致性统计。
    """
    f = _frame(df)
    eff = _as_ts(as_of, f if not f.empty else None)
    rep = {
        "symbol": str(symbol or ""),
        "as_of": eff.strftime("%Y-%m-%d"),
        "calendar": calendar_effects(symbol, f, eff),
        "month": month_effect(f, eff),
        "weekday": weekday_effect(f),
        "holiday": holiday_effects(f, eff),
        "earnings": earnings_window(symbol, eff, None),
        "two_sessions": two_sessions(f, eff),
        "month_end": month_end_effect(f, eff),
        "fib_window": fib_time_window(f, eff),
        "gann_cycle": gann_time_cycle(f, eff),
        "solar_terms": solar_terms(eff, 4),
    }
    # --- 共识：只统计有方向的弱信号 -------------------------------------- #
    items = [
        ("月份效应", rep["month"].get("direction", 0), rep["month"].get("strength", 0.0)),
        ("长假效应", rep["holiday"].get("direction", 0), rep["holiday"].get("strength", 0.0)),
        ("两会窗口", rep["two_sessions"].get("direction", 0),
         rep["two_sessions"].get("strength", 0.0)),
        ("月末效应", rep["month_end"].get("direction", 0),
         rep["month_end"].get("strength", 0.0)),
        ("财报窗口", rep["earnings"].get("direction", 0),
         0.25 if rep["earnings"].get("direction") else 0.0),
        ("斐波那契时间窗", rep["fib_window"].get("direction", 0),
         rep["fib_window"].get("strength", 0.0)),
        ("日历综合", 1 if _num(rep["calendar"].get("score"), 0.0) > 0.05 else
         (-1 if _num(rep["calendar"].get("score"), 0.0) < -0.05 else 0),
         abs(_num(rep["calendar"].get("score"), 0.0)) * MAX_CALENDAR_STRENGTH),
    ]
    bull = sum(1 for _, d, _s in items if d > 0)
    bear = sum(1 for _, d, _s in items if d < 0)
    wsum = sum(d * s for _, d, s in items)
    wabs = sum(abs(s) for _, _d, s in items)
    score = float(np.clip(wsum / wabs if wabs > 1e-9 else 0.0, -1.0, 1.0))
    direction = 1 if score > 0.1 else (-1 if score < -0.1 else 0)
    rep["consensus"] = {
        "direction": int(direction), "score": _r(score, 3),
        "bull_count": int(bull), "bear_count": int(bear),
        "detail": "季节性弱信号：偏多 %d 项 / 偏空 %d 项，加权分 %+.2f → %s。"
                  "日历效应与时间周期都是弱先验（强度上限 %.2f），"
                  "不构成独立的买卖依据。" % (
                      bull, bear, score,
                      "偏多" if direction > 0 else ("偏空" if direction < 0 else "中性"),
                      MAX_CALENDAR_STRENGTH),
        "votes": [{"name": n, "direction": int(d), "strength": _r(s, 3)}
                  for n, d, s in items],
    }
    # 便于报告直接取用的"今天命中什么"
    rep["today"] = {
        "date": rep["calendar"].get("date"),
        "weekday": rep["calendar"].get("_weekday_name"),
        "month": rep["calendar"].get("month"),
        "days_to_month_end": rep["calendar"].get("days_to_month_end"),
        "effects": [e["name"] for e in (rep["calendar"].get("effects") or [])],
        "weekday_best": rep["weekday"].get("best_day_name"),
        "weekday_worst": rep["weekday"].get("worst_day_name"),
    }
    return rep


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
#: 自测用的"注入长假"区间（[起, 止] 闭区间，删掉其中的 K 线模拟休市）。
#: :func:`core.selftest_data.make_ohlcv` 只跳过周末，不含法定长假，
#: 因此长假识别与统计必须靠注入缺口来验证。
_SELFTEST_HOLIDAYS = (("2023-10-02", "2023-10-06"),      # 国庆
                      ("2024-02-12", "2024-02-16"))      # 春节


def _inject_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """把 :data:`_SELFTEST_HOLIDAYS` 区间内的 K 线删掉，模拟真实长假缺口。"""
    d = df.copy()
    dates = pd.to_datetime(d["date"], errors="coerce")
    keep = pd.Series(True, index=d.index)
    for a, b in _SELFTEST_HOLIDAYS:
        keep &= ~((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
    return d[keep].reset_index(drop=True)


def _self_test() -> None:
    """离线自测：实质性断言 + 关键字段打印。"""
    from .selftest_data import make_financials_frame, make_ohlcv, make_index

    df = make_ohlcv(900, seed=3)
    as_of = pd.Timestamp(df["date"].iloc[-1])
    fin = make_financials_frame()

    print("=" * 78)
    print("core.seasonality 自测：make_ohlcv(900, seed=3)，as_of=%s（%s）" % (
        as_of.strftime("%Y-%m-%d"), ("周一", "周二", "周三", "周四", "周五")[as_of.weekday()]))
    print("=" * 78)

    # ---- 1) 日历效应 ---------------------------------------------------- #
    cal = calendar_effects("600519", df, as_of)
    print("[日历] %s 月=%d 星期=%s 距月末=%d 财报窗口=%s 两会=%s 节前=%s 节后=%s" % (
        cal["date"], cal["month"], cal["_weekday_name"], cal["days_to_month_end"],
        cal["is_earnings_window"], cal["is_two_sessions"], cal["is_pre_holiday"],
        cal["is_post_holiday"]))
    for e in cal["effects"]:
        print("       · %-18s 方向=%+d 强度=%.2f  %s" % (
            e["name"], e["direction"], e["strength"], e["detail"][:70]))
    print("       综合分 %+.3f" % cal["score"])
    assert set(cal.keys()) >= {"date", "month", "weekday", "days_to_month_end",
                               "is_earnings_window", "is_two_sessions",
                               "is_pre_holiday", "is_post_holiday", "effects", "score"}
    assert -1.0 <= cal["score"] <= 1.0
    for e in cal["effects"]:
        assert 0.0 <= e["strength"] <= MAX_CALENDAR_STRENGTH + 1e-9, "日历效应强度必须 ≤0.35"
        assert e["direction"] in (-1, 0, 1)

    # ---- 2) 月份效应 ---------------------------------------------------- #
    mo = month_effect(df, as_of)
    print("[月效应] %d 月：历史 %s%%/胜率 %s（%d 年）｜先验 %+.2f%% → 融合 %+.2f%% 方向=%+d 强度=%.2f" % (
        mo["current_month"],
        mo["hist_month_ret_pct"] if mo["hist_month_ret_pct"] is not None else "—",
        ("%.0f%%" % (mo["hist_month_win_rate"] * 100)) if mo["hist_month_win_rate"] else "—",
        mo["n_years"], mo["prior_ret_pct"], mo["blended_ret_pct"],
        mo["direction"], mo["strength"]))
    assert mo["direction"] in (-1, 0, 1)
    assert 0.0 <= mo["strength"] <= MAX_CALENDAR_STRENGTH + 1e-9
    assert mo["prior_ret_pct"] == config.MONTH_EFFECT_PRIOR[mo["current_month"]]
    # 样本 <3 年时先验权重必须 ≥0.7
    short = month_effect(df.tail(400), as_of)
    if short["n_years"] and short["n_years"] < 3:
        w = clamp(short["n_years"] / 10.0, 0.25, 0.75)
        assert w >= 0.25

    # ---- 3) 星期效应 ---------------------------------------------------- #
    we = weekday_effect(df)
    names = ("周一", "周二", "周三", "周四", "周五")
    print("[星期] %s" % " | ".join(
        "%s n=%d 均%+.3f%% 胜率%s" % (
            names[i], we["by_weekday"][i]["n"], we["by_weekday"][i]["mean_ret_pct"] or 0.0,
            ("%.0f%%" % (we["by_weekday"][i]["win_rate"] * 100))
            if we["by_weekday"][i]["win_rate"] is not None else "—")
        for i in range(5)))
    print("       最强 %s / 最弱 %s" % (names[we["best_day"]] if we["best_day"] is not None else "—",
                                       names[we["worst_day"]] if we["worst_day"] is not None else "—"))
    assert set(we["by_weekday"].keys()) == {0, 1, 2, 3, 4}
    assert 0 <= (we["best_day"] if we["best_day"] is not None else 0) <= 4

    # ---- 4) 长假效应 ---------------------------------------------------- #
    # make_ohlcv 只跳过周末、不含法定长假，因此这里**注入**国庆/春节缺口
    #（删掉节假日窗口内的 K 线）来验证识别与统计逻辑。
    hol_df = _inject_holidays(df)
    n_gap = len(_detect_holidays(hol_df))
    he = holiday_effects(hol_df, pd.Timestamp(hol_df["date"].iloc[-1]))
    print("[长假] 注入 %d 个长假缺口 → 识别 %d 个｜节前5日 %s%% / 节后5日 %s%%｜"
          "节前10日 %s%% / 节后10日 %s%% → 方向=%+d 强度=%.2f" % (
              len(_SELFTEST_HOLIDAYS), n_gap, he["pre5_ret"], he["post5_ret"],
              he["pre10_ret"], he["post10_ret"], he["direction"], he["strength"]))
    assert n_gap >= 2, "注入 2 个长假缺口后应至少识别出 2 个长假，实际 %d" % n_gap
    assert he["n_holidays"] >= 1, "应至少统计到 1 个可用的长假样本"
    assert he["direction"] in (-1, 0, 1)
    assert 0.0 <= he["strength"] <= MAX_CALENDAR_STRENGTH + 1e-9
    # 无样本时保守中性
    tiny = make_ohlcv(30, seed=1)
    he0 = holiday_effects(tiny, pd.Timestamp(tiny["date"].iloc[-1]))
    assert he0["direction"] == 0 and he0["strength"] == 0.0, "无样本必须返回中性"
    # 纯周末数据不应被误判为长假
    assert len(_detect_holidays(df)) == 0, "只有周末缺口的序列不该识别出长假"

    # ---- 5) 财报窗口 ---------------------------------------------------- #
    ew = earnings_window("600519", as_of, fin)
    ew_no = earnings_window("600519", as_of, None)
    print("[财报] 窗口=%s 报告期=%s 距截止=%d 天 风险=%s 方向=%+d" % (
        ew["window"], ew["next_report_period"], ew["days_to_deadline"], ew["risk"],
        ew["direction"]))
    print("       传 frame 后的业绩趋势：%s（营收同比 %s，净利同比 %s）" % (
        ew["_financial_trend"], ew["_rev_yoy"], ew["_profit_yoy"]))
    assert ew["window"] in ("年报窗口", "一季报窗口", "半年报窗口", "三季报窗口", "无")
    assert ew["days_to_deadline"] >= 0
    assert ew["risk"] in ("业绩地雷风险", "业绩超预期机会", "无") or ew["risk"].startswith("无")
    assert ew["direction"] in (-1, 0, 1)
    assert ew_no["direction"] == 0, "未提供财务数据时不应给出方向"
    # 截止日推算校验：3/1 距 4/30 有 60 天 → 未进窗口；4/20 距 10 天 → 高风险窗口
    e2 = earnings_window("600519", pd.Timestamp("2026-03-01"), None)
    assert e2["_deadline"] == "2026-04-30", e2["_deadline"]
    assert e2["is_window"] is False, "3/1 距 4/30 有 60 天，不该算窗口"
    e3 = earnings_window("600519", pd.Timestamp("2026-04-20"), None)
    assert e3["is_window"] and e3["days_to_deadline"] == 10
    assert e3["window"] == "年报窗口", e3["window"]
    e4 = earnings_window("600519", pd.Timestamp("2026-09-01"), None)
    assert e4["_deadline"] == "2026-10-31", e4["_deadline"]
    e5 = earnings_window("600519", pd.Timestamp("2026-08-20"), None)
    assert e5["window"] == "半年报窗口" and e5["days_to_deadline"] == 11, e5
    print("       截止日推算校验：2026-03-01→%s(未进窗口)、2026-04-20→%s(%d天)、"
          "2026-08-20→%s(%d天)、2026-09-01→%s ✓" % (
              e2["_deadline"], e3["_deadline"], e3["days_to_deadline"],
              e5["window"], e5["days_to_deadline"], e4["_deadline"]))
    # 业绩趋势 → 方向：负增长应给出"业绩地雷风险"且只在窗口内生效
    bear = pd.DataFrame([{"报告期": "2025Q%d" % (i + 1), "营业收入": 1e9, "营收同比": -18.0,
                          "净利润": 1e8, "净利同比": -35.0, "ROE": 3.0, "毛利率": 12.0,
                          "净利率": 5.0, "资产负债率": 68.0, "经营现金流": 1e7,
                          "每股收益": 0.1} for i in range(4)])
    ew_bear = earnings_window("600519", pd.Timestamp("2026-04-25"), bear)
    assert ew_bear["risk"] == "业绩地雷风险" and ew_bear["direction"] == -1, ew_bear
    ew_bull = earnings_window("600519", pd.Timestamp("2026-04-25"), fin)
    assert ew_bull["risk"] == "业绩超预期机会" and ew_bull["direction"] == 1, ew_bull
    fac_bear = [f for f in seasonality_factors("600519", df, pd.Timestamp("2026-04-25"), bear)
                if f.id == "earnings_window"][0]
    assert fac_bear.direction == -1 and fac_bear.strength == 0.30, fac_bear
    print("       业绩趋势→方向校验：负增长帧→%s(方向%+d)、正增长帧→%s(方向%+d)、"
          "财报因子 方向%+d/强度%.2f ✓" % (
              ew_bear["risk"], ew_bear["direction"], ew_bull["risk"], ew_bull["direction"],
              fac_bear.direction, fac_bear.strength))

    # ---- 6) 两会 -------------------------------------------------------- #
    ts = two_sessions(df, as_of)
    ts_in = two_sessions(df, pd.Timestamp("2026-03-05"))
    print("[两会] 当前 in_session=%s 距开始=%+d 天 历史均值=%s%% 方向=%+d（窗口 %s）" % (
        ts["in_session"], ts["days_to"], ts["hist_ret_pct"], ts["direction"], ts["_window"]))
    assert ts["in_session"] is False or ts["in_session"] is True
    assert ts_in["in_session"] is True, "2026-03-05 应处于两会窗口内"
    assert ts_in["hist_ret_pct"] is None or isinstance(ts_in["hist_ret_pct"], float)

    # ---- 7) 月末效应 ---------------------------------------------------- #
    mee = month_end_effect(df, as_of)
    print("[月末] is_month_end=%s 距月末=%d 日｜历史月末3日 %s%% / 次月前3日 %s%% 方向=%+d" % (
        mee["is_month_end"], mee["days_to_month_end"], mee["hist_last3_ret"],
        mee["hist_first3_ret"], mee["direction"]))
    assert isinstance(mee["is_month_end"], bool)
    assert 0.0 <= mee["strength"] <= MAX_CALENDAR_STRENGTH + 1e-9

    # ---- 8) 斐波那契时间窗 ---------------------------------------------- #
    ftw = fib_time_window(df, as_of)
    print("[斐波时间窗] 锚点 %s(%s %.2f) 已过 %d 个交易日｜未来窗口：%s" % (
        ftw["anchor_date"], ftw["anchor_type"], ftw["anchor_price"], ftw["bars_since"],
        "; ".join("%s(+%d)" % (w["date"], w["bars_ahead"]) for w in ftw["next_windows"][:5])))
    prev_d = None
    for w in ftw["next_windows"]:
        assert 0 <= w["bars_ahead"] <= 60
        d = pd.Timestamp(w["date"])
        assert d >= as_of, "窗口日期必须在未来（>= as_of）"
        assert d.weekday() < 5, "窗口日期必须是交易日（非周末）：%s" % w["date"]
        if prev_d is not None:
            assert d >= prev_d, "窗口日期必须递增"
        prev_d = d
    assert len(ftw["next_windows"]) >= 2, "未来 60 个交易日内应有至少 2 个斐波那契窗口"
    assert ftw["anchor_type"] in ("high", "low")

    # ---- 9) 江恩时间周期 ------------------------------------------------ #
    gc = gann_time_cycle(df, as_of)
    print("[江恩周期] 锚点 %s｜%s" % (gc["anchor_date"], "; ".join(
        "%d日→%s(%s)" % (c["cycle"], c["date"], c["kind"]) for c in gc["cycles"][:4])))
    assert len(gc["cycles"]) == len(GANN_CYCLES)
    assert [c["cycle"] for c in gc["cycles"]] == sorted(GANN_CYCLES)
    assert gc["direction"] == 0, "江恩时间周期不给方向"

    # ---- 10) 节气 ------------------------------------------------------- #
    for base in ("2026-01-15", "2026-06-01", "2025-12-25"):
        terms = solar_terms(pd.Timestamp(base), 6)
        print("[节气] 基准 %s → %s" % (base, "、".join(
            "%s %s(+%d)" % (t["name"], t["date"], t["days_ahead"]) for t in terms)))
        prev = None
        for t in terms:
            d = pd.Timestamp(t["date"])
            assert d >= pd.Timestamp(base), "节气日期必须在未来"
            assert t["days_ahead"] >= 0
            if prev is not None:
                assert d > prev, "节气日期必须严格递增"
            prev = d
        assert len(terms) == 6
    # 立春 2026 应为 2026-02-04（题目给出的校验点）
    lc = [t for t in solar_terms(pd.Timestamp("2026-01-01"), 24) if t["name"] == "立春"]
    assert lc and lc[0]["date"] == "2026-02-04", "立春近似日期应为 2026-02-04，实际 %s" % (
        lc[0]["date"] if lc else "无")
    lx = [t for t in solar_terms(pd.Timestamp("2026-01-01"), 24) if t["name"] == "立夏"]
    assert lx and lx[0]["date"] == "2026-05-05", "立夏近似日期应为 2026-05-05，实际 %s" % (
        lx[0]["date"] if lx else "无")

    # ---- 11) 因子输出 --------------------------------------------------- #
    factors = seasonality_factors("600519", df, as_of, fin)
    print("-" * 78)
    print("生成因子 %d 个：" % len(factors))
    for fac in factors:
        assert isinstance(fac, Factor), "必须是 Factor 实例"
        assert fac.category == Category.CYCLE
        assert 0.0 <= fac.strength <= 1.0 and 0.0 <= fac.reliability <= 1.0
        assert fac.strength <= MAX_CALENDAR_STRENGTH + 1e-9, \
            "%s 的 strength 必须 ≤0.35，实际 %.3f" % (fac.id, fac.strength)
        print("  %-20s %-16s 方向=%+d 强度=%.2f 可信=%.2f 周期=%-22s 理论=%s" % (
            fac.id, fac.name, fac.direction, fac.strength, fac.reliability,
            "/".join(fac.horizon) if fac.horizon else "全部", fac.theory))
    expect = {"calendar_effect", "month_effect", "weekday_effect", "holiday_effect",
              "earnings_window", "two_sessions", "month_end", "fib_time_window",
              "gann_time_cycle", "solar_term"}
    got = {f.id for f in factors}
    assert expect.issubset(got), "缺少因子: %s" % (expect - got)
    hor = {f.id: f.horizon for f in factors}
    assert hor["calendar_effect"] == ("d20", "m1", "q1")
    assert hor["month_effect"] == ("d20", "m1", "q1")
    assert hor["earnings_window"] == ("d5", "d10", "d20", "m1")
    assert hor["fib_time_window"] == ("d5", "d10", "d20")
    assert hor["gann_time_cycle"] == ("d5", "d10", "d20")
    assert hor["holiday_effect"] == ("d5", "d10")

    # ---- 12) 汇总 ------------------------------------------------------- #
    rep = seasonality_report("600519", df, as_of)
    print("-" * 78)
    print("seasonality_report 关键字段：")
    print("  as_of      : %s（%s，%d 月，距月末 %d 个交易日）" % (
        rep["as_of"], rep["today"]["weekday"], rep["today"]["month"],
        rep["today"]["days_to_month_end"]))
    print("  命中效应   : %s" % "、".join(rep["today"]["effects"]) or "无")
    print("  month      : %d月 融合 %+.2f%% 方向=%+d" % (
        rep["month"]["current_month"], rep["month"]["blended_ret_pct"],
        rep["month"]["direction"]))
    print("  weekday    : 最强 %s / 最弱 %s" % (
        rep["today"]["weekday_best"] or "—", rep["today"]["weekday_worst"] or "—"))
    print("  holiday    : 节后5日 %s%% 方向=%+d（%d 个长假样本）" % (
        rep["holiday"]["post5_ret"], rep["holiday"]["direction"],
        rep["holiday"]["n_holidays"]))
    print("  earnings   : %s 报告期 %s 距截止 %d 天 风险=%s" % (
        rep["earnings"]["window"], rep["earnings"]["next_report_period"],
        rep["earnings"]["days_to_deadline"], rep["earnings"]["risk"]))
    print("  two_sess   : in=%s 距 %+d 天 历史 %s%%" % (
        rep["two_sessions"]["in_session"], rep["two_sessions"]["days_to"],
        rep["two_sessions"]["hist_ret_pct"]))
    print("  month_end  : is=%s 距 %d 日 历史月末3日 %s%%" % (
        rep["month_end"]["is_month_end"], rep["month_end"]["days_to_month_end"],
        rep["month_end"]["hist_last3_ret"]))
    print("  fib_window : 锚 %s 未来 %d 个窗口，最近 %s" % (
        rep["fib_window"]["anchor_date"], len(rep["fib_window"]["next_windows"]),
        rep["fib_window"]["next_windows"][0]["date"] if rep["fib_window"]["next_windows"] else "—"))
    print("  gann_cycle : 锚 %s 临近=%s" % (rep["gann_cycle"]["anchor_date"],
                                            rep["gann_cycle"]["near_cycle"]))
    print("  solar      : %s" % "、".join("%s %s" % (t["name"], t["date"])
                                          for t in rep["solar_terms"][:3]))
    print("  consensus  : 方向=%+d 分=%+.3f（偏多 %d / 偏空 %d）" % (
        rep["consensus"]["direction"], rep["consensus"]["score"],
        rep["consensus"]["bull_count"], rep["consensus"]["bear_count"]))
    print("               %s" % rep["consensus"]["detail"])
    assert set(rep.keys()) >= {"calendar", "month", "weekday", "holiday", "earnings",
                               "two_sessions", "month_end", "fib_window",
                               "gann_cycle", "solar_terms", "consensus"}
    assert rep["consensus"]["direction"] in (-1, 0, 1)

    # ---- 13) 降级容错 --------------------------------------------------- #
    print("-" * 78)
    for bad in (None, pd.DataFrame(), make_ohlcv(12), "not a frame", 12345):
        assert isinstance(calendar_effects("600519", bad), dict)
        assert isinstance(month_effect(bad), dict)
        assert isinstance(weekday_effect(bad), dict)
        assert isinstance(holiday_effects(bad), dict)
        assert isinstance(earnings_window("600519", None, None), dict)
        assert isinstance(two_sessions(bad), dict)
        assert isinstance(month_end_effect(bad), dict)
        assert isinstance(fib_time_window(bad), dict)
        assert isinstance(gann_time_cycle(bad), dict)
        assert isinstance(seasonality_factors("600519", bad), list)
        assert isinstance(seasonality_report("600519", bad), dict)
    assert len(solar_terms(None, 6)) == 6
    idx = make_index(900)
    assert isinstance(seasonality_report("sh000001", idx, None), dict)
    print("降级容错：空表/短表/非法输入/None 基准日/指数数据 全部安全返回 ✓")

    print("-" * 78)
    print("seasonality self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()

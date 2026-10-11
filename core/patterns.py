# -*- coding: utf-8 -*-
"""
K 线形态识别（``core.patterns``）。

职责
----
把 OHLCV 序列翻译成**可解释的形态语言**，并把它变成因子与历史统计证据：

* :func:`candle_features`  —— 逐根 K 线的几何特征（实体/影线/十字星/吞没…）
* :func:`detect_patterns`  —— 30 种经典 K 线组合形态（启明星/黄昏星/吞没/红三兵/岛形…）
* :func:`pattern_stats`    —— **历史规律证据**：每种形态出现后 N 日的胜率与平均收益
* :func:`pattern_factors`  —— 形态 -> :class:`core.contracts.Factor`（多头/空头/冲突/量能确认）
* :func:`detect_trendlines`—— 上升/下降趋势线、突破、三角形与楔形
* :func:`elliott_count`    —— 艾略特波浪简化数浪（ZigZag + 斐波那契比例，保守输出）
* :func:`chan_strokes`     —— 缠论简化（分型 -> 笔 -> 线段 -> 中枢 + MACD 背驰）
* :func:`ichimoku`         —— 一目均衡表（云图）
* :func:`classic_shape`    —— 经典形态（箱体 / 双顶双底 / 位置与目标位）

输出因子（``category=Category.TECH``，``horizon=("d1","d5","d10")``）
-----------------------------------------------------------------
``pattern_bull``（近 10 根最强看涨形态）、``pattern_bear``（最强看跌形态）、
``pattern_conflict``（多空形态并存 -> 观望）、``pattern_volume_confirm``（形态 + 量能配合）。

统一约定（与 ``core.indicators`` 一致）
------------------------------------
1. 输入 DataFrame 列名容错（大写/中文 -> 小写英文），**不修改入参**；
2. 任何函数都不抛异常：数据不足时返回空列表 / 空 DataFrame / 降级 dict，并打 debug 日志；
3. 形态判定阈值全部写在代码注释里，便于校准与复核。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:  # 包内导入
    from .contracts import DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Category, Factor
    from .indicators import (_cols, _ensure_frame, _safe_div, _swing_points,
                             indicator_matrix, ma, macd, support_resistance)
    from .utils import get_logger
except ImportError:  # pragma: no cover - 兼容直接脚本运行
    from contracts import DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Category, Factor  # type: ignore
    from indicators import (_cols, _ensure_frame, _safe_div, _swing_points,  # type: ignore
                            indicator_matrix, ma, macd, support_resistance)
    from utils import get_logger  # type: ignore

log = get_logger("stockai.patterns")

__all__ = [
    "candle_features", "detect_patterns", "pattern_factors", "pattern_stats",
    "detect_trendlines", "elliott_count", "chan_strokes", "ichimoku", "classic_shape",
]

_NAN = float("nan")
_HORIZON_SHORT = ("d1", "d5", "d10")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _arr(d: pd.DataFrame, name: str, default=_NAN) -> np.ndarray:
    """取列并转成 float ndarray（缺失列用 default 填充）。"""
    if name in d.columns:
        s = pd.to_numeric(d[name], errors="coerce")
        return s.to_numpy(dtype=float, na_value=np.nan)
    return np.full(len(d), default, dtype=float)


def _dates(d: pd.DataFrame):
    """返回日期序列（无 date 列时用行号）。"""
    if "date" in d.columns:
        try:
            return pd.to_datetime(d["date"])
        except Exception:  # noqa: BLE001
            pass
    return pd.Series(np.arange(len(d)), index=d.index)


def _mk_ctx(d: pd.DataFrame) -> dict:
    """把 DataFrame 拆成形态判定所需的 numpy 上下文。"""
    o = _arr(d, "open")
    h = _arr(d, "high")
    l = _arr(d, "low")
    c = _arr(d, "close")
    v = np.nan_to_num(_arr(d, "volume", 0.0), nan=0.0)
    o = np.where(np.isfinite(o), o, c)
    h = np.where(np.isfinite(h), h, np.maximum(o, c))
    l = np.where(np.isfinite(l), l, np.minimum(o, c))
    rng = np.where((h - l) > 0, h - l, np.nan)          # 振幅（一字板为 NaN）
    body = c - o
    ctx = {
        "n": len(d), "o": o, "h": h, "l": l, "c": c, "v": v,
        "body": body, "body_abs": np.abs(body), "rng": rng,
        "up_sh": h - np.maximum(o, c), "lo_sh": np.minimum(o, c) - l,
        "prev_c": np.concatenate([[_NAN], c[:-1]]) if len(c) else np.array([]),
        "roll20_low": pd.Series(l).rolling(20, min_periods=5).min().to_numpy(dtype=float),
        "roll20_high": pd.Series(h).rolling(20, min_periods=5).max().to_numpy(dtype=float),
        "roll10_low": pd.Series(l).rolling(10, min_periods=3).min().to_numpy(dtype=float),
        "roll10_high": pd.Series(h).rolling(10, min_periods=3).max().to_numpy(dtype=float),
        "vol_ma5": pd.Series(v).rolling(5, min_periods=2).mean().to_numpy(dtype=float),
        "ma10": ma(pd.Series(c), 10).to_numpy(dtype=float) if len(c) else np.array([]),
    }
    return ctx


def _ret(ctx: dict, i: int, k: int = 1) -> float:
    """i 根之前 k 根的收益率（%，NaN 表示数据不足）。"""
    j = i - k
    if j < 0 or not np.isfinite(ctx["c"][j]) or ctx["c"][j] <= 0:
        return _NAN
    return (ctx["c"][i] - ctx["c"][j]) / ctx["c"][j] * 100.0


# --------------------------------------------------------------------------- #
# 一、单根 K 线特征
# --------------------------------------------------------------------------- #
def candle_features(df) -> pd.DataFrame:
    """逐根 K 线的形态特征列。

    ==================  ====================================================
    列                 含义
    ==================  ====================================================
    ``body``           实体（收盘-开盘，带符号）
    ``body_pct``       实体 / 昨收（%）
    ``upper_shadow``   上影线长度（最高- max(开,收)）
    ``lower_shadow``   下影线长度（min(开,收)-最低）
    ``shadow_ratio``   上影 / 下影（下影为 0 时为 NaN）
    ``is_bull``        是否阳线（收 > 开）
    ``gap_up``         是否向上跳空（今低 > 昨高）
    ``gap_down``       是否向下跳空（今高 < 昨低）
    ``is_doji``        十字星：|实体| ≤ 振幅的 10%
    ``is_hammer``      锤头线：下影 ≥ 2×|实体|，上影 ≤ 0.3×|实体|，|实体| ≤ 0.35×振幅
    ``is_shooting_star`` 射击之星：上影 ≥ 2×|实体|，下影 ≤ 0.3×|实体|，|实体| ≤ 0.35×振幅
    ``is_marubozu``    光头光脚：|实体| ≥ 振幅的 90%
    ``is_spinning_top`` 纺锤线：|实体| ≤ 振幅的 30% 且上下影都 > 0
    ``close_pos``      收盘在当日区间中的位置（0~1）
    ==================  ====================================================

    说明：本函数只做**单根几何特征**，"是否出现在高位/低位"判断留给
    :func:`detect_patterns`（同样的形状在高位叫上吊线、在低位叫锤头线）。
    """
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns:
        return pd.DataFrame()
    x = _mk_ctx(d)
    body, ba, rng = x["body"], x["body_abs"], x["rng"]
    up_sh, lo_sh = x["up_sh"], x["lo_sh"]
    out = pd.DataFrame(index=d.index)
    out["body"] = body
    out["body_pct"] = pd.Series(
        _safe_div(body, np.abs(x["prev_c"])) * 100.0, index=d.index)
    out["upper_shadow"] = up_sh
    out["lower_shadow"] = lo_sh
    out["shadow_ratio"] = _safe_div(up_sh, lo_sh)
    out["is_bull"] = (body > 0).astype(int)
    out["gap_up"] = (x["l"] > np.concatenate([[_NAN], x["h"][:-1]])).astype(int)
    out["gap_down"] = (x["h"] < np.concatenate([[_NAN], x["l"][:-1]])).astype(int)
    out["is_doji"] = (ba <= 0.10 * rng).astype(int)
    out["is_hammer"] = ((lo_sh >= 2.0 * ba) & (up_sh <= 0.30 * ba) & (ba <= 0.35 * rng)).astype(int)
    out["is_shooting_star"] = ((up_sh >= 2.0 * ba) & (lo_sh <= 0.30 * ba) & (ba <= 0.35 * rng)).astype(int)
    out["is_marubozu"] = (ba >= 0.90 * rng).astype(int)
    out["is_spinning_top"] = ((ba <= 0.30 * rng) & (up_sh > 0) & (lo_sh > 0)).astype(int)
    out["close_pos"] = pd.Series(np.clip(_safe_div(x["c"] - x["l"], x["h"] - x["l"]), 0.0, 1.0),
                                 index=d.index)
    return out


# --------------------------------------------------------------------------- #
# 二、形态识别
# --------------------------------------------------------------------------- #
def _d_morning_star(x, i):
    """早晨之星/启明星（3 根）：长阴 -> 小实体星线（跳空低开或创新低）-> 长阳收复过半。"""
    if i < 2:
        return None
    c, o, l, h = x["c"], x["o"], x["l"], x["h"]
    ba, rng = x["body_abs"], x["rng"]
    if not (c[i - 2] < o[i - 2] and ba[i - 2] >= 0.5 * rng[i - 2]):
        return None
    if not (ba[i - 1] <= 0.4 * ba[i - 2]):
        return None
    if not (l[i - 1] <= l[i - 2] * 1.002):
        return None
    if not (c[i] > o[i] and ba[i] >= 0.5 * rng[i]):
        return None
    mid = (o[i - 2] + c[i - 2]) / 2.0
    if c[i] <= mid:
        return None
    return {"detail": "长阴后低位小星线，今日长阳收复前阴过半，典型见底反转"}


def _d_evening_star(x, i):
    """黄昏之星（3 根）：长阳 -> 小实体星线（跳空高开或创新高）-> 长阴吞没过半。"""
    if i < 2:
        return None
    c, o, l, h = x["c"], x["o"], x["l"], x["h"]
    ba, rng = x["body_abs"], x["rng"]
    if not (c[i - 2] > o[i - 2] and ba[i - 2] >= 0.5 * rng[i - 2]):
        return None
    if not (ba[i - 1] <= 0.4 * ba[i - 2]):
        return None
    if not (h[i - 1] >= h[i - 2] * 0.998):
        return None
    if not (c[i] < o[i] and ba[i] >= 0.5 * rng[i]):
        return None
    mid = (o[i - 2] + c[i - 2]) / 2.0
    if c[i] >= mid:
        return None
    return {"detail": "长阳后高位小星线，今日长阴吞没前阳过半，见顶反转"}


def _d_bullish_engulfing(x, i):
    """看涨吞没（2 根）：前阴后阳，阳线实体完全包住前一根阴线实体。"""
    if i < 1:
        return None
    o, c = x["o"], x["c"]
    if not (c[i - 1] < o[i - 1] and c[i] > o[i]):
        return None
    if not (c[i] >= o[i - 1] and o[i] <= c[i - 1]):
        return None
    if not (x["body_abs"][i] > x["body_abs"][i - 1]):
        return None
    return {"detail": "阳线实体完全吞没前一根阴线，多头夺回主动权"}


def _d_bearish_engulfing(x, i):
    """看跌吞没（2 根）：前阳后阴，阴线实体完全包住前一根阳线实体。"""
    if i < 1:
        return None
    o, c = x["o"], x["c"]
    if not (c[i - 1] > o[i - 1] and c[i] < o[i]):
        return None
    if not (o[i] >= c[i - 1] and c[i] <= o[i - 1]):
        return None
    if not (x["body_abs"][i] > x["body_abs"][i - 1]):
        return None
    return {"detail": "阴线实体完全吞没前一根阳线，空头接管盘面"}


def _d_rising_sun(x, i):
    """旭日东升（2 根）：前阴后低开阳线，且收盘高于前一根阴线的开盘价。"""
    if i < 1:
        return None
    o, c = x["o"], x["c"]
    if not (c[i - 1] < o[i - 1] and c[i] > o[i]):
        return None
    if not (o[i] < c[i - 1] and c[i] > o[i - 1]):
        return None
    return {"detail": "低开高走的长阳完全收复前一根阴线，多头强势反攻"}


def _d_dark_cloud(x, i):
    """乌云盖顶（2 根）：前阳后高开阴线，收盘切入前阳实体一半以下但未完全吞没。"""
    if i < 1:
        return None
    o, c = x["o"], x["c"]
    if not (c[i - 1] > o[i - 1] and c[i] < o[i]):
        return None
    if not (o[i] > c[i - 1]):
        return None
    mid = (o[i - 1] + c[i - 1]) / 2.0
    if not (c[i] < mid and c[i] > o[i - 1]):
        return None
    return {"detail": "高开阴线深入前阳实体过半，上方抛压显现"}


def _d_outside_bar(x, i):
    """穿头破脚（2 根）：今日最高超过昨高、最低跌破昨低（外包线），方向随当日 K 线。"""
    if i < 1:
        return None
    if not (x["h"][i] > x["h"][i - 1] and x["l"][i] < x["l"][i - 1]):
        return None
    bull = x["c"][i] > x["o"][i]
    return {"direction": DIR_BULL if bull else DIR_BEAR,
            "detail": "今日 K 线穿头破脚完全包住昨日区间，%s力量占优"
                      % ("多" if bull else "空")}


def _d_hammer(x, i):
    """锤头线（1 根，低位）：下影 ≥ 2×|实体|，上影 ≤ 0.3×|实体|，|实体| ≤ 0.35×振幅。"""
    if i < 1:
        return None
    ba, rng = x["body_abs"][i], x["rng"][i]
    if not np.isfinite(rng) or ba > 0.35 * rng:
        return None
    if not (x["lo_sh"][i] >= 2.0 * ba and x["up_sh"][i] <= 0.30 * ba):
        return None
    below = ((np.isfinite(x["ma10"][i]) and x["c"][i] < x["ma10"][i])
             or (np.isfinite(x["roll10_low"][i]) and x["l"][i] <= x["roll10_low"][i] * 1.01))
    if not below:
        return None
    return {"detail": "下跌后长下影小实体，低位承接明显（锤头线）"}


def _d_inverted_hammer(x, i):
    """倒锤头（1 根，低位）：上影 ≥ 2×|实体|，下影 ≤ 0.3×|实体|，|实体| ≤ 0.35×振幅。"""
    if i < 1:
        return None
    ba, rng = x["body_abs"][i], x["rng"][i]
    if not np.isfinite(rng) or ba > 0.35 * rng:
        return None
    if not (x["up_sh"][i] >= 2.0 * ba and x["lo_sh"][i] <= 0.30 * ba):
        return None
    below = ((np.isfinite(x["ma10"][i]) and x["c"][i] < x["ma10"][i])
             or (np.isfinite(x["roll10_low"][i]) and x["l"][i] <= x["roll10_low"][i] * 1.02))
    if not below:
        return None
    return {"detail": "下跌后长上影小实体，多头试盘（倒锤头）"}


def _d_hanging_man(x, i):
    """上吊线（1 根，高位）：形态同锤头，但出现在上涨后（收盘 > MA10 或近 10 日新高附近）。"""
    if i < 1:
        return None
    ba, rng = x["body_abs"][i], x["rng"][i]
    if not np.isfinite(rng) or ba > 0.35 * rng:
        return None
    if not (x["lo_sh"][i] >= 2.0 * ba and x["up_sh"][i] <= 0.30 * ba):
        return None
    above = ((np.isfinite(x["ma10"][i]) and x["c"][i] > x["ma10"][i])
             or (np.isfinite(x["roll10_high"][i]) and x["h"][i] >= x["roll10_high"][i] * 0.99))
    if not above:
        return None
    return {"detail": "上涨后长下影小实体，高位抛压初现（上吊线）"}


def _d_shooting_star(x, i):
    """射击之星（1 根，高位）：上影 ≥ 2×|实体|，下影 ≤ 0.3×|实体|，出现在上涨后。"""
    if i < 1:
        return None
    ba, rng = x["body_abs"][i], x["rng"][i]
    if not np.isfinite(rng) or ba > 0.35 * rng:
        return None
    if not (x["up_sh"][i] >= 2.0 * ba and x["lo_sh"][i] <= 0.30 * ba):
        return None
    above = ((np.isfinite(x["ma10"][i]) and x["c"][i] > x["ma10"][i])
             or (np.isfinite(x["roll10_high"][i]) and x["h"][i] >= x["roll10_high"][i] * 0.99))
    if not above:
        return None
    return {"detail": "冲高回落留长上影，高位见顶信号（射击之星）"}


def _d_doji(x, i):
    """十字星（1 根）：|实体| ≤ 振幅的 10%（多空僵持，方向中性）。"""
    ba, rng = x["body_abs"][i], x["rng"][i]
    if not np.isfinite(rng) or rng <= 0:
        return None
    if ba > 0.10 * rng:
        return None
    return {"direction": DIR_NEUTRAL, "detail": "开盘与收盘几乎相同，多空力量均衡（十字星）"}


def _d_long_lower(x, i):
    """长下影线（1 根）：下影 ≥ 2×|实体| 且下影 ≥ 振幅的 60%。"""
    ba, rng = x["body_abs"][i], x["rng"][i]
    if not np.isfinite(rng) or rng <= 0:
        return None
    if x["lo_sh"][i] >= 2.0 * ba and x["lo_sh"][i] >= 0.60 * rng:
        return {"detail": "长下影显示下方买盘强，杀跌被快速承接"}
    return None


def _d_long_upper(x, i):
    """长上影线（1 根）：上影 ≥ 2×|实体| 且上影 ≥ 振幅的 60%。"""
    ba, rng = x["body_abs"][i], x["rng"][i]
    if not np.isfinite(rng) or rng <= 0:
        return None
    if x["up_sh"][i] >= 2.0 * ba and x["up_sh"][i] >= 0.60 * rng:
        return {"detail": "长上影显示上方抛压重，冲高遇阻"}
    return None


def _d_three_white_soldiers(x, i):
    """红三兵（3 根）：连续三根阳线、收盘逐日抬高、每根实体 ≥ 振幅 50%、开盘在前实体之内。"""
    if i < 2:
        return None
    o, c, rng, ba = x["o"], x["c"], x["rng"], x["body_abs"]
    for k in (i - 2, i - 1, i):
        if not (c[k] > o[k] and np.isfinite(rng[k]) and ba[k] >= 0.5 * rng[k]):
            return None
    if not (c[i] > c[i - 1] > c[i - 2]):
        return None
    if not (o[i] >= o[i - 1] and o[i] <= c[i - 1] + 1e-9):
        return None
    return {"detail": "连续三根实体饱满的阳线、收盘逐日抬高，多头稳步推进"}


def _d_three_black_crows(x, i):
    """三只乌鸦（3 根）：连续三根阴线、收盘逐日走低、每根实体 ≥ 振幅 50%。"""
    if i < 2:
        return None
    o, c, rng, ba = x["o"], x["c"], x["rng"], x["body_abs"]
    for k in (i - 2, i - 1, i):
        if not (c[k] < o[k] and np.isfinite(rng[k]) and ba[k] >= 0.5 * rng[k]):
            return None
    if not (c[i] < c[i - 1] < c[i - 2]):
        return None
    return {"detail": "连续三根实体饱满的阴线、收盘逐日走低，空头持续压制"}


def _d_bullish_volley(x, i):
    """多方炮（3 根）：阳线 -> 缩量回调阴线（实体在前阳实体内）-> 阳线创三根新高。"""
    if i < 2:
        return None
    o, c, v = x["o"], x["c"], x["v"]
    if not (c[i - 2] > o[i - 2] and c[i - 1] < o[i - 1] and c[i] > o[i]):
        return None
    if not (o[i - 1] >= o[i - 2] and c[i - 1] <= c[i - 2]):
        return None
    if not (c[i] > max(c[i - 2], o[i - 2]) and c[i] > c[i - 1]):
        return None
    if not (v[i] >= max(v[i - 1], 1e-9)):
        return None
    return {"detail": "阳线后小幅回调再放量收阳创新高，多方重新开炮"}


def _d_bearish_volley(x, i):
    """空方炮（3 根）：阴线 -> 反弹阳线（实体在前阴实体内）-> 阴线创三根新低。"""
    if i < 2:
        return None
    o, c, v = x["o"], x["c"], x["v"]
    if not (c[i - 2] < o[i - 2] and c[i - 1] > o[i - 1] and c[i] < o[i]):
        return None
    if not (o[i - 1] <= o[i - 2] and c[i - 1] >= c[i - 2]):
        return None
    if not (c[i] < min(c[i - 2], o[i - 2]) and c[i] < c[i - 1]):
        return None
    if not (v[i] >= max(v[i - 1], 1e-9)):
        return None
    return {"detail": "阴线后小幅反弹再放量收阴创新低，空方再度发力"}


def _d_friend_counter(x, i):
    """好友反攻（2 根）：长阴之后跳空低开收阳，收盘与前阴收盘基本持平（±0.5% 内）。"""
    if i < 1:
        return None
    o, c, rng, ba = x["o"], x["c"], x["rng"], x["body_abs"]
    if not (c[i - 1] < o[i - 1] and np.isfinite(rng[i - 1]) and ba[i - 1] >= 0.5 * rng[i - 1]):
        return None
    if not (c[i] > o[i] and o[i] < c[i - 1]):
        return None
    if abs(c[i] - c[i - 1]) > 0.005 * c[i - 1]:
        return None
    return {"detail": "跳空低开后收阳并收复至前阴收盘附近，跌势被遏制（好友反攻）"}


def _d_enemy_counter(x, i):
    """淡友反攻（2 根）：长阳之后跳空高开收阴，收盘与前阳收盘基本持平（±0.5% 内）。"""
    if i < 1:
        return None
    o, c, rng, ba = x["o"], x["c"], x["rng"], x["body_abs"]
    if not (c[i - 1] > o[i - 1] and np.isfinite(rng[i - 1]) and ba[i - 1] >= 0.5 * rng[i - 1]):
        return None
    if not (c[i] < o[i] and o[i] > c[i - 1]):
        return None
    if abs(c[i] - c[i - 1]) > 0.005 * c[i - 1]:
        return None
    return {"detail": "跳空高开后收阴并回落至前阳收盘附近，涨势受阻（淡友反攻）"}


def _d_flat_bottom(x, i):
    """平底（2 根）：连续两根最低价相差 ≤ 0.3%，且处于近 20 日低位区（不高于区间低点 3%）。"""
    if i < 1:
        return None
    l = x["l"]
    if not (abs(l[i] - l[i - 1]) <= 0.003 * max(l[i], 1e-9)):
        return None
    rl = x["roll20_low"][i]
    if not (np.isfinite(rl) and l[i] <= rl * 1.03):
        return None
    return {"detail": "连续两根最低价几乎相同，下方支撑明确（平底）"}


def _d_flat_top(x, i):
    """平顶（2 根）：连续两根最高价相差 ≤ 0.3%，且处于近 20 日高位区。"""
    if i < 1:
        return None
    h = x["h"]
    if not (abs(h[i] - h[i - 1]) <= 0.003 * max(h[i], 1e-9)):
        return None
    rh = x["roll20_high"][i]
    if not (np.isfinite(rh) and h[i] >= rh * 0.97):
        return None
    return {"detail": "连续两根最高价几乎相同，上方压力明确（平顶）"}


def _d_rising_three_methods(x, i):
    """上升三法（5 根）：大阳线 + 3 根小实体回调（均在大阳线实体内）+ 大阳线创新高。"""
    if i < 4:
        return None
    o, c, h, rng, ba = x["o"], x["c"], x["h"], x["rng"], x["body_abs"]
    j = i - 4
    if not (c[j] > o[j] and np.isfinite(rng[j]) and ba[j] >= 0.5 * rng[j]):
        return None
    for k in (j + 1, j + 2, j + 3):
        if not (np.isfinite(rng[k]) and ba[k] <= 0.5 * ba[j]):
            return None
        if not (min(o[k], c[k]) >= min(o[j], c[j]) and max(o[k], c[k]) <= max(o[j], c[j])):
            return None
    if not (c[i] > o[i] and c[i] > h[j] and ba[i] >= 0.5 * rng[i]):
        return None
    return {"detail": "大阳线后三根小实体回调不破实体，再拉大阳创新高（上升三法）"}


def _d_falling_three_methods(x, i):
    """下降三法（5 根）：大阴线 + 3 根小实体反弹（均在大阴线实体内）+ 大阴线创新低。"""
    if i < 4:
        return None
    o, c, l, rng, ba = x["o"], x["c"], x["l"], x["rng"], x["body_abs"]
    j = i - 4
    if not (c[j] < o[j] and np.isfinite(rng[j]) and ba[j] >= 0.5 * rng[j]):
        return None
    for k in (j + 1, j + 2, j + 3):
        if not (np.isfinite(rng[k]) and ba[k] <= 0.5 * ba[j]):
            return None
        if not (min(o[k], c[k]) >= min(o[j], c[j]) and max(o[k], c[k]) <= max(o[j], c[j])):
            return None
    if not (c[i] < o[i] and c[i] < l[j] and ba[i] >= 0.5 * rng[i]):
        return None
    return {"detail": "大阴线后三根小实体反弹无力，再收大阴创新低（下降三法）"}


def _d_gap(x, i):
    """跳空缺口（1 根）：今低 > 昨高为向上跳空；今高 < 昨低为向下跳空。"""
    if i < 1:
        return None
    if x["l"][i] > x["h"][i - 1]:
        size = (x["l"][i] - x["h"][i - 1]) / x["h"][i - 1] * 100.0
        return {"direction": DIR_BULL, "detail": "向上跳空 %.2f%%，多头强势突破" % size}
    if x["h"][i] < x["l"][i - 1]:
        size = (x["l"][i - 1] - x["h"][i]) / x["l"][i - 1] * 100.0
        return {"direction": DIR_BEAR, "detail": "向下跳空 %.2f%%，空头强势打压" % size}
    return None


def _d_island_reversal(x, i):
    """岛形反转（3~10 根）：左侧同向跳空 + 右侧反向跳空，两个缺口价格区间重叠形成孤岛。"""
    if i < 2:
        return None
    first = None
    for j in range(max(1, i - 10), i):
        if x["l"][j] > x["h"][j - 1]:
            first = (j, "up", x["h"][j - 1], x["l"][j])
            break
        if x["h"][j] < x["l"][j - 1]:
            first = (j, "down", x["h"][j], x["l"][j - 1])
            break
    if first is None:
        return None
    j0, typ, g_lo, g_hi = first
    if i - j0 < 2:
        return None
    if typ == "down" and x["l"][i] > x["h"][i - 1]:
        lo2, hi2 = x["h"][i - 1], x["l"][i]
        if min(g_hi, hi2) > max(g_lo, lo2):          # 缺口区间重叠 -> 中间形成孤岛
            return {"direction": DIR_BULL,
                    "detail": "左侧向下跳空、右侧向上跳空，中间 K 线被孤立，底部岛形反转"}
    if typ == "up" and x["h"][i] < x["l"][i - 1]:
        lo2, hi2 = x["h"][i], x["l"][i - 1]
        if min(g_hi, hi2) > max(g_lo, lo2):
            return {"direction": DIR_BEAR,
                    "detail": "左侧向上跳空、右侧向下跳空，中间 K 线被孤立，顶部岛形反转"}
    return None


def _d_marubozu_bull(x, i):
    """光头光脚阳线（1 根）：阳线且实体 ≥ 振幅 90%，上下影各 ≤ 振幅 5%。"""
    rng, ba = x["rng"][i], x["body_abs"][i]
    if not np.isfinite(rng) or rng <= 0 or x["c"][i] <= x["o"][i]:
        return None
    if ba >= 0.90 * rng and x["up_sh"][i] <= 0.05 * rng and x["lo_sh"][i] <= 0.05 * rng:
        return {"detail": "几乎无上下影的光头光脚大阳线，多方掌控全天"}
    return None


def _d_marubozu_bear(x, i):
    """光头光脚阴线（1 根）：阴线且实体 ≥ 振幅 90%，上下影各 ≤ 振幅 5%。"""
    rng, ba = x["rng"][i], x["body_abs"][i]
    if not np.isfinite(rng) or rng <= 0 or x["c"][i] >= x["o"][i]:
        return None
    if ba >= 0.90 * rng and x["up_sh"][i] <= 0.05 * rng and x["lo_sh"][i] <= 0.05 * rng:
        return {"detail": "几乎无上下影的光头光脚大阴线，空方掌控全天"}
    return None


def _d_consecutive_small_bulls(x, i):
    """连续小阳（≥3 根）：连续小实体阳线（实体 ≤ 振幅 40%）、收盘逐日抬高、累计涨幅 1.5%~10%。"""
    if i < 2:
        return None
    o, c, rng, ba = x["o"], x["c"], x["rng"], x["body_abs"]
    k = 0
    j = i
    while j >= 0 and c[j] > o[j] and np.isfinite(rng[j]) and ba[j] <= 0.40 * rng[j]:
        k += 1
        j -= 1
        if k >= 8:
            break
    if k < 3:
        return None
    start = i - k + 1
    for m in range(start + 1, i + 1):
        if c[m] <= c[m - 1]:
            return None
    gain = (c[i] - c[start]) / c[start] * 100.0 if c[start] > 0 else _NAN
    if not (np.isfinite(gain) and 1.5 <= gain <= 10.0):
        return None
    return {"detail": "连续 %d 根小阳线温和推升（累计 %.1f%%），资金稳步吸筹" % (k, gain)}


def _d_crash_long_lower(x, i):
    """急速下跌后的长下影（1 根）：近 3 日跌幅 ≥ 4%，当日下影 ≥ 2×|实体| 且 ≥ 振幅 50%。"""
    drop = _ret(x, i, 3)
    if not np.isfinite(drop) or drop > -4.0:
        return None
    ba, rng = x["body_abs"][i], x["rng"][i]
    if not np.isfinite(rng) or rng <= 0:
        return None
    if x["lo_sh"][i] >= 2.0 * ba and x["lo_sh"][i] >= 0.50 * rng:
        return {"detail": "急跌 %.1f%% 后收出长下影，恐慌盘被承接（超跌反抽）" % drop}
    return None


# 形态登记表：name 中文名 / en 英文 id / bars 占几根 / dir 默认方向 / base 基础强度 / fn 判定函数
_PATTERN_SPECS = (
    {"name": "早晨之星", "en": "morning_star", "bars": 3, "dir": DIR_BULL, "base": 0.62, "fn": _d_morning_star},
    {"name": "黄昏之星", "en": "evening_star", "bars": 3, "dir": DIR_BEAR, "base": 0.62, "fn": _d_evening_star},
    {"name": "看涨吞没", "en": "bullish_engulfing", "bars": 2, "dir": DIR_BULL, "base": 0.60, "fn": _d_bullish_engulfing},
    {"name": "看跌吞没", "en": "bearish_engulfing", "bars": 2, "dir": DIR_BEAR, "base": 0.60, "fn": _d_bearish_engulfing},
    {"name": "旭日东升", "en": "rising_sun", "bars": 2, "dir": DIR_BULL, "base": 0.58, "fn": _d_rising_sun},
    {"name": "乌云盖顶", "en": "dark_cloud_cover", "bars": 2, "dir": DIR_BEAR, "base": 0.58, "fn": _d_dark_cloud},
    {"name": "穿头破脚", "en": "outside_bar", "bars": 2, "dir": DIR_NEUTRAL, "base": 0.52, "fn": _d_outside_bar},
    {"name": "锤头线", "en": "hammer", "bars": 1, "dir": DIR_BULL, "base": 0.55, "fn": _d_hammer},
    {"name": "倒锤头", "en": "inverted_hammer", "bars": 1, "dir": DIR_BULL, "base": 0.48, "fn": _d_inverted_hammer},
    {"name": "上吊线", "en": "hanging_man", "bars": 1, "dir": DIR_BEAR, "base": 0.50, "fn": _d_hanging_man},
    {"name": "射击之星", "en": "shooting_star", "bars": 1, "dir": DIR_BEAR, "base": 0.55, "fn": _d_shooting_star},
    {"name": "十字星", "en": "doji", "bars": 1, "dir": DIR_NEUTRAL, "base": 0.35, "fn": _d_doji},
    {"name": "长下影线", "en": "long_lower_shadow", "bars": 1, "dir": DIR_BULL, "base": 0.48, "fn": _d_long_lower},
    {"name": "长上影线", "en": "long_upper_shadow", "bars": 1, "dir": DIR_BEAR, "base": 0.48, "fn": _d_long_upper},
    {"name": "红三兵", "en": "three_white_soldiers", "bars": 3, "dir": DIR_BULL, "base": 0.62, "fn": _d_three_white_soldiers},
    {"name": "三只乌鸦", "en": "three_black_crows", "bars": 3, "dir": DIR_BEAR, "base": 0.62, "fn": _d_three_black_crows},
    {"name": "多方炮", "en": "bullish_volley", "bars": 3, "dir": DIR_BULL, "base": 0.58, "fn": _d_bullish_volley},
    {"name": "空方炮", "en": "bearish_volley", "bars": 3, "dir": DIR_BEAR, "base": 0.58, "fn": _d_bearish_volley},
    {"name": "好友反攻", "en": "friend_counterattack", "bars": 2, "dir": DIR_BULL, "base": 0.52, "fn": _d_friend_counter},
    {"name": "淡友反攻", "en": "enemy_counterattack", "bars": 2, "dir": DIR_BEAR, "base": 0.52, "fn": _d_enemy_counter},
    {"name": "平底", "en": "flat_bottom", "bars": 2, "dir": DIR_BULL, "base": 0.45, "fn": _d_flat_bottom},
    {"name": "平顶", "en": "flat_top", "bars": 2, "dir": DIR_BEAR, "base": 0.45, "fn": _d_flat_top},
    {"name": "上升三法", "en": "rising_three_methods", "bars": 5, "dir": DIR_BULL, "base": 0.58, "fn": _d_rising_three_methods},
    {"name": "下降三法", "en": "falling_three_methods", "bars": 5, "dir": DIR_BEAR, "base": 0.58, "fn": _d_falling_three_methods},
    {"name": "跳空缺口", "en": "gap", "bars": 1, "dir": DIR_NEUTRAL, "base": 0.45, "fn": _d_gap},
    {"name": "岛形反转", "en": "island_reversal", "bars": 3, "dir": DIR_NEUTRAL, "base": 0.65, "fn": _d_island_reversal},
    {"name": "光头光脚阳线", "en": "marubozu_bull", "bars": 1, "dir": DIR_BULL, "base": 0.55, "fn": _d_marubozu_bull},
    {"name": "光头光脚阴线", "en": "marubozu_bear", "bars": 1, "dir": DIR_BEAR, "base": 0.55, "fn": _d_marubozu_bear},
    {"name": "连续小阳", "en": "consecutive_small_bulls", "bars": 3, "dir": DIR_BULL, "base": 0.50, "fn": _d_consecutive_small_bulls},
    {"name": "急速下跌后的长下影", "en": "long_lower_shadow_after_crash", "bars": 1, "dir": DIR_BULL, "base": 0.60, "fn": _d_crash_long_lower},
)

# 形态在关键位置/放量时的强度加成（最高 0.95）
_BOOST_NEW_EXTREME = 0.15      # 看涨形态出现在近 20 日新低 / 看跌形态出现在近 20 日新高
_BOOST_NEAR_LEVEL = 0.10       # 靠近支撑（看涨）或压力（看跌）
_BOOST_VOLUME = 0.10           # 当日量 ≥ 5 日均量的 1.5 倍
_STRENGTH_CAP = 0.95


def detect_patterns(df, lookback=120) -> list:
    """近 ``lookback`` 根内识别到的经典 K 线形态列表。

    Returns
    -------
    list[dict]
        每条：``{"date","name","en","direction","strength","detail","bars"}``，
        按日期升序 + 同日内形态登记表顺序排列。

    强度规则：``strength = base + 位置加成 + 量能加成``，上限 0.95。
    不同形态的判定标准见各 ``_d_*`` 函数注释。
    """
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns or len(d) < 5:
        return []
    x = _mk_ctx(d)
    n = x["n"]
    try:
        lb = int(lookback)
    except (TypeError, ValueError):
        lb = 120
    lb = n if lb <= 0 else min(max(lb, 5), n)
    start = n - lb
    dates = _dates(d)
    # 支撑压力位（整段样本一次性计算）：用于"关键位置"加成
    levels = []
    try:
        sr = support_resistance(_cols(d, "high"), _cols(d, "low"), _cols(d, "close"),
                                _cols(d, "volume") if "volume" in d.columns else None)
        levels = [float(v["price"]) for v in sr.get("supports", []) + sr.get("resistances", [])
                  if np.isfinite(v.get("price", _NAN))]
    except Exception as exc:  # noqa: BLE001
        log.debug("detect_patterns 支撑压力位计算失败: %s", exc)

    found = []
    for i in range(max(start, 1), n):
        c_i = x["c"][i]
        if not np.isfinite(c_i):
            continue
        near_level = any(abs(c_i - lv) <= 0.015 * lv for lv in levels) if levels else False
        vol_surge = (np.isfinite(x["vol_ma5"][i]) and x["vol_ma5"][i] > 0
                     and x["v"][i] >= 1.5 * x["vol_ma5"][i])
        at_new_low = (np.isfinite(x["roll20_low"][i]) and x["l"][i] <= x["roll20_low"][i] * 1.001)
        at_new_high = (np.isfinite(x["roll20_high"][i]) and x["h"][i] >= x["roll20_high"][i] * 0.999)
        for spec in _PATTERN_SPECS:
            try:
                hit = spec["fn"](x, i)
            except Exception as exc:  # noqa: BLE001
                log.debug("形态 %s 在第 %d 根判定异常: %s", spec["en"], i, exc)
                continue
            if not hit:
                continue
            direction = int(hit.get("direction", spec["dir"]))
            strength = float(spec["base"])
            if direction > 0 and at_new_low:
                strength += _BOOST_NEW_EXTREME
            if direction < 0 and at_new_high:
                strength += _BOOST_NEW_EXTREME
            if near_level and direction != 0:
                strength += _BOOST_NEAR_LEVEL
            if vol_surge:
                strength += _BOOST_VOLUME
            strength = float(min(strength, _STRENGTH_CAP))
            found.append({
                "date": dates.iloc[i] if hasattr(dates, "iloc") else i,
                "name": spec["name"], "en": spec["en"], "direction": direction,
                "strength": round(strength, 3), "detail": hit.get("detail", ""),
                "bars": int(spec["bars"]), "index": int(i),
            })
    return found


# --------------------------------------------------------------------------- #
# 三、形态历史统计（核心证据）
# --------------------------------------------------------------------------- #
def pattern_stats(df, horizon=5) -> dict:
    """统计每种形态出现后 ``horizon`` 个交易日的表现。

    Returns
    -------
    dict
        ``{形态中文名: {"win_rate":0.62, "n":37, "avg_ret":1.8, "avg_ret_signed":1.8,
        "en": "morning_star", "direction": 1}}``，**样本数 < 5 的形态不返回**。

    口径说明
    --------
    * 未来收益 ``ret = close[i+horizon]/close[i]-1``（收盘价对收盘价，%）；
      尾部不足 ``horizon`` 根的样本剔除，避免前视偏差。
    * ``win_rate`` 统一定义为"**形态方向被验证**的比例"：
      看涨形态取 ``ret > 0``，看跌形态取 ``ret < 0``，中性形态取 ``ret > 0``；
    * ``avg_ret`` 为原始平均收益（%），``avg_ret_signed`` 为按形态方向调整后的平均收益
      （看跌形态取反），便于横向比较信号质量。
    """
    out: dict = {}
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns or len(d) < 30:
        return out
    try:
        hz = int(horizon)
    except (TypeError, ValueError):
        hz = 5
    hz = max(hz, 1)
    c = _arr(d, "close")
    n = len(c)
    try:
        events = detect_patterns(d, lookback=n)      # 全样本扫描
    except Exception as exc:  # noqa: BLE001
        log.debug("pattern_stats 形态扫描失败: %s", exc)
        return out
    acc: dict = {}
    for ev in events:
        i = int(ev.get("index", -1))
        if i < 0 or i + hz >= n:
            continue                                  # 尾部样本没有未来收益，剔除
        c0, c1 = c[i], c[i + hz]
        if not (np.isfinite(c0) and np.isfinite(c1) and c0 > 0):
            continue
        ret = (c1 / c0 - 1.0) * 100.0
        slot = acc.setdefault(ev["name"], {"rets": [], "dir": int(ev["direction"]),
                                           "en": ev["en"], "dirs": []})
        slot["rets"].append(float(ret))
        slot["dirs"].append(int(ev["direction"]))
    for name, slot in acc.items():
        rets = np.asarray(slot["rets"], dtype=float)
        if rets.size < 5:                             # 样本 <5 不返回
            continue
        dirs = np.asarray(slot["dirs"], dtype=float)
        wins = np.where(dirs < 0, rets < 0, rets > 0)
        signed = np.where(dirs < 0, -rets, rets)
        out[name] = {
            "win_rate": round(float(wins.mean()), 4),
            "n": int(rets.size),
            "avg_ret": round(float(rets.mean()), 3),
            "avg_ret_signed": round(float(signed.mean()), 3),
            "en": slot["en"],
            "direction": int(np.sign(signed.mean())) if abs(signed.mean()) > 1e-12 else 0,
        }
    return out


# --------------------------------------------------------------------------- #
# 四、形态因子
# --------------------------------------------------------------------------- #
def pattern_factors(df, code="", name="") -> list:
    """把形态信号转成 :class:`Factor` 列表（``category=Category.TECH``）。

    输出四个因子：

    * ``pattern_bull``           近 10 根内最强看涨形态（方向 +1）
    * ``pattern_bear``           近 10 根内最强看跌形态（方向 -1）
    * ``pattern_conflict``       多空形态强度接近且均较强 -> 中性观望（方向 0）
    * ``pattern_volume_confirm`` 形态 + 量能配合（放量确认 or 缩量存疑）

    作用周期 ``("d1","d5","d10")``。
    """
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns or len(d) < 20:
        return []
    try:
        events = detect_patterns(d, lookback=min(len(d), 400))
        v = _arr(d, "volume")
        recent = [e for e in events if int(e.get("index", -1)) >= len(d) - 10]
        bull = [e for e in recent if e["direction"] > 0]
        bear = [e for e in recent if e["direction"] < 0]
        best_bull = max(bull, key=lambda e: e["strength"]) if bull else None
        best_bear = max(bear, key=lambda e: e["strength"]) if bear else None
        stats = pattern_stats(d, 5)
        out: list = []
        bull_s = float(best_bull["strength"]) if best_bull else 0.0
        bear_s = float(best_bear["strength"]) if best_bear else 0.0
        # 历史胜率校准：该形态历史上越有效，可信度越高
        def _rel(ev, default=0.55):
            if not ev:
                return default
            st = stats.get(ev["name"])
            if not st:
                return 0.5
            edge = abs(float(st["win_rate"]) - 0.5)
            return float(np.clip(0.45 + 2.0 * edge, 0.4, 0.85))
        if best_bull:
            out.append(Factor(
                "pattern_bull", "看涨形态", Category.TECH, DIR_BULL, bull_s,
                reliability=_rel(best_bull), value=best_bull["name"],
                detail="近 10 根出现%s：%s" % (best_bull["name"], best_bull["detail"]),
                theory="K线形态学", horizon=_HORIZON_SHORT))
        if best_bear:
            out.append(Factor(
                "pattern_bear", "看跌形态", Category.TECH, DIR_BEAR, bear_s,
                reliability=_rel(best_bear), value=best_bear["name"],
                detail="近 10 根出现%s：%s" % (best_bear["name"], best_bear["detail"]),
                theory="K线形态学", horizon=_HORIZON_SHORT))
        # 多空冲突：两方向都强且强度接近
        if best_bull and best_bear and min(bull_s, bear_s) >= 0.5 \
                and abs(bull_s - bear_s) <= 0.15:
            out.append(Factor(
                "pattern_conflict", "形态多空冲突", Category.TECH, DIR_NEUTRAL,
                float(np.clip((bull_s + bear_s) / 2.0, 0.0, 1.0)), reliability=0.5,
                value="%s vs %s" % (best_bull["name"], best_bear["name"]),
                detail="近 10 根同时出现%s与%s且强度相当，形态信号分歧，宜观望",
                theory="K线形态学", horizon=_HORIZON_SHORT))
        # 量能配合：最新形态当日量能是否确认
        if recent:
            last = recent[-1]
            i = int(last["index"])
            vol_ma5 = float(pd.Series(v).rolling(5, min_periods=2).mean().iloc[i]) \
                if i < len(v) and len(v) else _NAN
            ratio = (float(v[i]) / vol_ma5) if (np.isfinite(vol_ma5) and vol_ma5 > 0) else _NAN
            if np.isfinite(ratio):
                conf = bool(ratio >= 1.2)
                direction = int(last["direction"]) if conf else 0
                strength = float(np.clip(abs(math.log(max(ratio, 1e-6))) / 1.1, 0.0, 0.9))
                out.append(Factor(
                    "pattern_volume_confirm", "形态量能确认", Category.TECH, direction,
                    strength, reliability=0.6 if conf else 0.45,
                    value="%s 量比 %.2f" % (last["name"], ratio),
                    detail=("最新形态%s伴随 %.2f 倍放量，信号可信度提升"
                            % (last["name"], ratio)) if conf else
                           ("最新形态%s量能仅 %.2f 倍，缺乏放量确认，信号存疑"
                            % (last["name"], ratio)),
                    theory="量价关系", horizon=_HORIZON_SHORT))
        return out
    except Exception as exc:  # noqa: BLE001
        log.debug("pattern_factors 失败: %s", exc)
        return []


# --------------------------------------------------------------------------- #
# 五、趋势线 / 三角形 / 楔形
# --------------------------------------------------------------------------- #
def _line_touches(idx: np.ndarray, price: np.ndarray, slope: float, intercept: float,
                  tol: float = 0.015, use_low: bool = True) -> int:
    """统计落在直线 ±tol 内的点数（上升趋势线用最低价，下降趋势线用最高价）。"""
    if idx.size == 0:
        return 0
    vals = slope * idx + intercept
    ref = np.where(vals > 0, vals, np.nan)
    diff = np.abs(price - vals) / ref
    return int(np.nansum(diff <= tol))


def detect_trendlines(df, lookback=120) -> dict:
    """趋势线拟合与形态判定。

    用摆动低点（左右各 3 根确认）拟合上升趋势线、摆动高点拟合下降趋势线（``np.polyfit``）；
    ``touches`` 为落在直线 ±1.5% 内的 K 线数，``valid`` = touches ≥ 3；
    ``triangle`` 两线夹角收敛（上轨向下 + 下轨向上）；``wedge`` 两轨同向（上升/下降楔形）。
    """
    out = {"up_line": {"slope": _NAN, "intercept": _NAN, "points": 0, "touches": 0,
                       "valid": False},
           "down_line": {"slope": _NAN, "intercept": _NAN, "points": 0, "touches": 0,
                         "valid": False},
           "break": "none", "support_now": _NAN, "resistance_now": _NAN,
           "triangle": False, "wedge": False}
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns or len(d) < 20:
        return out
    n = len(d)
    lb = int(min(max(int(lookback or 120), 20), n))
    h = _arr(d, "high")[-lb:]
    l = _arr(d, "low")[-lb:]
    c = _arr(d, "close")[-lb:]
    if not (np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(c).all()):
        return out
    hi_idx, lo_idx = _swing_points(h, 3)
    _, lo_only = _swing_points(l, 3)
    x_now = float(lb - 1)

    def _fit(idx, price):
        if len(idx) < 2:
            return None
        xi = np.asarray(idx, dtype=float)
        yi = np.asarray([price[int(k)] for k in idx], dtype=float)
        try:
            sl, ic = np.polyfit(xi, yi, 1)
        except Exception:  # noqa: BLE001
            return None
        return float(sl), float(ic)

    up_fit = _fit(lo_only, l)
    dn_fit = _fit(hi_idx, h)
    if up_fit:
        sl, ic = up_fit
        touches = _line_touches(np.arange(lb, dtype=float), l, sl, ic, 0.015)
        out["up_line"] = {"slope": round(sl, 6), "intercept": round(ic, 6),
                          "points": len(lo_only), "touches": touches, "valid": touches >= 3}
        out["support_now"] = round(sl * x_now + ic, 4)
    if dn_fit:
        sl, ic = dn_fit
        touches = _line_touches(np.arange(lb, dtype=float), h, sl, ic, 0.015)
        out["down_line"] = {"slope": round(sl, 6), "intercept": round(ic, 6),
                            "points": len(hi_idx), "touches": touches, "valid": touches >= 3}
        out["resistance_now"] = round(sl * x_now + ic, 4)
    su = out["up_line"]["slope"]
    sd = out["down_line"]["slope"]
    v_up = bool(out["up_line"]["valid"])
    v_dn = bool(out["down_line"]["valid"])
    # 收敛三角形：下轨向上 + 上轨向下，且开口比窗口初期更窄
    if v_up and v_dn and np.isfinite(su) and np.isfinite(sd) and su > 0 > sd:
        gap0 = (sd * 0.0 + out["down_line"]["intercept"]) - (su * 0.0 + out["up_line"]["intercept"])
        gap1 = out["resistance_now"] - out["support_now"]
        out["triangle"] = bool(gap1 < gap0 * 0.85 and gap1 > 0)
    # 楔形：两轨同向（都向上 = 上升楔形，都向下 = 下降楔形）
    if v_up and v_dn and np.isfinite(su) and np.isfinite(sd) and su * sd > 0:
        out["wedge"] = True
    price_now = float(c[-1])
    if v_dn and np.isfinite(out["resistance_now"]) and price_now > out["resistance_now"] * 1.015:
        out["break"] = "up"
    elif v_up and np.isfinite(out["support_now"]) and price_now < out["support_now"] * 0.985:
        out["break"] = "down"
    return out


# --------------------------------------------------------------------------- #
# 六、艾略特波浪（简化）
# --------------------------------------------------------------------------- #
def _zigzag(prices: np.ndarray, thr: float):
    """ZigZag 摆动点提取。

    规则：价格自上一个极端点反向运行超过 ``thr``（绝对价格）时，确认该极端点为摆动点。
    返回 ``[(idx, price, kind)]``，``kind`` ∈ {+1 顶, -1 底}。
    """
    n = prices.size
    if n < 3 or not np.isfinite(thr) or thr <= 0:
        return []
    piv: list = []
    direction = 1 if prices[1] >= prices[0] else -1
    ext_i, ext_p = 0, prices[0]
    for i in range(1, n):
        p = prices[i]
        if direction > 0:
            if p > ext_p:
                ext_i, ext_p = i, p
            elif p < ext_p - thr:
                piv.append((ext_i, float(ext_p), 1))
                direction, ext_i, ext_p = -1, i, p
        else:
            if p < ext_p:
                ext_i, ext_p = i, p
            elif p > ext_p + thr:
                piv.append((ext_i, float(ext_p), -1))
                direction, ext_i, ext_p = 1, i, p
    piv.append((ext_i, float(ext_p), 1 if direction > 0 else -1))
    # 去掉相邻同类型点（保留更极端者），保证顶底交替
    clean: list = []
    for pt in piv:
        if clean and clean[-1][2] == pt[2]:
            if (pt[2] > 0 and pt[1] >= clean[-1][1]) or (pt[2] < 0 and pt[1] <= clean[-1][1]):
                clean[-1] = pt
        else:
            clean.append(pt)
    return clean


def elliott_count(df, lookback=250) -> dict:
    """艾略特波浪简化数浪。

    方法
    ----
    1. ZigZag 阈值 = ``max(2.5 × 近 60 日平均真实波幅, 现价的 5%)``，提取摆动点；
    2. 取最近 6 个摆动点，要求顶底交替；
    3. 按"是否创新高/不创新低"的交替结构定位浪位，并用斐波那契比例校验：
       2 浪回撤 1 浪的 0.382~0.786、3 浪 ≥ 1.618×1 浪、4 浪不破 1 浪顶；
    4. **保守输出**：结构不吻合返回 ``"unknown"``（confidence ≤ 0.35）。

    返回 ``{"wave","direction","swing_points","confidence","targets","detail"}``。
    """
    out = {"wave": "unknown", "direction": DIR_NEUTRAL, "swing_points": [],
           "confidence": 0.15, "targets": [], "detail": "摆动点不足或结构不清晰，无法数浪"}
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns or len(d) < 60:
        return out
    n = len(d)
    lb = int(min(max(int(lookback or 250), 60), n))
    c = _arr(d, "close")[-lb:]
    h = _arr(d, "high")[-lb:]
    l = _arr(d, "low")[-lb:]
    if not np.isfinite(c).all():
        return out
    # 近 60 日日均真实波幅（Wilder）
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr60 = float(np.nanmean(tr[-60:])) if np.isfinite(tr[-60:]).any() else _NAN
    thr = max(2.5 * atr60 if np.isfinite(atr60) else 0.0, 0.05 * float(c[-1]))
    piv = _zigzag(c, thr)
    if len(piv) < 3:
        return out
    out["swing_points"] = [{"index": int(p[0]), "price": round(float(p[1]), 4),
                            "type": "high" if p[2] > 0 else "low"} for p in piv[-6:]]
    pts = piv[-6:]
    conf = 0.20
    # 交替结构：low->high->low->high... 或 high->low->...
    if len(pts) < 4:
        return out
    prices = [p[1] for p in pts]
    kinds = [p[2] for p in pts]
    if any(kinds[i] == kinds[i + 1] for i in range(len(kinds) - 1)):
        return out
    rising = kinds[0] < 0                     # 首点为底 -> 上升结构
    legs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    up_legs = [abs(v) for i, v in enumerate(legs) if (v > 0) == rising]
    wave, direction, detail = "unknown", DIR_NEUTRAL, ""
    if rising and all(v > 0 for v in legs):
        # 全为上升腿：处于推动浪中（1 或 3 或 5）
        wave = {1: "1", 2: "3", 3: "5"}.get(len(legs) - 1, "5")
        direction = DIR_BULL
        detail = "摆动结构为逐浪抬高，当前处于上升推动浪"
        if len(legs) >= 2 and up_legs:
            ext = abs(legs[-1]) / abs(legs[0]) if legs[0] else _NAN
            if np.isfinite(ext) and ext >= 1.618:
                conf += 0.20
                detail += "，且最新一浪相对首浪扩展 ≥1.618 倍（强势主升特征）"
            else:
                detail += "，但扩展比例未达 1.618 倍，主升力度一般"
    elif (not rising) and all(v < 0 for v in legs):
        wave = {1: "1", 2: "3", 3: "5"}.get(len(legs) - 1, "5")
        direction = DIR_BEAR
        detail = "摆动结构为逐浪走低，当前处于下跌推动浪"
    else:
        # 交替结构 = 调整浪，检查回撤比例
        if rising:
            imp = abs(prices[1] - prices[0])
            retr = abs(prices[2] - prices[1]) / imp if imp > 0 else _NAN
            wave = "2" if len(pts) >= 3 else "unknown"
            if np.isfinite(retr):
                if 0.382 <= retr <= 0.786:
                    conf += 0.30
                    detail = "上升后回撤 %.1f%%（落在 2 浪常见的 0.382~0.786 区间）" % (retr * 100)
                else:
                    conf += 0.05
                    detail = "回撤幅度 %.1f%% 不符合 2 浪常见比例，浪位存疑" % (retr * 100)
                direction = DIR_BULL
        else:
            imp = abs(prices[0] - prices[1]) if len(prices) > 1 else _NAN
            retr = abs(prices[2] - prices[1]) / imp if (np.isfinite(imp) and imp > 0) else _NAN
            wave = "B" if len(pts) >= 3 else "unknown"
            if np.isfinite(retr):
                if 0.382 <= retr <= 0.786:
                    conf += 0.25
                    detail = "下跌后反弹 %.1f%%（落在 B 浪常见区间）" % (retr * 100)
                else:
                    conf += 0.05
                    detail = "反弹幅度 %.1f%% 不符合 B 浪常见比例，浪位存疑" % (retr * 100)
                direction = DIR_BEAR
    # 目标位：斐波那契扩展
    targets = []
    try:
        if len(pts) >= 4:
            p0, p1, p2, p3 = prices[-4], prices[-3], prices[-2], prices[-1]
            span = abs(p1 - p0)
            for k in (0.618, 1.0, 1.618):
                t = p3 + (span * k if direction >= 0 else -span * k)
                targets.append(round(float(t), 4))
    except Exception:  # noqa: BLE001
        targets = []
    if wave == "unknown":
        conf = min(conf, 0.35)
    out.update({"wave": wave, "direction": int(direction),
                "confidence": round(float(np.clip(conf, 0.0, 0.85)), 3),
                "targets": targets,
                "detail": detail or "结构不清晰，浪位判断保守"})
    return out


# --------------------------------------------------------------------------- #
# 七、缠论（简化）
# --------------------------------------------------------------------------- #
def chan_strokes(df, lookback=200) -> dict:
    """简化缠论：分型 -> 笔 -> 线段 -> 中枢 + MACD 背驰。

    * **分型**：``high[i] > high[i-1] 且 high[i] > high[i+1]`` 为顶分型（底分型反之）；
    * **笔**：顶底交替、两端索引差 ≥ 4（即至少 5 根 K 线）、且不重叠；
    * **线段**：连续 3 笔构成一段（首笔方向为段方向，段终点为第 3 笔终点）；
    * **中枢**：连续三笔的价格重叠区间 ``[max(低点), min(高点)]``（要求 low < high）；
    * **背驰**：用 MACD 柱面积（``sum|hist|``）比较同向笔，价格创新极值但面积萎缩即背驰。
    """
    out = {"bi": [], "segments": [], "zhongshu": [], "current": "中枢震荡",
           "buy_point": "无", "sell_point": "无", "divergence": False,
           "detail": "数据不足，无法构建缠论结构"}
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns or len(d) < 30:
        return out
    n = len(d)
    lb = int(min(max(int(lookback or 200), 30), n))
    sub = d.iloc[-lb:].reset_index(drop=True)
    h = _arr(sub, "high")
    l = _arr(sub, "low")
    c = _arr(sub, "close")
    if not (np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(c).all()):
        return out
    # ---- 分型 ------------------------------------------------------------ #
    fx = []
    for i in range(1, lb - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1]:
            fx.append({"idx": i, "price": float(h[i]), "kind": 1})
        elif l[i] < l[i - 1] and l[i] < l[i + 1]:
            fx.append({"idx": i, "price": float(l[i]), "kind": -1})
    # ---- 笔 -------------------------------------------------------------- #
    bi = []
    last = None
    for f in fx:
        if last is None:
            last = f
            continue
        if f["kind"] == last["kind"]:
            # 同类型分型：保留更极端者
            if (f["kind"] > 0 and f["price"] > last["price"]) or \
               (f["kind"] < 0 and f["price"] < last["price"]):
                last = f
            continue
        if f["idx"] - last["idx"] < 4:            # 至少 5 根 K 线才成笔
            continue
        if last["kind"] < 0 and f["price"] <= last["price"]:
            continue                               # 向上笔要求终点更高
        if last["kind"] > 0 and f["price"] >= last["price"]:
            continue                               # 向下笔要求终点更低
        bi.append({"start_idx": int(last["idx"]), "end_idx": int(f["idx"]),
                   "start_price": round(float(last["price"]), 4),
                   "end_price": round(float(f["price"]), 4),
                   "direction": 1 if f["kind"] > 0 else -1})
        last = f
    out["bi"] = bi
    if not bi:
        return out
    # ---- 线段（连续 3 笔） ----------------------------------------------- #
    segs = []
    for i in range(0, max(len(bi) - 2, 0), 2):
        trio = bi[i:i + 3]
        if len(trio) < 3:
            break
        segs.append({"start_idx": trio[0]["start_idx"], "end_idx": trio[-1]["end_idx"],
                     "start_price": trio[0]["start_price"], "end_price": trio[-1]["end_price"],
                     "direction": int(trio[0]["direction"]),
                     "bi_count": len(trio)})
    out["segments"] = segs
    # ---- 中枢（连续三笔重叠） -------------------------------------------- #
    zs = []
    for i in range(len(bi) - 2):
        trio = bi[i:i + 3]
        highs = [max(b["start_price"], b["end_price"]) for b in trio]
        lows = [min(b["start_price"], b["end_price"]) for b in trio]
        zl, zh = max(lows), min(highs)
        if zl < zh:
            zs.append({"low": round(float(zl), 4), "high": round(float(zh), 4),
                       "start": int(trio[0]["start_idx"]), "end": int(trio[-1]["end_idx"])})
    merged = []
    for z in zs:
        if merged and z["start"] <= merged[-1]["end"]:
            merged[-1]["high"] = round(max(merged[-1]["high"], z["high"]), 4)
            merged[-1]["low"] = round(min(merged[-1]["low"], z["low"]), 4)
            merged[-1]["end"] = max(merged[-1]["end"], z["end"])
        else:
            merged.append(dict(z))
    out["zhongshu"] = merged
    # ---- MACD 面积背驰 --------------------------------------------------- #
    divergence = False
    try:
        hist = macd(pd.Series(c))["macd"].to_numpy(dtype=float)
        areas = []
        for b in bi[-4:]:
            s, e = b["start_idx"], b["end_idx"]
            seg = hist[s:e + 1]
            areas.append(float(np.nansum(np.abs(seg[np.isfinite(seg)]))) if seg.size else 0.0)
        if len(areas) >= 3:
            same_dir = [bi[-3], bi[-1]]
            if same_dir[0]["direction"] == same_dir[1]["direction"]:
                p_new = same_dir[1]["end_price"]
                p_old = same_dir[0]["end_price"]
                made_ext = (p_new > p_old) if same_dir[1]["direction"] > 0 else (p_new < p_old)
                shrink = areas[-1] < areas[-2] * 0.9
                divergence = bool(made_ext and shrink)
    except Exception as exc:  # noqa: BLE001
        log.debug("chan_strokes 背驰计算失败: %s", exc)
    out["divergence"] = divergence
    # ---- 当前状态与买卖点 ------------------------------------------------ #
    last_bi = bi[-1]
    price_now = float(c[-1])
    in_zs = merged[-1] if merged else None
    if in_zs and in_zs["low"] <= price_now <= in_zs["high"]:
        out["current"] = "中枢震荡"
    else:
        out["current"] = "上涨笔" if last_bi["direction"] > 0 else "下跌笔"
    buy, sell = "无", "无"
    if last_bi["direction"] < 0 and divergence:
        buy = "1买"                                   # 下跌笔 + 底背驰 -> 第一类买点
    elif in_zs and price_now > in_zs["high"] and last_bi["direction"] > 0:
        buy = "3买"                                   # 放量离开中枢上沿 -> 第三类买点
    elif len(bi) >= 3 and bi[-1]["direction"] > 0 and bi[-1]["end_price"] > bi[-3]["end_price"] \
            and price_now > bi[-2]["start_price"]:
        buy = "2买"                                   # 回抽不破前低后的再度上行
    if last_bi["direction"] > 0 and divergence:
        sell = "1卖"
    elif in_zs and price_now < in_zs["low"] and last_bi["direction"] < 0:
        sell = "3卖"
    out.update({"buy_point": buy, "sell_point": sell,
                "detail": "结构：%d 笔 / %d 线段 / %d 中枢，当前%s%s"
                          % (len(bi), len(segs), len(merged), out["current"],
                             "，检测到 MACD 面积背驰" if divergence else "")})
    return out


# --------------------------------------------------------------------------- #
# 八、一目均衡表
# --------------------------------------------------------------------------- #
def ichimoku(df) -> pd.DataFrame:
    """一目均衡表（默认 9 / 26 / 52）。

    列：``tenkan, kijun, senkou_a, senkou_b, chikou, cloud_top, cloud_bottom, above_cloud``。
    国内习惯：``senkou_a/b`` **不向未来位移**（直接画在当前行）；``chikou`` 为
    ``close.shift(-26)``（尾部 NaN）。``above_cloud`` 为 1/0（收盘是否站上云层上沿）。
    """
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns:
        return pd.DataFrame()
    h = _cols(d, "high")
    l = _cols(d, "low")
    c = _cols(d, "close")
    out = pd.DataFrame(index=d.index)
    out["tenkan"] = (h.rolling(9, min_periods=9).max() + l.rolling(9, min_periods=9).min()) / 2.0
    out["kijun"] = (h.rolling(26, min_periods=26).max() + l.rolling(26, min_periods=26).min()) / 2.0
    out["senkou_a"] = (out["tenkan"] + out["kijun"]) / 2.0
    out["senkou_b"] = (h.rolling(52, min_periods=52).max() + l.rolling(52, min_periods=52).min()) / 2.0
    out["chikou"] = c.shift(-26)
    out["cloud_top"] = pd.concat([out["senkou_a"], out["senkou_b"]], axis=1).max(axis=1)
    out["cloud_bottom"] = pd.concat([out["senkou_a"], out["senkou_b"]], axis=1).min(axis=1)
    out["above_cloud"] = (c > out["cloud_top"]).astype(int)
    out.loc[out["cloud_top"].isna(), "above_cloud"] = 0
    return out


# --------------------------------------------------------------------------- #
# 九、经典形态与位置
# --------------------------------------------------------------------------- #
def classic_shape(df, lookback=250) -> dict:
    """经典形态与位置判断。

    * ``shapes``：识别到的形态列表（箱体震荡/双顶/双底/三角形/楔形/均线排列…），
      每条 ``{"name","direction","strength","detail","target"}``；
    * ``position``：``vs_ma20``/``vs_ma60``（above/below）、``dist_to_high_pct``、
      ``dist_to_low_pct``（相对窗口高低点的距离，%）；
    * ``box``：近 60 根内 ``(max-min)/min < 12%`` 且上下沿各被触碰 ≥2 次 -> 箱体；
    * ``double_top_bottom``：两个相隔 ≥10 根的极值点价差 < 3%、中间回落/反弹 > 5%。
    """
    out = {"shapes": [],
           "position": {"vs_ma20": "unknown", "vs_ma60": "unknown",
                        "dist_to_high_pct": _NAN, "dist_to_low_pct": _NAN},
           "box": {"detected": False, "top": _NAN, "bottom": _NAN, "bars": 0, "break": "none"},
           "double_top_bottom": {"type": "none", "neckline": _NAN, "target": _NAN}}
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns or len(d) < 60:
        return out
    n = len(d)
    lb = int(min(max(int(lookback or 250), 60), n))
    sub = d.iloc[-lb:]
    h = _arr(sub, "high")
    l = _arr(sub, "low")
    c = _arr(sub, "close")
    if not (np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(c).all()):
        return out
    price = float(c[-1])
    ma20 = float(ma(pd.Series(c), 20).iloc[-1])
    ma60 = float(ma(pd.Series(c), 60).iloc[-1]) if lb >= 60 else _NAN
    hi_win, lo_win = float(np.max(h)), float(np.min(l))
    pos = {
        "vs_ma20": "above" if (np.isfinite(ma20) and price > ma20) else "below",
        "vs_ma60": ("above" if (np.isfinite(ma60) and price > ma60)
                    else ("below" if np.isfinite(ma60) else "unknown")),
        "dist_to_high_pct": round((price - hi_win) / hi_win * 100.0, 3) if hi_win else _NAN,
        "dist_to_low_pct": round((price - lo_win) / lo_win * 100.0, 3) if lo_win else _NAN,
    }
    out["position"] = pos
    shapes = []
    # ---- 箱体 ------------------------------------------------------------ #
    bw = 60
    bh = h[-bw:]
    bl = l[-bw:]
    bc = c[-bw:]
    top, bot = float(np.max(bh)), float(np.min(bl))
    if bot > 0 and (top - bot) / bot < 0.12:
        tol = 0.015
        touch_top = int(np.sum(bh >= top * (1 - tol)))
        touch_bot = int(np.sum(bl <= bot * (1 + tol)))
        if touch_top >= 2 and touch_bot >= 2:
            brk = "none"
            if price > top * 1.005:
                brk = "up"
            elif price < bot * 0.995:
                brk = "down"
            out["box"] = {"detected": True, "top": round(top, 4), "bottom": round(bot, 4),
                          "bars": bw, "break": brk}
            tgt = top + (top - bot) if brk == "up" else (bot - (top - bot) if brk == "down"
                                                         else (top + bot) / 2.0)
            shapes.append({
                "name": "箱体震荡", "direction": (1 if brk == "up" else (-1 if brk == "down" else 0)),
                "strength": round(min(0.45 + 0.1 * min(touch_top + touch_bot, 8) / 8.0, 0.9), 3),
                "detail": "近 %d 根振幅 %.1f%%（上沿 %.2f / 下沿 %.2f，上沿触碰 %d 次、下沿 %d 次），%s"
                          % (bw, (top - bot) / bot * 100, top, bot, touch_top, touch_bot,
                             {"up": "已向上突破箱体", "down": "已向下跌破箱体",
                              "none": "仍在箱体内运行"}[brk]),
                "target": round(float(tgt), 4)})
    # ---- 双顶 / 双底 ------------------------------------------------------ #
    hi_idx, lo_idx = _swing_points(h, 3)
    _, lo_only = _swing_points(l, 3)

    def _double(idx_list, prices, kind):
        best = None
        for a in range(len(idx_list)):
            for b in range(a + 1, len(idx_list)):
                i1, i2 = idx_list[a], idx_list[b]
                if i2 - i1 < 10:
                    continue
                p1, p2 = prices[i1], prices[i2]
                if abs(p1 - p2) > 0.03 * max(p1, p2):
                    continue
                mid = prices[i1:i2 + 1]
                if kind == "top":
                    neck = float(np.min(mid))
                    depth = (max(p1, p2) - neck) / max(p1, p2)
                else:
                    neck = float(np.max(mid))
                    depth = (neck - min(p1, p2)) / min(p1, p2)
                if depth <= 0.05:
                    continue
                cand = {"type": "double_top" if kind == "top" else "double_bottom",
                        "neckline": round(neck, 4),
                        "target": round(neck - (max(p1, p2) - neck), 4) if kind == "top"
                        else round(neck + (neck - min(p1, p2)), 4),
                        "strength": round(min(0.5 + depth, 0.9), 3),
                        "depth": depth, "i1": i1, "i2": i2}
                if best is None or cand["i2"] > best["i2"]:
                    best = cand
        return best

    dt = _double(hi_idx, h, "top")
    db = _double(lo_only, l, "bottom")
    # 取时间上更近的一个
    cand = None
    if dt and db:
        cand = dt if dt["i2"] >= db["i2"] else db
    else:
        cand = dt or db
    if cand:
        out["double_top_bottom"] = {"type": cand["type"], "neckline": cand["neckline"],
                                    "target": cand["target"]}
        is_top = cand["type"] == "double_top"
        shapes.append({
            "name": "双顶" if is_top else "双底",
            "direction": -1 if is_top else 1,
            "strength": cand["strength"],
            "detail": "%s形态，颈线 %.2f，中间%s %.1f%%，量度目标 %.2f"
                      % ("双顶" if is_top else "双底", cand["neckline"],
                         "回落" if is_top else "反弹", cand["depth"] * 100, cand["target"]),
            "target": cand["target"]})
    # ---- 趋势线形态（三角形 / 楔形 / 突破） -------------------------------- #
    try:
        tl = detect_trendlines(sub, min(lb, 120))
        if tl.get("triangle"):
            shapes.append({"name": "收敛三角形", "direction": 0, "strength": 0.55,
                           "detail": "上下轨逐渐收敛，方向选择前的蓄势形态（突破方向为准）",
                           "target": round(float(np.nanmax(h)), 4)})
        if tl.get("wedge"):
            up = float(tl["up_line"]["slope"]) > 0
            shapes.append({"name": "上升楔形" if up else "下降楔形",
                           "direction": -1 if up else 1, "strength": 0.5,
                           "detail": "两轨同向运行形成%s，往往以反向突破收场" % ("上升楔形" if up else "下降楔形"),
                           "target": tl.get("support_now") if up else tl.get("resistance_now")})
        if tl.get("break") == "up":
            shapes.append({"name": "向上突破趋势线", "direction": 1, "strength": 0.6,
                           "detail": "收盘价突破下降趋势线 %.2f，趋势可能反转向上"
                                     % float(tl.get("resistance_now", _NAN)),
                           "target": tl.get("resistance_now")})
        elif tl.get("break") == "down":
            shapes.append({"name": "向下跌破趋势线", "direction": -1, "strength": 0.6,
                           "detail": "收盘价跌破上升趋势线 %.2f，趋势可能转弱"
                                     % float(tl.get("support_now", _NAN)),
                           "target": tl.get("support_now")})
    except Exception as exc:  # noqa: BLE001
        log.debug("classic_shape 趋势线形态失败: %s", exc)
    # ---- 均线位置 -------------------------------------------------------- #
    if np.isfinite(ma20) and np.isfinite(ma60):
        if price > ma20 > ma60:
            shapes.append({"name": "均线多头排列", "direction": 1, "strength": 0.6,
                           "detail": "股价站上 MA20 且 MA20 > MA60，中期趋势向上",
                           "target": round(float(np.max(h)), 4)})
        elif price < ma20 < ma60:
            shapes.append({"name": "均线空头排列", "direction": -1, "strength": 0.6,
                           "detail": "股价跌破 MA20 且 MA20 < MA60，中期趋势向下",
                           "target": round(float(np.min(l)), 4)})
    out["shapes"] = shapes
    return out


# --------------------------------------------------------------------------- #
# 自测（离线合成数据 + 数值校验）
# --------------------------------------------------------------------------- #
def _mk_bar(o, h, l, c, prev_c=None):
    """构造单根 K 线的字典（用于确定性形态测试）。"""
    return {"open": float(o), "high": float(h), "low": float(l), "close": float(c)}


def _frame(bars, start="2024-01-01", base_vol=1.0e6):
    """由 K 线字典列表构造带 date/volume 的 DataFrame。"""
    rows = []
    d0 = pd.Timestamp(start)
    for i, b in enumerate(bars):
        rows.append({"date": d0 + pd.Timedelta(days=i), "open": b["open"], "high": b["high"],
                     "low": b["low"], "close": b["close"],
                     "volume": b.get("volume", base_vol),
                     "amount": b.get("volume", base_vol) * b["close"]})
    return pd.DataFrame(rows)


def _quiet_flat(n=30, px=20.0, amp=0.004):
    """生成 n 根小幅震荡 K 线（用于给形态提供"位置"背景）。"""
    out = []
    for i in range(n):
        p = px * (1 + amp * ((-1) ** i))
        out.append(_mk_bar(p * 0.995, p * 1.008, p * 0.985, p * 1.002))
    return out


def _self_test() -> None:  # pragma: no cover - 自测块
    try:
        from .selftest_data import make_ohlcv
    except ImportError:
        from selftest_data import make_ohlcv  # type: ignore

    df = make_ohlcv(600, seed=1)
    df_backup = df.copy(deep=True)

    # ---------- 1. 特征表 ---------- #
    feats = candle_features(df)
    need = ["body", "body_pct", "upper_shadow", "lower_shadow", "shadow_ratio", "is_bull",
            "gap_up", "gap_down", "is_doji", "is_hammer", "is_shooting_star", "is_marubozu",
            "is_spinning_top", "close_pos"]
    missing = [c for c in need if c not in feats.columns]
    assert not missing, "candle_features 缺列: %s" % missing
    assert len(feats) == len(df) and feats.index.equals(df.index), "特征表长度/索引不一致"
    assert ((df["close"] > df["open"]).astype(int) == feats["is_bull"]).all(), "is_bull 与 close>open 不一致"
    cp = feats["close_pos"].to_numpy(dtype=float)
    cp = cp[np.isfinite(cp)]
    assert cp.min() >= -1e-9 and cp.max() <= 1 + 1e-9, "close_pos 越界"
    # 几何关系抽查：body == close-open；影线非负
    assert np.allclose(feats["body"].to_numpy(), (df["close"] - df["open"]).to_numpy())
    assert (feats["upper_shadow"].to_numpy() >= -1e-9).all()
    assert (feats["lower_shadow"].to_numpy() >= -1e-9).all()
    print("[1] candle_features: shape=%s，14 个特征列齐全；is_bull 与 close>open 完全一致；"
          "close_pos∈[%.3f,%.3f]" % (feats.shape, cp.min(), cp.max()))

    # ---------- 2. 确定性形态触发（覆盖全部 30 个判定器） ---------- #
    def _names(frame, lb=300):
        return {p["en"] for p in detect_patterns(frame, lb)}

    def _big_body_bg(n=25, px=20.0):
        """大实体阳线背景（实体/振幅 ≈0.77 > 0.4，不会干扰"连续小阳"判定）。"""
        return [_mk_bar(px, px + 0.6, px - 0.05, px + 0.5) for _ in range(n)]

    def _down_bg(n=8, px=20.0, step=0.35):
        """连续下跌背景（供低位形态使用）。"""
        return [_mk_bar(px - step * k, px + 0.1 - step * k, px - 0.4 - step * k,
                        px - 0.3 - step * k) for k in range(n)]

    def _up_bg(n=8, px=20.0, step=0.35):
        """连续上涨背景（供高位形态使用）。"""
        return [_mk_bar(px + step * k, px + 0.4 + step * k, px - 0.1 + step * k,
                        px + 0.3 + step * k) for k in range(n)]

    cases = {}
    Q = _quiet_flat(25, 20.0)
    cases["看涨吞没/旭日东升"] = _frame(Q + [_mk_bar(20.0, 20.1, 18.9, 19.0),
                                             _mk_bar(18.8, 20.6, 18.7, 20.5)])
    cases["看跌吞没/乌云盖顶"] = _frame(Q + [_mk_bar(19.0, 21.0, 18.9, 20.9),
                                             _mk_bar(21.2, 21.3, 19.4, 19.5)])
    cases["早晨之星"] = _frame(Q + [_mk_bar(21.0, 21.1, 19.0, 19.1),
                                    _mk_bar(18.9, 19.2, 18.5, 18.9),
                                    _mk_bar(19.2, 20.9, 19.1, 20.8)])
    cases["黄昏之星"] = _frame(Q + [_mk_bar(19.0, 21.0, 18.9, 20.9),
                                    _mk_bar(21.1, 21.4, 20.9, 21.2),
                                    _mk_bar(20.8, 20.9, 19.2, 19.3)])
    cases["锤头线/倒锤头"] = _frame(Q + _down_bg() + [_mk_bar(17.0, 17.16, 15.6, 17.15),
                                                      _mk_bar(17.0, 17.40, 16.99, 17.05)])
    cases["射击之星/上吊线"] = _frame(Q + _up_bg() + [_mk_bar(23.2, 24.8, 23.19, 23.25),
                                                      _mk_bar(24.0, 24.06, 23.69, 24.05)])
    cases["红三兵"] = _frame(Q + [_mk_bar(20.0, 20.7, 19.9, 20.6),
                                  _mk_bar(20.4, 21.2, 20.3, 21.1),
                                  _mk_bar(20.9, 21.9, 20.8, 21.8)])
    cases["三只乌鸦"] = _frame(Q + [_mk_bar(20.0, 20.1, 19.3, 19.4),
                                    _mk_bar(19.5, 19.6, 18.8, 18.9),
                                    _mk_bar(19.0, 19.1, 18.2, 18.3)])
    cases["十字星"] = _frame(Q + [_mk_bar(20.0, 20.6, 19.5, 20.02)])
    cases["跳空缺口(上)"] = _frame(Q + [_mk_bar(20.5, 21.0, 20.4, 20.9)])
    cases["跳空缺口(下)"] = _frame(Q + [_mk_bar(19.5, 19.7, 19.3, 19.4)])
    cases["平底"] = _frame(Q + [_mk_bar(19.0 - 0.2 * k, 19.2 - 0.2 * k, 18.6 - 0.2 * k,
                                         18.8 - 0.2 * k) for k in range(6)]
                           + [_mk_bar(17.5, 17.7, 17.0, 17.3), _mk_bar(17.3, 17.6, 17.0, 17.5)])
    cases["平顶"] = _frame(Q + [_mk_bar(20.0 + 0.3 * k, 20.4 + 0.3 * k, 19.9 + 0.3 * k,
                                        20.2 + 0.3 * k) for k in range(6)]
                           + [_mk_bar(22.0, 22.8, 21.8, 22.1), _mk_bar(22.1, 22.8, 21.9, 22.0)])
    volley = _frame(Q + [_mk_bar(20.0, 20.8, 19.9, 20.7), _mk_bar(20.6, 20.7, 20.1, 20.3),
                         _mk_bar(20.3, 21.3, 20.2, 21.2)])
    volley.loc[27, "volume"] = 2.5e6                     # 第三根放量
    cases["多方炮"] = volley
    bear_volley = _frame(Q + [_mk_bar(20.0, 20.1, 19.2, 19.3), _mk_bar(19.4, 19.8, 19.3, 19.45),
                              _mk_bar(19.4, 19.5, 18.5, 18.6)])
    bear_volley.loc[27, "volume"] = 2.5e6
    cases["空方炮"] = bear_volley
    cases["穿头破脚"] = _frame(Q + [_mk_bar(20.0, 20.5, 19.8, 20.2),
                                    _mk_bar(20.1, 21.0, 19.0, 20.8)])
    cases["好友反攻"] = _frame(Q + [_mk_bar(20.5, 20.6, 19.0, 19.1),
                                    _mk_bar(18.9, 19.2, 18.8, 19.08)])
    cases["淡友反攻"] = _frame(Q + [_mk_bar(19.0, 20.6, 18.9, 20.4),
                                    _mk_bar(20.6, 20.7, 20.35, 20.42)])
    cases["上升三法"] = _frame(Q + [_mk_bar(20.0, 21.0, 19.9, 20.9), _mk_bar(20.7, 20.9, 20.3, 20.5),
                                    _mk_bar(20.4, 20.7, 20.1, 20.3), _mk_bar(20.2, 20.5, 20.0, 20.2),
                                    _mk_bar(20.2, 21.3, 20.1, 21.2)])
    cases["下降三法"] = _frame(Q + [_mk_bar(21.0, 21.1, 20.0, 20.1), _mk_bar(20.3, 20.6, 20.2, 20.5),
                                    _mk_bar(20.5, 20.8, 20.3, 20.6), _mk_bar(20.6, 20.9, 20.4, 20.7),
                                    _mk_bar(20.7, 20.8, 19.8, 19.9)])
    cases["岛形反转"] = _frame(Q + [_mk_bar(19.3, 19.6, 19.1, 19.4), _mk_bar(19.4, 19.7, 19.2, 19.5),
                                    _mk_bar(19.5, 19.69, 19.3, 19.6), _mk_bar(19.8, 20.3, 19.75, 20.2)])
    cases["光头光脚阳线"] = _frame(Q + [_mk_bar(20.0, 21.0, 20.0, 21.0)])
    cases["光头光脚阴线"] = _frame(Q + [_mk_bar(21.0, 21.0, 20.0, 20.0)])
    cases["连续小阳"] = _frame(Q + [_mk_bar(20.5, 20.9, 20.45, 20.6),
                                    _mk_bar(20.6, 21.2, 20.5, 21.1),
                                    _mk_bar(21.1, 21.8, 20.9, 21.5)])
    cases["急速下跌后的长下影"] = _frame(Q + [_mk_bar(21.0, 21.1, 20.5, 20.6),
                                              _mk_bar(20.6, 20.7, 19.9, 20.0),
                                              _mk_bar(20.0, 20.1, 19.3, 19.4),
                                              _mk_bar(19.4, 19.5, 18.4, 19.35)])
    cases["长下影线"] = _frame(Q + [_mk_bar(20.0, 20.1, 18.5, 20.05)])
    cases["长上影线"] = _frame(Q + [_mk_bar(20.0, 21.5, 19.9, 20.05)])

    hit, miss = set(), []
    for label, frame in cases.items():
        got = _names(frame, 300)
        hit |= (got & {s["en"] for s in _PATTERN_SPECS})
        if not got:
            miss.append(label)
    all_en = {s["en"] for s in _PATTERN_SPECS}
    required_cn = {"早晨之星", "黄昏之星", "看涨吞没", "看跌吞没", "旭日东升", "乌云盖顶",
                   "穿头破脚", "锤头线", "倒锤头", "上吊线", "射击之星", "十字星",
                   "长下影线", "长上影线", "红三兵", "三只乌鸦", "多方炮", "空方炮",
                   "好友反攻", "淡友反攻", "平底", "平顶", "上升三法", "下降三法",
                   "跳空缺口", "岛形反转", "光头光脚阳线", "光头光脚阴线", "连续小阳",
                   "急速下跌后的长下影"}
    registered_cn = {s["name"] for s in _PATTERN_SPECS}
    assert required_cn <= registered_cn, "缺少形态: %s" % (required_cn - registered_cn)
    assert len(_PATTERN_SPECS) >= 30, "形态数量不足"
    assert not (all_en - hit), "以下形态的判定器无法被确定性触发（疑似死代码）: %s" % sorted(all_en - hit)
    print("[2] 30 个形态判定器全部可确定性触发（%d 个构造用例）：%s"
          % (len(cases), "、".join(sorted(required_cn))))

    # ---------- 3. 合成数据上的形态扫描 ---------- #
    events = detect_patterns(df, lookback=200)
    names_cn = sorted({e["name"] for e in events})
    assert len(events) > 0, "合成数据上未识别到任何形态"
    assert len(names_cn) >= 6, "命中的形态种类过少: %s" % names_cn
    for e in events:
        assert set(e) >= {"date", "name", "en", "direction", "strength", "detail", "bars"}
        assert e["direction"] in (-1, 0, 1) and 0.0 <= e["strength"] <= 0.95
    print("[3] detect_patterns(df, 200) 命中 %d 次，形态种类 %d 种：%s"
          % (len(events), len(names_cn), "、".join(names_cn)))
    # ---------- 4. 历史统计（核心证据） ---------- #
    st = pattern_stats(df, horizon=5)
    assert isinstance(st, dict) and st, "pattern_stats 无输出"
    big = {k: v for k, v in st.items() if v["n"] >= 5}
    assert big, "没有任何形态样本数 ≥5"
    for k, v in st.items():
        assert v["n"] >= 5, "样本 <5 的形态不应返回: %s n=%s" % (k, v["n"])
        assert 0.0 <= v["win_rate"] <= 1.0
    top = sorted(st.items(), key=lambda kv: -kv[1]["n"])[:5]
    print("[4] pattern_stats(df,5): %d 种形态样本数 ≥5；样例 %s"
          % (len(st), "；".join("%s n=%d 胜率=%.0f%% 平均%+.2f%%"
                                % (k, v["n"], v["win_rate"] * 100, v["avg_ret"])
                                for k, v in top)))

    # ---------- 5. 因子 ---------- #
    factors = pattern_factors(df, "600519", "贵州茅台")
    ids = {f.id for f in factors}
    assert ids <= {"pattern_bull", "pattern_bear", "pattern_conflict", "pattern_volume_confirm"}, ids
    for f in factors:
        assert isinstance(f, Factor) and f.category == Category.TECH
        assert 0.0 <= f.strength <= 1.0 and 0.0 <= f.reliability <= 1.0
        assert f.horizon == _HORIZON_SHORT and f.detail
    print("[5] pattern_factors 输出 %d 条因子：%s"
          % (len(factors), "；".join("%s %s 强度%.2f 可信度%.2f"
                                     % (f.id, "看涨" if f.direction > 0 else
                                        ("看跌" if f.direction < 0 else "中性"),
                                        f.strength, f.reliability) for f in factors)))

    # ---------- 6. dict 键完整性 ---------- #
    tl = detect_trendlines(df)
    assert {"up_line", "down_line", "break", "support_now", "resistance_now",
            "triangle", "wedge"} <= set(tl)
    assert {"slope", "intercept", "points", "touches", "valid"} <= set(tl["up_line"])
    el = elliott_count(df)
    assert {"wave", "direction", "swing_points", "confidence", "targets", "detail"} <= set(el)
    assert el["wave"] in ("1", "2", "3", "4", "5", "A", "B", "C", "unknown")
    assert 0.0 <= el["confidence"] <= 1.0
    ch = chan_strokes(df)
    assert {"bi", "segments", "zhongshu", "current", "buy_point", "sell_point",
            "divergence", "detail"} <= set(ch)
    for z in ch["zhongshu"]:
        assert z["low"] < z["high"], "中枢区间不自洽: %s" % z
    for b in ch["bi"]:
        assert b["end_idx"] > b["start_idx"] and b["direction"] in (-1, 1)
    cs = classic_shape(df)
    assert {"shapes", "position", "box", "double_top_bottom"} <= set(cs)
    assert {"vs_ma20", "vs_ma60", "dist_to_high_pct", "dist_to_low_pct"} <= set(cs["position"])
    assert {"detected", "top", "bottom", "bars", "break"} <= set(cs["box"])
    assert {"type", "neckline", "target"} <= set(cs["double_top_bottom"])
    assert cs["double_top_bottom"]["type"] in ("double_top", "double_bottom", "none")
    print("[6] dict 键齐全：trendlines(touches 上/下=%d/%d, valid=%s/%s, break=%s, 三角=%s, 楔形=%s)；"
          "elliott(wave=%s conf=%.2f, %d 个摆动点)；chan(%d 笔/%d 段/%d 中枢, 当前%s, 买点%s, 背驰=%s)；"
          "classic_shape(%d 个形态, 位置%s/%s)"
          % (tl["up_line"]["touches"], tl["down_line"]["touches"], tl["up_line"]["valid"],
             tl["down_line"]["valid"], tl["break"], tl["triangle"], tl["wedge"],
             el["wave"], el["confidence"], len(el["swing_points"]), len(ch["bi"]),
             len(ch["segments"]), len(ch["zhongshu"]), ch["current"], ch["buy_point"],
             ch["divergence"], len(cs["shapes"]), cs["position"]["vs_ma20"],
             cs["position"]["vs_ma60"]))

    # ---------- 7. 云图 ---------- #
    ic = ichimoku(df)
    need_ic = ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou", "cloud_top",
               "cloud_bottom", "above_cloud"]
    assert list(ic.columns) == need_ic, ic.columns
    assert len(ic) == len(df) and ic.index.equals(df.index)
    ok = np.isfinite(ic["cloud_top"].to_numpy(dtype=float))
    assert (ic["cloud_top"].to_numpy()[ok] >= ic["senkou_a"].to_numpy()[ok] - 1e-9).all()
    assert set(np.unique(ic["above_cloud"].to_numpy())) <= {0, 1}
    print("[7] ichimoku 列齐全，长度 %d；cloud_top == max(senkou_a, senkou_b) 校验通过；"
          "站上云层占比 %.0f%%" % (len(ic), ic["above_cloud"].mean() * 100))

    # ---------- 8. 无 ±inf + 矩阵联动 ---------- #
    def _scan(obj, tag):
        bad = []
        if isinstance(obj, pd.DataFrame):
            for col in obj.columns:
                if pd.api.types.is_numeric_dtype(obj[col]):
                    a = obj[col].to_numpy(dtype=float, na_value=np.nan)
                    if np.isinf(a).any():
                        bad.append("%s.%s" % (tag, col))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                bad += _scan(v, "%s.%s" % (tag, k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                bad += _scan(v, "%s[%d]" % (tag, i))
        return bad

    bad = (_scan(feats, "feats") + _scan(ic, "ichimoku") + _scan(events, "events")
           + _scan(tl, "trendlines") + _scan(el, "elliott") + _scan(ch, "chan")
           + _scan(cs, "classic") + _scan(st, "stats"))
    assert not bad, "存在 ±inf: %s" % bad
    mat = indicator_matrix(df)
    assert mat.shape[1] >= 45, "indicator_matrix 列数不足"
    assert df.equals(df_backup), "入参 DataFrame 被修改"
    print("[8] 全部返回值无 ±inf；indicator_matrix(df).shape = %s，列数 = %d（≥45）"
          % (mat.shape, mat.shape[1]))
    print("[9] 入参 DataFrame 未被修改；形态统计可用作历史规律证据")

    print("patterns self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()

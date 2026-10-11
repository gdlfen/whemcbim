# -*- coding: utf-8 -*-
"""
技术指标库（``core.indicators``）。

职责
----
把日线 OHLCV 压缩成 **可被因子层直接消费的数值序列**，覆盖国内主流软件
（通达信 / 同花顺 / 钱龙）的常用指标口径：

* 均线族：``ma`` / ``ema`` / ``smma``(通达信 SMA(X,N,1)) / ``wma`` / ``expma`` / ``bbi``
* 趋势族：``macd`` / ``dkx`` / ``dmi`` / ``sar`` / ``trix`` / ``dma`` / ``lon`` / ``roc`` / ``mtm``
* 摆动族：``kdj`` / ``rsi`` / ``rsi_multi`` / ``cci`` / ``wr`` / ``psy`` / ``bias``
* 通道族：``boll`` / ``ene`` / ``channel_fit``
* 量能族：``vr`` / ``obv`` / ``obv_ma`` / ``asi`` / ``emv`` / ``wvad`` / ``brar``
  / ``cr`` / ``vol_price_corr`` / ``amount_ma`` / ``turnover_rate``
* 筹码与价位：``vwap_cost`` / ``chip_distribution`` / ``support_resistance``
  / ``fib_levels`` / ``golden_pit`` / ``gap_stats`` / ``trend_strength``
* K 线重采样：``renko``（等比砖图） / ``equal_kline``（等价/等量 K 线） / ``resample_kline``
* 汇总：``indicator_matrix``（一次性输出 ≥56 列，供技术面因子层与 ML 特征使用）

统一约定
--------
1. 入参接受 ``pd.Series`` / ``np.ndarray`` / ``list``；内部 :func:`_as_series` 兜底转换。
   输入若为 Series，输出索引与之一致；输入若为 ndarray，输出为 ``RangeIndex``。
2. 返回 ``pd.Series`` 时 **长度与输入一致，头部数据不足处为 NaN，不裁剪、不 fillna**。
3. **绝不抛异常**：任何取不到数据/算不出的情形都返回 NaN 序列或降级 dict，
   并通过 :func:`core.utils.get_logger` 记录 debug 日志。
4. 不修改传入对象（DataFrame 一律 ``.copy()``）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:  # 包内导入（python -m core.indicators）
    from . import config
    from .utils import get_logger, limit_up_pct
except ImportError:  # pragma: no cover - 兼容直接脚本运行
    import config  # type: ignore
    from utils import get_logger, limit_up_pct  # type: ignore

log = get_logger("stockai.indicators")

__all__ = [
    # 均线族
    "ma", "ema", "smma", "wma", "expma", "bbi",
    # 趋势族
    "macd", "dkx", "dmi", "sar", "trix", "dma", "lon", "roc", "mtm",
    # 摆动族
    "kdj", "rsi", "rsi_multi", "cci", "wr", "psy", "bias",
    # 通道族
    "boll", "ene", "channel_fit",
    # 量能族
    "vr", "obv", "obv_ma", "asi", "emv", "wvad", "brar", "cr",
    "vol_price_corr", "amount_ma", "turnover_rate",
    # 筹码 / 价位
    "vwap_cost", "chip_distribution", "support_resistance", "fib_levels",
    "golden_pit", "gap_stats", "trend_strength", "limit_status", "limit_status_full",
    # K 线重构
    "renko", "equal_kline", "resample_kline",
    # 汇总
    "indicator_matrix",
]

_NAN = float("nan")

# 列名容错映射（大小写 / 中文 -> 标准小写英文）
_CANON_COLS = {"date", "open", "high", "low", "close", "volume", "amount",
               "pct_chg", "turnover", "float_shares", "pre_close", "code", "name"}
_CN_COLS = {
    "日期": "date", "时间": "date", "trade_date": "date",
    "开盘": "open", "开盘价": "open", "最高": "high", "最高价": "high",
    "最低": "low", "最低价": "low", "收盘": "close", "收盘价": "close",
    "成交量": "volume", "成交额": "amount", "成交金额": "amount", "成交金额(元)": "amount",
    "换手率": "turnover", "涨跌幅": "pct_chg", "昨收": "pre_close", "前收盘": "pre_close",
    "流通股本": "float_shares", "代码": "code", "名称": "name",
}


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _as_series(x, name: str = "x") -> pd.Series:
    """把任意一维输入转成 float 的 ``pd.Series``（保留原索引；失败返回空 Series）。"""
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        try:
            s = pd.to_numeric(x, errors="coerce")
        except Exception:  # noqa: BLE001
            s = pd.Series(np.full(len(x), np.nan), index=x.index, dtype=float)
        if not pd.api.types.is_float_dtype(s):
            s = s.astype(float)
        return s
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:  # noqa: BLE001
        log.debug("_as_series(%s) 转换失败", name)
        return pd.Series(dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.ravel()
    return pd.Series(arr, dtype=float)


def _ensure_frame(df) -> pd.DataFrame:
    """列名容错（大写/中文 -> 小写英文），并保证 ``date`` 为 datetime；不修改入参。"""
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        out = df.copy()
    else:
        try:
            out = pd.DataFrame(df)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    ren = {}
    for col in out.columns:
        key = str(col).strip()
        low = key.lower()
        if low in _CANON_COLS:
            ren[col] = low
        elif key in _CN_COLS:
            ren[col] = _CN_COLS[key]
    if ren:
        out = out.rename(columns=ren)
    if "date" in out.columns:
        try:
            out["date"] = pd.to_datetime(out["date"])
        except Exception:  # noqa: BLE001
            pass
    return out


def _cols(df: pd.DataFrame, *names, default=_NAN):
    """按名字取列，缺失则返回常量填充的 Series（保证下游不 KeyError）。"""
    for nm in names:
        if nm in df.columns:
            return _as_series(df[nm], nm)
    return pd.Series(np.full(len(df), default), index=df.index, dtype=float)


def _num(x, default=_NAN) -> float:
    """安全取最后一个有效数值。"""
    try:
        if isinstance(x, pd.Series):
            if x.empty:
                return default
            v = x.iloc[-1]
        else:
            v = x
        v = float(v)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:  # noqa: BLE001
        return default


def _mask_warmup(values, n: int, first_valid: int | None):
    """把前 ``n-1`` 个样本（相对首个有效值）置为 NaN，实现"头部不足为 NaN"。"""
    out = np.asarray(values, dtype=float).copy()
    if first_valid is None or out.size == 0:
        return out
    end = int(first_valid) + max(int(n), 1) - 1
    if end > 0:
        out[:min(end, out.size)] = np.nan
    return out


def _first_valid(arr) -> int | None:
    idx = np.flatnonzero(np.isfinite(np.asarray(arr, dtype=float)))
    return int(idx[0]) if idx.size else None


def _safe_div(a, b, fill=np.nan):
    """逐元素安全除法：分母为 0/NaN 时返回 fill（不产生 ±inf）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b, out=np.full(np.broadcast(a, b).shape, fill, dtype=float),
                        where=(b != 0) & np.isfinite(b) & np.isfinite(a))
    return out


def _smma_core(arr, n: int) -> tuple[np.ndarray, int | None]:
    """通达信 ``SMA(X,N,1)`` 递归：``Y = (X + (N-1)*Y') / N``。

    从**首个有效值**开始播种（Y'=X），与通达信完全一致；中间遇到 NaN 时
    输出 NaN 但保留递归状态（不 reset）。
    """
    n = max(int(n), 1)
    a = np.asarray(arr, dtype=float)
    out = np.full(a.size, np.nan)
    prev = np.nan
    started = False
    first = None
    for i in range(a.size):
        x = a[i]
        if not np.isfinite(x):
            continue
        if not started:
            prev = x
            started = True
            first = i
        else:
            prev = (x + (n - 1) * prev) / n
        out[i] = prev
    return out, first


def _cross_state(a: pd.Series, b: pd.Series) -> str:
    """判断最后一根 K 线上 a 与 b 的交叉：``"gold"`` / ``"dead"`` / ``"none"``。"""
    try:
        if len(a) < 2 or len(b) < 2:
            return "none"
        a1, a2 = float(a.iloc[-1]), float(a.iloc[-2])
        b1, b2 = float(b.iloc[-1]), float(b.iloc[-2])
        if not all(np.isfinite([a1, a2, b1, b2])):
            return "none"
        if a2 <= b2 and a1 > b1:
            return "gold"
        if a2 >= b2 and a1 < b1:
            return "dead"
    except Exception:  # noqa: BLE001
        pass
    return "none"


def _rolling_sum(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n), min_periods=int(n)).sum()


# --------------------------------------------------------------------------- #
# 一、均线族
# --------------------------------------------------------------------------- #
def ma(close, n=5) -> pd.Series:
    """简单移动平均 ``MA(X,N)``。"""
    s = _as_series(close, "close")
    n = max(int(n), 1)
    if s.empty:
        return s
    return s.rolling(n, min_periods=n).mean()


def ema(close, n=12) -> pd.Series:
    """指数移动平均 ``EMA(X,N)``：``Y = 2X/(N+1) + (N-1)Y'/(N+1)``。

    与通达信一致采用 ``adjust=False``；头部 ``N-1`` 根返回 NaN。
    """
    s = _as_series(close, "close")
    n = max(int(n), 1)
    if s.empty:
        return s
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def smma(close, n=14) -> pd.Series:
    """通达信 ``SMA(X,N,1)`` 平滑：``Y = (X + (N-1)*Y') / N``。

    递归从首个有效值播种（与通达信数值一致），但输出头部 ``N-1`` 根置 NaN，
    以符合"数据不足返回 NaN"的全局约定。
    """
    s = _as_series(close, "close")
    n = max(int(n), 1)
    if s.empty:
        return s
    raw, first = _smma_core(s.to_numpy(dtype=float), n)
    return pd.Series(_mask_warmup(raw, n, first), index=s.index)


def wma(close, n=5) -> pd.Series:
    """加权移动平均 ``WMA(X,N)``，权重 ``1,2,...,N``（越近权重越大）。"""
    s = _as_series(close, "close")
    n = max(int(n), 1)
    if s.empty:
        return s
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    return s.rolling(n, min_periods=n).apply(lambda x: float(np.dot(x, w)), raw=True)


def expma(close, n1=12, n2=50) -> dict:
    """EXPMA（指数平滑移动平均）双线。"""
    return {"expma1": ema(close, n1), "expma2": ema(close, n2)}


def bbi(close, n1=3, n2=6, n3=12, n4=24) -> pd.Series:
    """BBI 多空指数 = ``(MA3+MA6+MA12+MA24)/4``。"""
    s = _as_series(close, "close")
    if s.empty:
        return s
    return (ma(s, n1) + ma(s, n2) + ma(s, n3) + ma(s, n4)) / 4.0


# --------------------------------------------------------------------------- #
# 二、趋势族
# --------------------------------------------------------------------------- #
def dkx(open_, high, low, close, n=10, m=10) -> dict:
    """多空线 DKX。

    ``MID=(3*C+L+O+H)/6``；``DKX=MA(MID,m)``；
    ``trend_up = DKX 上升 且 MID>DKX``；``cross``：MID 上/下穿 DKX。
    """
    o = _as_series(open_, "open")
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    out = {"mid": pd.Series(dtype=float), "dkx": pd.Series(dtype=float),
           "trend_up": False, "cross": "none"}
    if c.empty:
        return out
    mid = (3.0 * c + l + o + h) / 6.0
    dkx_line = ma(mid, m)
    du = bool(np.isfinite(dkx_line.iloc[-1]) and np.isfinite(dkx_line.iloc[-2])
              and dkx_line.iloc[-1] > dkx_line.iloc[-2])
    trend_up = bool(du and np.isfinite(mid.iloc[-1]) and np.isfinite(dkx_line.iloc[-1])
                    and mid.iloc[-1] > dkx_line.iloc[-1])
    return {"mid": mid, "dkx": dkx_line, "trend_up": trend_up,
            "cross": _cross_state(mid, dkx_line)}


def macd(close, fast=12, slow=26, signal=9) -> dict:
    """MACD（平滑异同移动平均）。

    ``DIF=EMA(C,fast)-EMA(C,slow)``；``DEA=EMA(DIF,signal)``；
    柱 ``MACD=2*(DIF-DEA)``（国内软件口径）。
    ``hist_trend`` ∈ ``growing/shrinking/turning_up/turning_down/flat``。
    """
    s = _as_series(close, "close")
    empty = pd.Series(dtype=float)
    out = {"dif": empty, "dea": empty, "macd": empty, "cross": "none", "hist_trend": "flat"}
    if s.empty:
        return out
    dif = ema(s, fast) - ema(s, slow)
    dea = ema(dif, signal)
    hist = 2.0 * (dif - dea)
    h3 = hist.iloc[-3:].to_numpy(dtype=float)
    trend = "flat"
    if h3.size == 3 and np.isfinite(h3).all():
        if h3[1] > h3[0] and h3[2] > h3[1]:
            trend = "growing"
        elif h3[1] < h3[0] and h3[2] < h3[1]:
            trend = "shrinking"
        elif h3[2] > h3[1]:
            trend = "turning_up"
        elif h3[2] < h3[1]:
            trend = "turning_down"
    elif h3.size >= 2 and np.isfinite(h3[-2:]).all():
        trend = "turning_up" if h3[-1] > h3[-2] else ("turning_down" if h3[-1] < h3[-2] else "flat")
    return {"dif": dif, "dea": dea, "macd": hist,
            "cross": _cross_state(dif, dea), "hist_trend": trend}


def kdj(high, low, close, n=9, m1=3, m2=3) -> dict:
    """随机指标 KDJ（通达信口径）。

    ``RSV=(C-LLV(L,n))/(HHV(H,n)-LLV(L,n))*100``；
    ``K=SMA(RSV,m1,1)``，``D=SMA(K,m2,1)``，``J=3K-2D``；**初值 K=D=50**。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    empty = pd.Series(dtype=float)
    out = {"k": empty, "d": empty, "j": empty, "cross": "none", "j_value": _NAN}
    if c.empty:
        return out
    n = max(int(n), 1)
    llv = l.rolling(n, min_periods=n).min()
    hhv = h.rolling(n, min_periods=n).max()
    rng = (hhv - llv).to_numpy(dtype=float)
    rsv = pd.Series(_safe_div((c - llv).to_numpy(dtype=float), rng) * 100.0, index=c.index)
    # ---- K = SMA(RSV, m1, 1)，初值 50 ------------------------------------ #
    rsv_arr = rsv.to_numpy(dtype=float)
    first = _first_valid(rsv_arr)
    k_arr = np.full(rsv_arr.size, np.nan)
    d_arr = np.full(rsv_arr.size, np.nan)
    if first is not None:
        prev_k, prev_d = 50.0, 50.0
        m1 = max(int(m1), 1)
        m2 = max(int(m2), 1)
        for i in range(first, rsv_arr.size):
            x = rsv_arr[i]
            if not np.isfinite(x):
                continue
            prev_k = (x + (m1 - 1) * prev_k) / m1
            prev_d = (prev_k + (m2 - 1) * prev_d) / m2
            k_arr[i], d_arr[i] = prev_k, prev_d
    k = pd.Series(k_arr, index=c.index)
    d = pd.Series(d_arr, index=c.index)
    j = 3.0 * k - 2.0 * d
    return {"k": k, "d": d, "j": j, "cross": _cross_state(k, d), "j_value": _num(j)}


def rsi(close, n=14) -> pd.Series:
    """相对强弱指标（通达信口径）。

    ``RSI = SMA(MAX(C-REF(C,1),0),N,1) / SMA(ABS(C-REF(C,1)),N,1) * 100``。
    """
    s = _as_series(close, "close")
    n = max(int(n), 1)
    if s.empty:
        return s
    diff = s.diff()
    up = diff.clip(lower=0.0)
    ab = diff.abs()
    num_raw, f1 = _smma_core(up.to_numpy(dtype=float), n)
    den_raw, f2 = _smma_core(ab.to_numpy(dtype=float), n)
    val = _safe_div(num_raw, den_raw) * 100.0
    first = f1 if f1 is not None else f2
    return pd.Series(_mask_warmup(val, n, first), index=s.index)


def rsi_multi(close, ns=(6, 12, 24)) -> dict:
    """多周期 RSI，键为 ``rsi6`` / ``rsi12`` / ``rsi24`` …"""
    return {"rsi%d" % int(n): rsi(close, n) for n in ns}


def boll(close, n=20, k=2.0) -> dict:
    """布林带 BOLL。

    ``mid=MA(C,n)``；``upper/lower=mid±k*STD(C,n)``（样本标准差 ddof=1，同通达信 STD）；
    ``width=(upper-lower)/mid``；``%B=(C-lower)/(upper-lower)``；``bandwidth_pct=width*100``。
    """
    s = _as_series(close, "close")
    empty = pd.Series(dtype=float)
    out = {"mid": empty, "upper": empty, "lower": empty, "width": empty,
           "pct_b": empty, "bandwidth_pct": empty}
    if s.empty:
        return out
    n = max(int(n), 1)
    mid = ma(s, n)
    std = s.rolling(n, min_periods=n).std(ddof=1)
    upper = mid + float(k) * std
    lower = mid - float(k) * std
    width = pd.Series(_safe_div((upper - lower).to_numpy(dtype=float),
                                mid.to_numpy(dtype=float)), index=s.index)
    pct_b = pd.Series(_safe_div((s - lower).to_numpy(dtype=float),
                                (upper - lower).to_numpy(dtype=float)), index=s.index)
    return {"mid": mid, "upper": upper, "lower": lower, "width": width,
            "pct_b": pct_b, "bandwidth_pct": width * 100.0}


def ene(close, n=25, k_up=11.0, k_dn=9.0) -> dict:
    """轨道线 ENE：``upper=(1+k_up/100)*MA(C,n)``，``lower=(1-k_dn/100)*MA(C,n)``。

    ``position``：最新一根 ``above`` / ``below`` / ``inside``。
    """
    s = _as_series(close, "close")
    empty = pd.Series(dtype=float)
    if s.empty:
        return {"upper": empty, "lower": empty, "mid": empty, "position": "inside"}
    mid = ma(s, n)
    upper = (1.0 + float(k_up) / 100.0) * mid
    lower = (1.0 - float(k_dn) / 100.0) * mid
    p_now, u_now, l_now = _num(s), _num(upper), _num(lower)
    if not (np.isfinite(p_now) and np.isfinite(u_now) and np.isfinite(l_now)):
        pos = "inside"
    elif p_now > u_now:
        pos = "above"
    elif p_now < l_now:
        pos = "below"
    else:
        pos = "inside"
    return {"upper": upper, "lower": lower, "mid": mid, "position": pos}


def sar(high, low, af_step=0.02, af_max=0.2) -> dict:
    """抛物线转向 SAR（Wilder 标准算法）。

    * 首根 NaN，第二根用前两根高低点确定初始方向与初始 SAR；
    * ``trend`` ∈ {+1 多头, -1 空头}；``flip``：+1 由空翻多、-1 由多翻空、0 无翻转。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    n = min(len(h), len(l))
    sar_s = pd.Series(np.full(len(h), np.nan), index=h.index, dtype=float)
    trend_s = pd.Series(np.zeros(len(h), dtype=int), index=h.index)
    flip_s = pd.Series(np.zeros(len(h), dtype=int), index=h.index)
    if n < 3:
        return {"sar": sar_s, "trend": trend_s, "flip": flip_s}
    hi = h.to_numpy(dtype=float)[:n]
    lo = l.to_numpy(dtype=float)[:n]
    out = np.full(n, np.nan)
    tr = np.zeros(n, dtype=int)
    fl = np.zeros(n, dtype=int)
    # 用前两根方向初始化
    bull = hi[1] >= hi[0]
    if bull:
        trend, ep, sarv, af = 1, hi[1], lo[0], float(af_step)
    else:
        trend, ep, sarv, af = -1, lo[1], hi[0], float(af_step)
    out[1], tr[1] = sarv, trend
    for i in range(2, n):
        prev = out[i - 1]
        if trend == 1:
            cur = prev + af * (ep - prev)
            cur = min(cur, lo[i - 1], lo[i - 2])       # 多头 SAR 不得高于前两根最低价
            if lo[i] < cur:                            # 跌破 -> 翻空
                trend, cur, ep, af = -1, ep, lo[i], float(af_step)
                fl[i] = -1
            elif hi[i] > ep:                           # 创新高 -> 加速
                ep = hi[i]
                af = min(af + float(af_step), float(af_max))
        else:
            cur = prev + af * (ep - prev)
            cur = max(cur, hi[i - 1], hi[i - 2])       # 空头 SAR 不得低于前两根最高价
            if hi[i] > cur:                            # 突破 -> 翻多
                trend, cur, ep, af = 1, ep, hi[i], float(af_step)
                fl[i] = 1
            elif lo[i] < ep:
                ep = lo[i]
                af = min(af + float(af_step), float(af_max))
        out[i], tr[i] = cur, trend
    try:
        sar_s.iloc[:n] = out
        trend_s.iloc[:n] = tr
        flip_s.iloc[:n] = fl
    except Exception:  # noqa: BLE001
        sar_s = pd.Series(out, index=h.index[:n])
        trend_s = pd.Series(tr, index=h.index[:n])
        flip_s = pd.Series(fl, index=h.index[:n])
    return {"sar": sar_s, "trend": trend_s, "flip": flip_s}


def bias(close, ns=(6, 12, 24)) -> dict:
    """乖离率 ``BIAS=(C-MA(C,N))/MA(C,N)*100``。"""
    s = _as_series(close, "close")
    out = {}
    for n in ns:
        m = ma(s, n)
        out["bias%d" % int(n)] = pd.Series(
            _safe_div((s - m).to_numpy(dtype=float), m.to_numpy(dtype=float)) * 100.0,
            index=s.index) if not s.empty else pd.Series(dtype=float)
    return out


def cci(high, low, close, n=14) -> pd.Series:
    """顺势指标 CCI：``(TP-MA(TP,n))/(0.015*AVEDEV(TP,n))``，``TP=(H+L+C)/3``。

    ``AVEDEV(X,N)`` 按国内软件（通达信/文华）函数手册口径：
    **窗口内每个点与「该窗口均值」之差的绝对值的平均**，即
    ``AVEDEV[i] = mean(|X[j] - MA(X,N)[i]|), j ∈ [i-N+1, i]``。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    if c.empty:
        return c
    n = max(int(n), 1)
    tp = (h + l + c) / 3.0
    matp = ma(tp, n)
    avedev = tp.rolling(n, min_periods=n).apply(
        lambda w: float(np.mean(np.abs(w - w.mean()))), raw=True)
    return pd.Series(_safe_div((tp - matp).to_numpy(dtype=float),
                               (0.015 * avedev).to_numpy(dtype=float)), index=c.index)


def dmi(high, low, close, n=14, m=6) -> dict:
    """动向指标 DMI / ADX / ADXR（Wilder 平滑口径）。

    ``TR=max(H-L,|H-C'|,|L-C'|)``；``+DM/-DM`` 为方向动量；
    ``PDI=100*SMA(+DM,n,1)/SMA(TR,n,1)``；``ADX=SMA(|MDI-PDI|/(MDI+PDI)*100,m)``；
    ``ADXR=(ADX+REF(ADX,m))/2``。

    ``signal``：PDI>MDI 且 ADX 上升 → ``bull``；PDI<MDI 且 ADX 上升 → ``bear``；
    ADX 下降 → ``weak``（趋势衰竭）；其余 → ``range``。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    empty = pd.Series(dtype=float)
    out = {"pdi": empty, "mdi": empty, "adx": empty, "adxr": empty, "signal": "range"}
    if c.empty:
        return out
    n, m = max(int(n), 1), max(int(m), 1)
    pc = c.shift(1)
    ph = h.shift(1)
    pl = l.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    hd = h - ph
    ld = pl - l
    pdm = hd.where((hd > 0) & (hd > ld), 0.0)
    mdm = ld.where((ld > 0) & (ld > hd), 0.0)
    pdm = pdm.where(pc.notna())      # 首根无意义
    mdm = mdm.where(pc.notna())
    atr = smma(tr, n)
    pdi = pd.Series(_safe_div(smma(pdm, n).to_numpy(dtype=float),
                              atr.to_numpy(dtype=float)) * 100.0, index=c.index)
    mdi = pd.Series(_safe_div(smma(mdm, n).to_numpy(dtype=float),
                              atr.to_numpy(dtype=float)) * 100.0, index=c.index)
    dx = pd.Series(_safe_div((mdi - pdi).abs().to_numpy(dtype=float),
                             (mdi + pdi).to_numpy(dtype=float)) * 100.0, index=c.index)
    adx = smma(dx, m)
    adxr = (adx + adx.shift(m)) / 2.0
    p_now, mm_now = _num(pdi), _num(mdi)
    a_now, a_prev = _num(adx), _num(adx.iloc[:-1]) if len(adx) > 1 else _NAN
    signal = "range"
    if np.isfinite(a_now) and np.isfinite(a_prev):
        rising = a_now > a_prev
        if rising and p_now > mm_now:
            signal = "bull"
        elif rising and p_now < mm_now:
            signal = "bear"
        elif a_now < a_prev:
            signal = "weak"
    return {"pdi": pdi, "mdi": mdi, "adx": adx, "adxr": adxr, "signal": signal}


def wr(high, low, close, ns=(6, 10)) -> dict:
    """威廉指标 WR，**返回负值**（-100 ~ 0）。

    ``WR = -100 * (HHV(H,N)-C) / (HHV(H,N)-LLV(L,N))``（国内软件习惯）。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    out = {}
    for nn in ns:
        nn = max(int(nn), 1)
        if c.empty:
            out["wr%d" % nn] = pd.Series(dtype=float)
            continue
        hhv = h.rolling(nn, min_periods=nn).max()
        llv = l.rolling(nn, min_periods=nn).min()
        out["wr%d" % nn] = pd.Series(
            _safe_div((hhv - c).to_numpy(dtype=float),
                      (hhv - llv).to_numpy(dtype=float)) * -100.0, index=c.index)
    return out


def dma(close, n1=10, n2=50, m=10) -> dict:
    """平行线差 DMA：``DMA=MA(C,n1)-MA(C,n2)``，``AMA=MA(DMA,m)``。"""
    s = _as_series(close, "close")
    if s.empty:
        return {"dma": s, "ama": s}
    d = ma(s, n1) - ma(s, n2)
    return {"dma": d, "ama": ma(d, m)}


def trix(close, n=12, m=20) -> dict:
    """三重指数平滑平均 TRIX。

    ``TR=EMA(EMA(EMA(C,n),n),n)``；``TRIX=(TR-REF(TR,1))/REF(TR,1)*100``；``TRMA=MA(TRIX,m)``。
    """
    s = _as_series(close, "close")
    empty = pd.Series(dtype=float)
    if s.empty:
        return {"trix": empty, "trma": empty, "cross": "none"}
    n = max(int(n), 1)
    tr = ema(ema(ema(s, n), n), n)
    prev = tr.shift(1)
    tx = pd.Series(_safe_div((tr - prev).to_numpy(dtype=float),
                             prev.to_numpy(dtype=float)) * 100.0, index=s.index)
    trma = ma(tx, m)
    return {"trix": tx, "trma": trma, "cross": _cross_state(tx, trma)}


def lon(close, n=10) -> dict:
    """钱龙长线 LON = ``(MA(C,n) + MA(REF(C,1),n)) / 2``，``lon_ma=MA(lon,5)``。"""
    s = _as_series(close, "close")
    if s.empty:
        return {"lon": s, "lon_ma": s}
    line = (ma(s, n) + ma(s.shift(1), n)) / 2.0
    return {"lon": line, "lon_ma": ma(line, 5)}


# --------------------------------------------------------------------------- #
# 三、量能族
# --------------------------------------------------------------------------- #
def brar(open_, high, low, close, n=26) -> dict:
    """人气意愿指标 BRAR。

    ``AR=SUM(H-O,N)/SUM(O-L,N)*100``；
    ``BR=SUM(MAX(0,H-REF(C,1)),N)/SUM(MAX(0,REF(C,1)-L),N)*100``。
    """
    o = _as_series(open_, "open")
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    if c.empty:
        return {"ar": c, "br": c}
    n = max(int(n), 1)
    ar = pd.Series(_safe_div(_rolling_sum(h - o, n).to_numpy(dtype=float),
                             _rolling_sum(o - l, n).to_numpy(dtype=float)) * 100.0,
                   index=c.index)
    pc = c.shift(1)
    up = (h - pc).clip(lower=0.0)
    dn = (pc - l).clip(lower=0.0)
    br = pd.Series(_safe_div(_rolling_sum(up, n).to_numpy(dtype=float),
                             _rolling_sum(dn, n).to_numpy(dtype=float)) * 100.0,
                   index=c.index)
    return {"ar": ar, "br": br}


def vr(close, volume, n=26) -> pd.Series:
    """成交量比率 VR。

    ``VR=(N日上涨日量+1/2平盘量)/(N日下跌日量+1/2平盘量)*100``。
    """
    c = _as_series(close, "close")
    v = _as_series(volume, "volume")
    if c.empty:
        return c
    n = max(int(n), 1)
    pc = c.shift(1)
    d = c - pc
    up = v.where(d > 0, 0.0)
    dn = v.where(d < 0, 0.0)
    flat = v.where((d == 0) | pc.isna(), 0.0)      # 首根按平盘处理（同通达信）
    num = _rolling_sum(up, n) + 0.5 * _rolling_sum(flat, n)
    den = _rolling_sum(dn, n) + 0.5 * _rolling_sum(flat, n)
    return pd.Series(_safe_div(num.to_numpy(dtype=float), den.to_numpy(dtype=float)) * 100.0,
                     index=c.index)


def obv(close, volume) -> pd.Series:
    """能量潮 OBV（首根记 0，与通达信 ``SUM(IF(...),0)`` 一致）。"""
    c = _as_series(close, "close")
    v = _as_series(volume, "volume")
    if c.empty:
        return c
    d = c.diff()
    step = np.where(d > 0, v.to_numpy(dtype=float),
                    np.where(d < 0, -v.to_numpy(dtype=float), 0.0))
    step = np.nan_to_num(step, nan=0.0)
    return pd.Series(np.cumsum(step), index=c.index)


def obv_ma(close, volume, n=30) -> dict:
    """OBV 及其均线。"""
    o = obv(close, volume)
    return {"obv": o, "obv_ma": ma(o, n)}


def asi(open_, high, low, close, n=26) -> dict:
    """累计震荡量 ASI。

    ``SI=16*(C-C'+0.5*(C-O)+0.25*(C'-O')) /
         (3*MAX(|H-C'|,|L-C'|)+0.5*|C'-O'|) * MAX(H-L,|H-C'|,|L-C'|)``；
    ``ASI=SUM(SI,n)``；``ASIMA=MA(ASI,6)``。
    """
    o = _as_series(open_, "open")
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    if c.empty:
        return {"asi": c, "asima": c, "si": c}
    n = max(int(n), 1)
    pc = c.shift(1)
    po = o.shift(1)
    num = (c - pc) + 0.5 * (c - o) + 0.25 * (pc - po)
    den = 3.0 * pd.concat([(h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1) \
        + 0.5 * (pc - po).abs()
    rng = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    si = pd.Series(_safe_div(16.0 * num.to_numpy(dtype=float), den.to_numpy(dtype=float),
                             fill=0.0) * rng.to_numpy(dtype=float), index=c.index)
    si = si.where(pc.notna())
    line = si.rolling(n, min_periods=n).sum()
    return {"asi": line, "asima": ma(line, 6), "si": si}


def emv(high, low, volume, n=14, m=9) -> dict:
    """简易波动指标 EMV。

    ``MID=(H+L)/2``；``EMV=SUM((MID-REF(MID,1))*(H-L)/(VOL/((H+L)/2)), n)``；
    ``EMV_MA=MA(EMV,m)``（默认 m=9）。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    v = _as_series(volume, "volume")
    if h.empty:
        return {"emv": h, "emv_ma": h}
    mid = (h + l) / 2.0
    dm = mid - mid.shift(1)
    box = pd.Series(_safe_div(v.to_numpy(dtype=float), mid.to_numpy(dtype=float)),
                    index=h.index)                      # 单位价格成交量
    raw = pd.Series(_safe_div((dm * (h - l)).to_numpy(dtype=float),
                              box.to_numpy(dtype=float), fill=0.0), index=h.index)
    line = raw.rolling(max(int(n), 1), min_periods=max(int(n), 1)).sum()
    return {"emv": line, "emv_ma": ma(line, m)}


def wvad(open_, high, low, close, volume, n=24) -> dict:
    """威廉变异离散量 WVAD：``SUM((C-O)/(H-L)*VOL, n)``，``WVAD_MA=MA(WVAD,6)``。"""
    o = _as_series(open_, "open")
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    v = _as_series(volume, "volume")
    if c.empty:
        return {"wvad": c, "wvad_ma": c}
    ratio = pd.Series(_safe_div((c - o).to_numpy(dtype=float),
                                (h - l).to_numpy(dtype=float), fill=0.0), index=c.index)
    raw = ratio * v
    line = raw.rolling(max(int(n), 1), min_periods=max(int(n), 1)).sum()
    return {"wvad": line, "wvad_ma": ma(line, 6)}


def roc(close, n=12, m=6) -> dict:
    """变动率 ROC：``(C-REF(C,n))/REF(C,n)*100``，``ROC_MA=MA(ROC,m)``。"""
    s = _as_series(close, "close")
    if s.empty:
        return {"roc": s, "roc_ma": s}
    ref = s.shift(max(int(n), 1))
    line = pd.Series(_safe_div((s - ref).to_numpy(dtype=float),
                               ref.to_numpy(dtype=float)) * 100.0, index=s.index)
    return {"roc": line, "roc_ma": ma(line, m)}


def psy(close, n=12) -> pd.Series:
    """心理线 PSY：``N 日内上涨天数占比 * 100``。"""
    s = _as_series(close, "close")
    n = max(int(n), 1)
    if s.empty:
        return s
    up = (s.diff() > 0).astype(float)
    up = up.where(s.diff().notna())
    return up.rolling(n, min_periods=n).sum() / n * 100.0


def mtm(close, n=12, m=6) -> dict:
    """动量指标 MTM：``C-REF(C,n)``，``MTM_MA=MA(MTM,m)``。"""
    s = _as_series(close, "close")
    if s.empty:
        return {"mtm": s, "mtm_ma": s}
    line = s - s.shift(max(int(n), 1))
    return {"mtm": line, "mtm_ma": ma(line, m)}


def cr(high, low, close, n=26) -> dict:
    """中间意愿指标 CR。

    ``MID=REF((H+L)/2,1)``；``CR=SUM(MAX(0,H-MID),n)/SUM(MAX(0,MID-L),n)*100``；
    ``MA1/MA2/MA3 = REF(MA(CR,5),3) / REF(MA(CR,10),5) / REF(MA(CR,20),9)``（通达信默认参数）。

    注：``close`` 保留以统一调用签名，CR 公式本身不使用收盘价。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    if h.empty:
        e = pd.Series(dtype=float)
        return {"cr": e, "ma1": e, "ma2": e, "ma3": e}
    n = max(int(n), 1)
    mid = ((h + l) / 2.0).shift(1)
    up = (h - mid).clip(lower=0.0)
    dn = (mid - l).clip(lower=0.0)
    line = pd.Series(_safe_div(_rolling_sum(up, n).to_numpy(dtype=float),
                               _rolling_sum(dn, n).to_numpy(dtype=float)) * 100.0,
                     index=h.index)
    return {"cr": line, "ma1": ma(line, 5).shift(3), "ma2": ma(line, 10).shift(5),
            "ma3": ma(line, 20).shift(9)}


def vol_price_corr(close, volume, n=20) -> pd.Series:
    """价量滚动相关系数（Pearson，窗口 n，0~1 为正相关）。"""
    c = _as_series(close, "close")
    v = _as_series(volume, "volume")
    if c.empty:
        return c
    n = max(int(n), 1)
    return c.rolling(n, min_periods=n).corr(v)


def amount_ma(amount, ns=(5, 20)) -> dict:
    """成交额均线，键为 ``amount_ma5`` / ``amount_ma20`` …"""
    a = _as_series(amount, "amount")
    return {"amount_ma%d" % int(n): ma(a, n) for n in ns}


def turnover_rate(volume, float_shares) -> pd.Series:
    """换手率（%）= ``成交量 / 流通股本 * 100``。"""
    v = _as_series(volume, "volume")
    if isinstance(float_shares, pd.Series):
        fs = _as_series(float_shares, "float_shares")
        den = fs.to_numpy(dtype=float)
    else:
        try:
            den = np.full(len(v), float(float_shares))
        except (TypeError, ValueError):
            den = np.full(len(v), np.nan)
    if v.empty:
        return v
    den = np.where(den > 0, den, np.nan)
    return pd.Series(_safe_div(v.to_numpy(dtype=float), den) * 100.0, index=v.index)


# --------------------------------------------------------------------------- #
# 四、筹码 / 成本
# --------------------------------------------------------------------------- #
def vwap_cost(high, low, close, volume, n=60) -> dict:
    """持仓成本线。

    * ``vwap``   ：近 ``n`` 日成交量加权均价（成交额缺失时以典型价×量近似）；
    * ``cost_60``：近 60 日成交量加权平均价；
    * ``profit_ratio``：近 250 日筹码分布中位于当前价以下的占比（0~1，越大获利盘越多）。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    v = _as_series(volume, "volume")
    empty = pd.Series(dtype=float)
    if c.empty:
        return {"vwap": empty, "cost_60": empty, "profit_ratio": _NAN}
    n = max(int(n), 1)
    tp = (h + l + c) / 3.0
    amt = tp * v

    def _vwap(win: int) -> pd.Series:
        win = max(int(win), 1)
        num = amt.rolling(win, min_periods=win).sum()
        den = v.rolling(win, min_periods=win).sum()
        return pd.Series(_safe_div(num.to_numpy(dtype=float), den.to_numpy(dtype=float)),
                         index=c.index)

    vwap = _vwap(n)
    cost60 = _vwap(60)
    chip = chip_distribution(h, l, c, v, bins=int(config.INDICATOR_PARAMS.get("chip_bins", 120)),
                             lookback=250)
    pr = chip.get("profit_ratio", _NAN)
    return {"vwap": vwap, "cost_60": cost60, "profit_ratio": pr}


def chip_distribution(high, low, close, volume, bins=120, lookback=250) -> dict:
    """筹码分布（成本分布）。

    把 ``lookback`` 日内每天的成交量按当日价格区间 ``[low, high]`` **均匀分摊**到
    ``bins`` 个价格箱，得到归一化筹码分布 ``hist``（``sum(hist)=1``）与箱中心 ``prices``。

    Returns
    -------
    dict
        ``hist`` 筹码分布（ndarray，长度 bins）, ``prices`` 箱中心（ndarray）,
        ``concentration`` 前 10% 价格箱承载的筹码占比（越高越集中）,
        ``avg_cost`` 筹码加权均价, ``peak_price`` 筹码峰价格（最大单箱）,
        ``main_peak`` 最大单峰区间加权中心, ``profit_ratio`` 当前价下方筹码占比。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    v = _as_series(volume, "volume")
    empty = {"hist": np.zeros(0), "prices": np.zeros(0), "concentration": _NAN,
             "avg_cost": _NAN, "peak_price": _NAN, "main_peak": _NAN, "profit_ratio": _NAN}
    m = min(len(h), len(l), len(c), len(v))
    if m < 2:
        return empty
    k = int(min(max(int(lookback), 2), m))
    hh = h.iloc[-k:].to_numpy(dtype=float)
    ll = l.iloc[-k:].to_numpy(dtype=float)
    cc = c.iloc[-k:].to_numpy(dtype=float)
    vv = np.nan_to_num(v.iloc[-k:].to_numpy(dtype=float), nan=0.0)
    ok = np.isfinite(hh) & np.isfinite(ll) & np.isfinite(cc)
    if not ok.any():
        return empty
    hh, ll, cc, vv = hh[ok], ll[ok], cc[ok], vv[ok]
    lo0, hi0 = float(np.nanmin(ll)), float(np.nanmax(hh))
    if not np.isfinite(lo0) or not np.isfinite(hi0) or hi0 <= lo0:
        return empty
    bins = max(int(bins), 4)
    edges = np.linspace(lo0, hi0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = np.diff(edges)
    hist = np.zeros(bins, dtype=float)
    for i in range(hh.size):
        hi_i, lo_i, vol_i = hh[i], ll[i], vv[i]
        if not (np.isfinite(hi_i) and np.isfinite(lo_i)) or vol_i <= 0:
            continue
        if hi_i <= lo_i:                                  # 一字板：全部落在同一箱
            j = int(np.clip(np.searchsorted(edges, lo_i, side="right") - 1, 0, bins - 1))
            hist[j] += vol_i
            continue
        overlap = np.clip(np.minimum(edges[1:], hi_i) - np.maximum(edges[:-1], lo_i), 0.0, None)
        hist += overlap / (hi_i - lo_i) * vol_i
    total = float(hist.sum())
    if total <= 0:
        return empty
    hist /= total
    price_now = float(cc[-1])
    avg_cost = float(np.dot(hist, centers))
    peak_idx = int(np.argmax(hist))
    peak_price = float(centers[peak_idx])
    # 最大单峰：从峰位向两侧扩张到局部低点，再取加权中心
    left = peak_idx
    while left - 1 >= 0 and hist[left - 1] <= hist[left]:
        left -= 1
    right = peak_idx
    while right + 1 < bins and hist[right + 1] <= hist[right]:
        right += 1
    seg = hist[left:right + 1]
    main_peak = float(np.dot(seg, centers[left:right + 1]) / seg.sum()) if seg.sum() > 0 else peak_price
    top_n = max(1, int(round(bins * 0.1)))
    concentration = float(np.sort(hist)[::-1][:top_n].sum())
    profit_ratio = float(hist[centers <= price_now].sum())
    return {"hist": hist, "prices": centers, "concentration": concentration,
            "avg_cost": avg_cost, "peak_price": peak_price, "main_peak": main_peak,
            "profit_ratio": profit_ratio}


# --------------------------------------------------------------------------- #
# 五、支撑压力 / 通道 / 黄金分割
# --------------------------------------------------------------------------- #
def _swing_points(arr: np.ndarray, r: int = 3):
    """左右各 ``r`` 根确认的局部极值点，返回 (高点索引列表, 低点索引列表)。"""
    a = np.asarray(arr, dtype=float)
    highs, lows = [], []
    n = a.size
    for i in range(r, n - r):
        around = np.concatenate([a[i - r:i], a[i + 1:i + r + 1]])
        if around.size == 0 or not np.isfinite(a[i]):
            continue
        if a[i] >= around.max():
            highs.append(i)
        if a[i] <= around.min():
            lows.append(i)
    return highs, lows


def support_resistance(high, low, close, volume=None, lookback=250,
                       cluster_pct=0.02) -> dict:
    """支撑压力位识别。

    步骤：(1) 取窗口内左右各 3 根确认的摆动高低点作为候选；
    (2) 用成交量给候选加权（无 volume 时权重为 1）；
    (3) 按 ``cluster_pct`` 一维聚类（以簇首价为基准，相邻价差 < cluster_pct*price 归簇），
    簇内加权平均得价位，簇权重 = 成员数 × 成员平均成交量；
    (4) ``strength`` 归一化到 0~1；``supports`` 由高到低、``resistances`` 由低到高。

    ``position``：现价距最近支撑/压力 2% 以内 → ``near_support`` / ``near_resistance``，否则 ``mid``。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    out = {"supports": [], "resistances": [], "nearest_support": None,
           "nearest_resistance": None, "position": "mid"}
    m = min(len(h), len(l), len(c))
    if m < 20:
        return out
    k = int(min(max(int(lookback), 20), m))
    hh = h.iloc[-k:].to_numpy(dtype=float)
    ll = l.iloc[-k:].to_numpy(dtype=float)
    cc = c.iloc[-k:].to_numpy(dtype=float)
    if volume is not None:
        vv = _as_series(volume, "volume").iloc[-k:].to_numpy(dtype=float)
        vv = np.nan_to_num(vv, nan=0.0)
        if not np.isfinite(vv).any() or vv.max() <= 0:
            vv = np.ones(k)
    else:
        vv = np.ones(k)
    price_now = float(cc[-1])
    if not np.isfinite(price_now):
        return out
    hi_idx, _ = _swing_points(hh, 3)     # 摆动高点（用最高价序列）
    _, lo_idx = _swing_points(ll, 3)     # 摆动低点（用最低价序列）
    cands = [(float(hh[i]), float(vv[i])) for i in hi_idx]
    cands += [(float(ll[i]), float(vv[i])) for i in lo_idx]
    if not cands:
        cands = [(float(np.nanmax(hh)), 1.0), (float(np.nanmin(ll)), 1.0)]
    cands = [t for t in cands if np.isfinite(t[0]) and t[0] > 0]
    if not cands:
        return out
    cands.sort(key=lambda t: t[0])
    clusters: list[list[tuple[float, float]]] = [[cands[0]]]
    for p, w in cands[1:]:
        anchor = clusters[-1][0][0]
        if (p - anchor) <= float(cluster_pct) * max(p, anchor):
            clusters[-1].append((p, w))
        else:
            clusters.append([(p, w)])
    levels = []
    for cl in clusters:
        ps = np.array([t[0] for t in cl], dtype=float)
        ws = np.array([t[1] for t in cl], dtype=float)
        if ws.sum() <= 0:
            ws = np.ones_like(ws)
        price = float(np.dot(ps, ws) / ws.sum())
        weight = float(len(cl) * ws.mean())
        levels.append({"price": price, "weight": weight})
    wmax = max(lv["weight"] for lv in levels) or 1.0
    for lv in levels:
        lv["strength"] = round(float(lv["weight"]) / wmax, 4)
    supports = sorted([lv for lv in levels if lv["price"] <= price_now],
                      key=lambda d: -d["price"])
    resistances = sorted([lv for lv in levels if lv["price"] > price_now],
                         key=lambda d: d["price"])
    # 兜底：保证两侧至少各有一条可参考的价位
    if not supports:
        lo_p = float(np.nanmin(ll))
        supports = [{"price": lo_p, "strength": 0.15, "fallback": True}]
    if not resistances:
        hi_p = float(np.nanmax(hh))
        if hi_p <= price_now:
            hi_p = price_now          # 现价即窗口最高价：该高点本身作为突破确认位
        resistances = [{"price": hi_p, "strength": 0.15, "fallback": True}]
    for lv in supports + resistances:
        lv.pop("weight", None)
        lv["price"] = round(float(lv["price"]), 4)
        if "strength" in lv:
            lv["strength"] = float(lv["strength"])
    ns = float(supports[0]["price"])
    nr = float(resistances[0]["price"])
    d_sup = (price_now - ns) / price_now if price_now else _NAN
    d_res = (nr - price_now) / price_now if price_now else _NAN
    position = "mid"
    if np.isfinite(d_sup) and d_sup <= 0.02 and (not np.isfinite(d_res) or d_sup <= d_res):
        position = "near_support"
    elif np.isfinite(d_res) and d_res <= 0.02:
        position = "near_resistance"
    out.update({"supports": supports, "resistances": resistances,
                "nearest_support": ns, "nearest_resistance": nr, "position": position,
                "price": round(price_now, 4)})
    return out


def channel_fit(high, low, close, lookback=120) -> dict:
    """线性回归通道。

    对窗口内的高点做上轨回归、低点做下轨回归（``np.polyfit`` 一次），
    ``channel``：两轨斜率同号且显著（``|slope|/price > 0.0002`` 每根）→ ``up``/``down``，否则 ``flat``。
    ``position``：现价在通道内的相对位置（0~1）；``break``：突破方向。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    out = {"upper_slope": _NAN, "lower_slope": _NAN, "upper_intercept": _NAN,
           "lower_intercept": _NAN, "upper_now": _NAN, "lower_now": _NAN,
           "channel": "flat", "position": _NAN, "width_pct": _NAN, "break": "none"}
    m = min(len(h), len(l), len(c))
    if m < 10:
        return out
    k = int(min(max(int(lookback), 10), m))
    hh = h.iloc[-k:].to_numpy(dtype=float)
    ll = l.iloc[-k:].to_numpy(dtype=float)
    cc = c.iloc[-k:].to_numpy(dtype=float)
    ok = np.isfinite(hh) & np.isfinite(ll) & np.isfinite(cc)
    if ok.sum() < 10:
        return out
    x = np.arange(k, dtype=float)[ok]
    try:
        su, iu = np.polyfit(x, hh[ok], 1)
        sl, il = np.polyfit(x, ll[ok], 1)
    except Exception as exc:  # noqa: BLE001
        log.debug("channel_fit polyfit 失败: %s", exc)
        return out
    x_now = float(k - 1)
    upper_now = float(su * x_now + iu)
    lower_now = float(sl * x_now + il)
    price_now = float(cc[ok][-1])
    ref = price_now if price_now else 1.0
    thr = 0.0002 * abs(ref)
    channel = "flat"
    if su > 0 and sl > 0 and abs(su) > thr and abs(sl) > thr:
        channel = "up"
    elif su < 0 and sl < 0 and abs(su) > thr and abs(sl) > thr:
        channel = "down"
    width = upper_now - lower_now
    position = float(np.clip((price_now - lower_now) / width, 0.0, 1.0)) if width > 0 else _NAN
    brk = "none"
    if price_now > upper_now:
        brk = "up"
    elif price_now < lower_now:
        brk = "down"
    return {"upper_slope": float(su), "lower_slope": float(sl),
            "upper_intercept": float(iu), "lower_intercept": float(il),
            "upper_now": upper_now, "lower_now": lower_now, "channel": channel,
            "position": position,
            "width_pct": float(width / ref * 100.0) if ref else _NAN, "break": brk}


def fib_levels(high, low, lookback=250) -> dict:
    """斐波那契回撤 / 扩展位。

    以窗口最高价 ``H`` 与最低价 ``L`` 为端点；最低点在最高点之前 → 上升波段（``up``）。
    上升波段回撤位 ``H-(H-L)*r``，下降波段反弹位 ``L+(H-L)*r``；
    扩展位（1.272/1.618/2.618）上升波段 ``L+(H-L)*k``、下降波段 ``H-(H-L)*k``。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    ratios = tuple(config.INDICATOR_PARAMS.get("fib_ratios", (0.191, 0.236, 0.382, 0.5, 0.618, 0.786, 0.809)))
    exts = (1.272, 1.618, 2.618)
    out = {"high": _NAN, "low": _NAN, "direction": "unknown", "levels": {},
           "retracement": {}, "extension": {}}
    m = min(len(h), len(l))
    if m < 5:
        return out
    k = int(min(max(int(lookback), 5), m))
    hh = h.iloc[-k:].to_numpy(dtype=float)
    ll = l.iloc[-k:].to_numpy(dtype=float)
    if not np.isfinite(hh).any() or not np.isfinite(ll).any():
        return out
    hi_p = float(np.nanmax(hh))
    lo_p = float(np.nanmin(ll))
    if not (np.isfinite(hi_p) and np.isfinite(lo_p)) or hi_p <= lo_p:
        return out
    i_hi = int(np.nanargmax(hh))
    i_lo = int(np.nanargmin(ll))
    direction = "up" if i_lo < i_hi else "down"
    span = hi_p - lo_p
    levels, ext = {}, {}
    for r in ratios:
        if direction == "up":
            levels["%.3f" % r] = round(hi_p - span * r, 4)
        else:
            levels["%.3f" % r] = round(lo_p + span * r, 4)
    for kk in exts:
        if direction == "up":
            ext["%.3f" % kk] = round(lo_p + span * kk, 4)
        else:
            ext["%.3f" % kk] = round(hi_p - span * kk, 4)
    return {"high": round(hi_p, 4), "low": round(lo_p, 4), "direction": direction,
            "levels": dict(levels), "retracement": dict(levels), "extension": ext}


def golden_pit(high, low, close, volume=None) -> dict:
    """黄金坑识别。

    判定条件（缺一不可）：

    1. 先有一波下跌：自区间高点（坑口）回撤 **≥8%**；
    2. 下探足够快：坑底出现在高点之后 **≤15 根** 内；
    3. **跌破前低**（假破位）：坑底低于此前 60 根的最低价（允许 0.5% 误差）；
    4. 放量收复：坑底后 **≤15 根** 内收盘价回到 **坑口价的 90% 以上**；
    5. 收复期均量 > 下跌期均量 × **1.3**（需传 ``volume``；不传则该条不参与，score 打 0.85 折）。

    Parameters
    ----------
    volume : pd.Series | np.ndarray | None
        成交量。为满足"量能 > 1.3 倍"这条判定而增加的**可选**参数，
        不传时其余四条判定照常工作（向后兼容 ``golden_pit(high, low, close)`` 调用）。

    Returns
    -------
    dict
        ``{"detected","pit_low","pit_depth_pct","recover_bars","score", ...}``
    """
    out = {"detected": False, "pit_low": _NAN, "pit_depth_pct": _NAN,
           "recover_bars": 0, "score": 0.0}
    lookback = 120                                     # 回溯窗口（保持文档签名不变）
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    m = min(len(h), len(l), len(c))
    if m < 40:
        return out
    k = int(min(max(int(lookback), 40), m))
    hh = h.iloc[-k:].to_numpy(dtype=float)
    ll = l.iloc[-k:].to_numpy(dtype=float)
    cc = c.iloc[-k:].to_numpy(dtype=float)
    if not (np.isfinite(hh).all() and np.isfinite(ll).all() and np.isfinite(cc).all()):
        return out
    use_vol = volume is not None
    if use_vol:
        vv = _as_series(volume, "volume")
        vv = vv.iloc[-k:].to_numpy(dtype=float) if len(vv) >= k else np.full(k, np.nan)
        use_vol = bool(np.isfinite(vv).all() and vv.sum() > 0)
    best = None
    for t in range(20, max(k - 5, 21)):
        mouth = float(hh[t - 20:t + 1].max())          # 坑口：起跌前的局部高点
        if mouth <= 0 or hh[t] < mouth * 0.999:        # 必须站在近 20 根的最高点
            continue
        win_end = min(k, t + 30)
        if win_end - t < 3:
            continue
        j = int(np.argmin(ll[t:win_end]))
        pit_low = float(ll[t + j])
        depth = (mouth - pit_low) / mouth * 100.0
        if depth < 8.0 or j > 15:                      # 跌幅 ≥8% 且下探 ≤15 根
            continue
        t_low = t + j
        prior_low = float(ll[max(0, t - 60):t].min()) if t > 0 else pit_low
        if pit_low > prior_low * 1.005:                # 必须跌破前低（允许 0.5% 误差）
            continue
        target = mouth * 0.90                          # 收复到坑口 90% 以上
        rec = None
        for q in range(t_low, min(k, t_low + 16)):
            if cc[q] >= target:
                rec = q
                break
        if rec is None:
            continue
        vol_ratio = _NAN
        if use_vol:
            dec_vol = float(np.mean(vv[t:t_low + 1])) if t_low > t else float(vv[t])
            rec_vol = float(np.mean(vv[t_low:rec + 1])) if rec > t_low else float(vv[t_low])
            vol_ratio = rec_vol / dec_vol if dec_vol > 0 else _NAN
            if not (np.isfinite(vol_ratio) and vol_ratio > 1.3):
                continue                               # 量能不足 -> 不是有效黄金坑
        amp = float(np.mean(hh[t_low:rec + 1] - ll[t_low:rec + 1])) if rec > t_low else 0.0
        score = float(np.clip(0.45 * min(depth / 20.0, 1.0)
                              + 0.35 * (1.0 - min((rec - t_low) / 15.0, 1.0))
                              + 0.20 * min(amp / max(mouth * 0.05, 1e-6), 1.0), 0.0, 1.0))
        if not use_vol:
            score *= 0.85                              # 无量能确认，置信度打折
        cand = {"detected": True, "pit_low": round(pit_low, 4),
                "pit_depth_pct": round(depth, 2), "recover_bars": int(rec - t_low),
                "score": round(score, 4), "mouth": round(mouth, 4),
                "vol_ratio": (round(float(vol_ratio), 3) if np.isfinite(vol_ratio) else None),
                "decline_bars": int(j)}
        if best is None or cand["score"] >= best["score"]:
            best = cand
    if best is None:
        return out
    out.update(best)
    return out


def gap_stats(open_, high, low, close) -> dict:
    """跳空缺口统计。

    向上跳空：``low > REF(high,1)``；向下跳空：``high < REF(low,1)``。
    ``size_pct`` 为缺口幅度（%），``filled`` 表示后续是否被回补（收盘价重回缺口区）。
    """
    h = _as_series(high, "high")
    l = _as_series(low, "low")
    c = _as_series(close, "close")
    dates = None
    for src in (high, low, close):
        if isinstance(src, pd.Series) and isinstance(src.index, pd.DatetimeIndex):
            dates = src.index
            break
    out = {"gaps": [], "up_gap_count": 0, "down_gap_count": 0, "last_gap": None}
    m = min(len(h), len(l), len(c))
    if m < 3:
        return out
    hv = h.iloc[:m].to_numpy(dtype=float)
    lv = l.iloc[:m].to_numpy(dtype=float)
    cv = c.iloc[:m].to_numpy(dtype=float)
    gaps = []
    for i in range(1, m):
        d = None
        typ = None
        size = _NAN
        if lv[i] > hv[i - 1]:
            typ, size = "up", (lv[i] - hv[i - 1]) / hv[i - 1] * 100.0
        elif hv[i] < lv[i - 1]:
            typ, size = "down", (lv[i - 1] - hv[i]) / lv[i - 1] * 100.0
        if typ is None:
            continue
        if typ == "up":
            filled = bool((lv[i + 1:] <= hv[i - 1]).any()) if i + 1 < m else False
        else:
            filled = bool((hv[i + 1:] >= lv[i - 1]).any()) if i + 1 < m else False
        d = {"date": dates[i] if dates is not None and i < len(dates) else i,
             "type": typ, "size_pct": round(float(size), 3), "filled": filled}
        # 连续跳空合并为同一缺口（同向相邻只保留首个）
        if gaps and gaps[-1]["type"] == typ and not gaps[-1]["filled"]:
            gaps[-1]["filled"] = filled
            gaps[-1]["size_pct"] = round(float(gaps[-1]["size_pct"]) + float(size), 3)
            continue
        gaps.append(d)
    out["gaps"] = gaps
    out["up_gap_count"] = sum(1 for g in gaps if g["type"] == "up")
    out["down_gap_count"] = sum(1 for g in gaps if g["type"] == "down")
    out["last_gap"] = gaps[-1] if gaps else None
    return out


def trend_strength(close, high=None, low=None, n=20) -> dict:
    """趋势强度（线性回归口径）。

    ``slope``：对近 ``n`` 根收盘价做一次线性回归，换算成每日涨跌幅（%）；
    ``r2``：拟合优度 0~1；``direction``：``r2>0.5`` 且斜率非零时给出 ±1，否则 0；
    ``adx_like``：0~100 的趋势强度（效率比与 r2 的加权），可近似替代 ADX 描述"趋势有多强"。
    """
    s = _as_series(close, "close")
    out = {"slope": _NAN, "r2": _NAN, "adx_like": _NAN, "direction": 0}
    n = max(int(n), 3)
    if len(s) < n:
        return out
    y = s.iloc[-n:].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        return out
    x = np.arange(n, dtype=float)
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:  # noqa: BLE001
        return out
    fit = slope * x + intercept
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    base = float(np.mean(y))
    slope_pct = float(slope) / base * 100.0 if base else _NAN
    diff = np.abs(np.diff(y))
    denom = float(diff.sum())
    er = abs(y[-1] - y[0]) / denom if denom > 0 else 0.0       # Kaufman 效率比
    adx_like = float(np.clip(100.0 * (0.6 * er + 0.4 * max(r2, 0.0)), 0.0, 100.0))
    direction = 0
    if r2 > 0.5 and np.isfinite(slope_pct) and abs(slope_pct) > 1e-9:
        direction = 1 if slope_pct > 0 else -1
    return {"slope": round(slope_pct, 5), "r2": round(float(max(r2, 0.0)), 5),
            "adx_like": round(adx_like, 3), "direction": int(direction)}


def limit_status(close, prev_close, code="600519", name="", high=None, low=None,
                 open_=None) -> pd.Series:
    """逐日涨跌停状态。

    返回值为 ``"limit_up"`` / ``"limit_down"`` / ``"one_word"``（一字板） /
    ``"broken"``（炸板：盘中触及涨停但收盘未封住） / ``""``。

    涨停幅度取 :func:`core.utils.limit_up_pct`（主板 9.8% / 创业科创 19.5% /
    北交所 29.5% / ST 4.8%），涨停价按 ``round(prev_close*(1+pct),2)`` 计算，
    并以 0.01 元为容差。

    Parameters
    ----------
    high, low, open_ : 可选
        ``"one_word"``（一字板）与 ``"broken"``（炸板）必须知道当日是否**离开过**涨停价，
        因此需要最高/最低/开盘价；只给 ``close``/``prev_close`` 时这两类无法判定，
        返回集合退化为 ``{"limit_up","limit_down",""}``（文档签名
        ``limit_status(close, prev_close, code, name)`` 仍可正常调用）。
    """
    c = _as_series(close, "close")
    pc = _as_series(prev_close, "prev_close")
    if c.empty:
        return pd.Series([], dtype=object)
    n = max(len(c), len(pc))

    def _pad(x):
        if x is None:
            return None
        s = _as_series(x, "col")
        if len(s) != n:
            return None
        return s.to_numpy(dtype=float)

    cv = np.full(n, np.nan)
    pv = np.full(n, np.nan)
    cv[:len(c)] = c.to_numpy(dtype=float)
    pv[:len(pc)] = pc.to_numpy(dtype=float)
    hv, lv, ov = _pad(high), _pad(low), _pad(open_)
    lim = float(limit_up_pct(code, name)) / 100.0
    vals = []
    for i in range(n):
        if not (np.isfinite(cv[i]) and np.isfinite(pv[i])) or pv[i] <= 0:
            vals.append("")
            continue
        up_px = round(pv[i] * (1.0 + lim), 2)
        dn_px = round(pv[i] * (1.0 - lim), 2)
        if cv[i] >= up_px - 0.005:                                   # 收在涨停价
            if hv is None or lv is None or ov is None:
                vals.append("limit_up")
            else:
                one_word = (np.isfinite(lv[i]) and np.isfinite(ov[i])
                            and lv[i] >= up_px - 0.01 and ov[i] >= up_px - 0.01)
                vals.append("one_word" if one_word else "limit_up")
        elif cv[i] <= dn_px + 0.005:
            vals.append("limit_down")
        elif hv is not None and np.isfinite(hv[i]) and hv[i] >= up_px - 0.01:
            vals.append("broken")                                    # 触板未封住
        else:
            vals.append("")
    return pd.Series(vals, index=c.index)


def limit_status_full(open_, high, low, close, prev_close, code="600519", name="") -> pd.Series:
    """带一字板/炸板判定的完整调用形式（等价于 :func:`limit_status` 传入 OHLC）。"""
    return limit_status(close, prev_close, code=code, name=name,
                        high=high, low=low, open_=open_)


# --------------------------------------------------------------------------- #
# 六、K 线重构（砖图 / 等价 K 线 / 重采样）
# --------------------------------------------------------------------------- #
def renko(close, high=None, low=None, brick_pct=0.02) -> pd.DataFrame:
    """等比砖图 Renko（固定百分比砖）。

    每块砖的尺寸 = 当前基准价 × ``brick_pct``（价格越高砖越大，"等比 K 线"思想）；
    不足一砖的波动忽略。返回 ``DataFrame[date,open,high,low,close,brick_dir]``，
    ``brick_dir`` ∈ {+1 阳砖, -1 阴砖}。

    第一个参数也接受**整个 OHLCV DataFrame**（``renko(df)``），此时自动做列名容错
    并取用其中的 close/high/low；日期取 ``date`` 列，否则用行号。
    """
    cols = ["date", "open", "high", "low", "close", "brick_dir"]
    if isinstance(close, pd.DataFrame):
        d = _ensure_frame(close)
        if "close" not in d.columns:
            return pd.DataFrame(columns=cols)
        if high is None and "high" in d.columns:
            high = d["high"]
        if low is None and "low" in d.columns:
            low = d["low"]
        if "date" in d.columns:
            d = d.set_index("date")
        close = d["close"]
    c = _as_series(close, "close")
    if c.empty:
        return pd.DataFrame(columns=cols)
    h = _as_series(high, "high") if high is not None else c
    l = _as_series(low, "low") if low is not None else c
    is_dt = isinstance(c.index, pd.DatetimeIndex)
    dates = c.index if is_dt else pd.RangeIndex(len(c))
    cv = c.to_numpy(dtype=float)
    hv = h.to_numpy(dtype=float) if len(h) == len(c) else cv.copy()
    lv = l.to_numpy(dtype=float) if len(l) == len(c) else cv.copy()
    brick_pct = float(brick_pct) if float(brick_pct) > 0 else 0.02
    rows = []
    base = float(cv[0])
    if not np.isfinite(base):
        return pd.DataFrame(columns=cols)
    for i in range(len(cv)):
        px = cv[i]
        if not np.isfinite(px):
            continue
        guard = 0
        while guard < 200:
            guard += 1
            size = base * brick_pct
            if size <= 0:
                break
            if px >= base + size:
                top = base + size
                rows.append({"date": dates[i], "open": round(base, 4), "high": round(top, 4),
                             "low": round(base, 4), "close": round(top, 4), "brick_dir": 1})
                base = top
            elif px <= base - size:
                bot = base - size
                rows.append({"date": dates[i], "open": round(base, 4), "high": round(base, 4),
                             "low": round(bot, 4), "close": round(bot, 4), "brick_dir": -1})
                base = bot
            else:
                break
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty and is_dt:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:  # noqa: BLE001
            pass
    return df


def equal_kline(df, mode="price", n_bars=60, box_pct=0.03, box_vol_ratio=1.5) -> pd.DataFrame:
    """等价 K 线 / 等量 K 线。

    * ``mode="price"``（等价 K 线）：价格振幅满一箱（箱高 = 区间中价 × ``box_pct``）即收一根；
    * ``mode="volume"``（等量 K 线）：成交量累计满一箱（箱量 = ``box_vol_ratio`` × 近 20 日均量）即收一根。

    ``n_bars``：参与聚合的**最近原始 K 线根数**（不足则用全部）。
    返回 ``DataFrame[bar_index,open,high,low,close,volume,amount,start_date,end_date,n_orig]``。
    """
    cols = ["bar_index", "open", "high", "low", "close", "volume", "amount",
            "start_date", "end_date", "n_orig"]
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns:
        return pd.DataFrame(columns=cols)
    n_all = len(d)
    k = int(min(max(int(n_bars), 2), n_all)) if n_bars else n_all
    d = d.iloc[-k:].reset_index(drop=True)
    o = _cols(d, "open").to_numpy(dtype=float)
    h = _cols(d, "high").to_numpy(dtype=float)
    l = _cols(d, "low").to_numpy(dtype=float)
    c = _cols(d, "close").to_numpy(dtype=float)
    v = np.nan_to_num(_cols(d, "volume", default=0.0).to_numpy(dtype=float), nan=0.0)
    a = np.nan_to_num(_cols(d, "amount", default=0.0).to_numpy(dtype=float), nan=0.0)
    o = np.where(np.isfinite(o), o, c)
    h = np.where(np.isfinite(h), h, c)
    l = np.where(np.isfinite(l), l, c)
    if not np.isfinite(c).any():
        return pd.DataFrame(columns=cols)
    dates = d["date"].to_numpy() if "date" in d.columns else np.arange(k)
    vma = pd.Series(v).rolling(20, min_periods=1).mean().to_numpy(dtype=float)
    rows = []
    i = 0
    bar = 0
    while i < k:
        start = i
        o0 = o[i]
        hi, lo = h[i], l[i]
        vol, amt = v[i], a[i]
        j = i
        if mode == "volume":
            box = float(box_vol_ratio) * (vma[i] if np.isfinite(vma[i]) and vma[i] > 0 else 1.0)
            acc = vol
            while j + 1 < k and acc < box:
                j += 1
                hi = max(hi, h[j])
                lo = min(lo, l[j])
                vol += v[j]
                amt += a[j]
                acc += v[j]
        else:
            pct = float(box_pct) if float(box_pct) > 0 else 0.03
            while j + 1 < k:
                j += 1
                hi = max(hi, h[j])
                lo = min(lo, l[j])
                vol += v[j]
                amt += a[j]
                if hi - lo >= (hi + lo) / 2.0 * pct:
                    break
        rows.append({"bar_index": bar, "open": round(float(o0), 4), "high": round(float(hi), 4),
                     "low": round(float(lo), 4), "close": round(float(c[j]), 4),
                     "volume": float(vol), "amount": float(amt),
                     "start_date": dates[start], "end_date": dates[j],
                     "n_orig": int(j - start + 1)})
        bar += 1
        i = j + 1
    out = pd.DataFrame(rows, columns=cols)
    for col in ("start_date", "end_date"):
        try:
            out[col] = pd.to_datetime(out[col])
        except Exception:  # noqa: BLE001
            pass
    return out


def resample_kline(df, rule="W") -> pd.DataFrame:
    """日线 -> 周线 / 月线。

    ``rule`` ∈ {"W","M"}（也接受 pandas 原生别名）。使用 ``W-FRI`` / ``ME``
    以兼容 pandas 3.0（``"W"`` 与 ``"M"`` 已废弃）；OHLC 取 first/max/min/last，
    ``volume``/``amount``/``turnover`` 求和，并附带 ``pct_chg`` 列。
    """
    d = _ensure_frame(df)
    if d.empty or "date" not in d.columns or "close" not in d.columns:
        return pd.DataFrame()
    rule_map = {"W": "W-FRI", "WEEK": "W-FRI", "WEEKLY": "W-FRI", "M": "ME", "MONTH": "ME",
                "MONTHLY": "ME", "Q": "QE", "Y": "YE", "A": "YE", "D": "D", "DAY": "D"}
    key = str(rule).strip().upper()
    freq = rule_map.get(key, str(rule))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in d.columns:
        agg["volume"] = "sum"
    if "amount" in d.columns:
        agg["amount"] = "sum"
    if "turnover" in d.columns:
        agg["turnover"] = "sum"
    try:
        sub = d.set_index("date")
        out = sub.resample(freq).agg(agg)
        out = out.dropna(subset=["close"]).reset_index()
    except Exception as exc:  # noqa: BLE001
        log.debug("resample_kline(%s) 失败: %s", rule, exc)
        return pd.DataFrame()
    if out.empty:
        return out
    try:
        out["pct_chg"] = (out["close"] / out["close"].shift(1) - 1.0) * 100.0
    except Exception:  # noqa: BLE001
        out["pct_chg"] = _NAN
    return out


# --------------------------------------------------------------------------- #
# 七、汇总矩阵
# --------------------------------------------------------------------------- #
def indicator_matrix(df, params=None) -> pd.DataFrame:
    """一次性计算全部指标，返回与输入同长度的 DataFrame（列名简洁英文）。

    必须列包括 ``ma5/ma20``、``macd_dif/dea/hist``、``k/d/j``、``rsi6/rsi14``、
    ``boll_mid/up/low/pctb``、``ene_up/low``、``sar/sar_trend``、``dkx``、``bbi``、
    ``expma1/2``、``bias6/12/24``、``cci``、``pdi/mdi/adx``、``wr6/wr10``、``dma/ama``、
    ``trix/trma``、``ar/br``、``vr``、``obv/obv_ma``、``asi/asima``、``emv/emv_ma``、
    ``wvad``、``roc``、``lon``、``psy``、``mtm``、``cr``、``vwap``、
    ``vol_ma5/vol_ma20``、``atr14``、``amp_pct``、``turnover_ma5``（共 56 列以上）。

    ``params`` 缺省使用 :data:`core.config.INDICATOR_PARAMS`。
    """
    p = dict(config.INDICATOR_PARAMS)
    if params:
        try:
            p.update(params)
        except Exception:  # noqa: BLE001
            pass
    d = _ensure_frame(df)
    if d.empty or "close" not in d.columns:
        return pd.DataFrame(index=d.index if isinstance(d, pd.DataFrame) else None)
    o = _cols(d, "open")
    h = _cols(d, "high")
    l = _cols(d, "low")
    c = _cols(d, "close")
    v = _cols(d, "volume", default=0.0)
    amt = _cols(d, "amount", default=_NAN)
    out = pd.DataFrame(index=d.index)
    try:
        # ---- 均线族 ------------------------------------------------------ #
        for n in p.get("ma", (5, 10, 20, 30, 60, 120, 250)):
            out["ma%d" % int(n)] = ma(c, n)
        for n in p.get("ema", (12, 26, 50)):
            out["ema%d" % int(n)] = ema(c, n)
        ex = expma(c, *tuple(p.get("expma", (12, 50)))[:2])
        out["expma1"], out["expma2"] = ex["expma1"], ex["expma2"]
        out["bbi"] = bbi(c, *tuple(p.get("bbi", (3, 6, 12, 24)))[:4])
        # ---- MACD / 趋势 ------------------------------------------------- #
        mp = tuple(p.get("macd", (12, 26, 9)))[:3]
        mc = macd(c, *mp)
        out["macd_dif"], out["macd_dea"], out["macd_hist"] = mc["dif"], mc["dea"], mc["macd"]
        out["macd_cross"] = _cross_code(mc["cross"])
        dp = tuple(p.get("dkx", (10, 10)))[:2]
        dk = dkx(o, h, l, c, *dp)
        out["dkx_mid"], out["dkx"] = dk["mid"], dk["dkx"]
        out["dkx_trend_up"] = 1.0 if dk["trend_up"] else 0.0
        sp = sar(h, l)
        out["sar"], out["sar_trend"] = sp["sar"], sp["trend"].astype(float)
        out["sar_flip"] = sp["flip"].astype(float)
        trp = tuple(p.get("trix", (12, 20)))[:2]
        tx = trix(c, *trp)
        out["trix"], out["trma"] = tx["trix"], tx["trma"]
        dmp = tuple(p.get("dma", (10, 50, 10)))[:3]
        dm = dma(c, *dmp)
        out["dma"], out["ama"] = dm["dma"], dm["ama"]
        lo = lon(c, *tuple(p.get("lon", (10,)))[:1])
        out["lon"], out["lon_ma"] = lo["lon"], lo["lon_ma"]
        rp = tuple(p.get("roc", (12, 6)))[:2]
        rc = roc(c, *rp)
        out["roc"], out["roc_ma"] = rc["roc"], rc["roc_ma"]
        mtp = tuple(p.get("mtm", (12, 6)))[:2]
        mt = mtm(c, *mtp)
        out["mtm"], out["mtm_ma"] = mt["mtm"], mt["mtm_ma"]
        # ---- 摆动族 ------------------------------------------------------ #
        kp = tuple(p.get("kdj", (9, 3, 3)))[:3]
        kj = kdj(h, l, c, *kp)
        out["k"], out["d"], out["j"] = kj["k"], kj["d"], kj["j"]
        out["kdj_cross"] = _cross_code(kj["cross"])
        for n in p.get("rsi", (6, 12, 14, 24)):
            out["rsi%d" % int(n)] = rsi(c, n)
        dmp = tuple(p.get("dmi", (14, 6)))[:2]
        di = dmi(h, l, c, *dmp)
        out["pdi"], out["mdi"], out["adx"], out["adxr"] = \
            di["pdi"], di["mdi"], di["adx"], di["adxr"]
        out["dmi_signal"] = _signal_code(di["signal"])
        for key, ser in bias(c, tuple(p.get("bias", (6, 12, 24)))).items():
            out[key] = ser
        out["cci"] = cci(h, l, c, *tuple(p.get("cci", (14,)))[:1])
        for key, ser in wr(h, l, c, tuple(p.get("wr", (6, 10)))[:2]).items():
            out[key] = ser
        out["psy"] = psy(c, *tuple(p.get("psy", (12,)))[:1])
        # ---- 通道族 ------------------------------------------------------ #
        bp = tuple(p.get("boll", (20, 2.0)))[:2]
        bo = boll(c, *bp)
        out["boll_mid"], out["boll_up"], out["boll_low"] = bo["mid"], bo["upper"], bo["lower"]
        out["boll_pctb"], out["boll_width"] = bo["pct_b"], bo["width"]
        out["boll_width_pct"] = bo["bandwidth_pct"]
        ep = tuple(p.get("ene", (25, 11, 9)))[:3]
        en = ene(c, *ep)
        out["ene_up"], out["ene_low"], out["ene_mid"] = en["upper"], en["lower"], en["mid"]
        # ---- 量能族 ------------------------------------------------------ #
        vp = tuple(p.get("brar", (26,)))[:1]
        bra = brar(o, h, l, c, *vp)
        out["ar"], out["br"] = bra["ar"], bra["br"]
        out["vr"] = vr(c, v, *tuple(p.get("vr", (26,)))[:1])
        ob = obv_ma(c, v, *tuple(p.get("obv", (30,)))[:1])
        out["obv"], out["obv_ma"] = ob["obv"], ob["obv_ma"]
        asp = tuple(p.get("asi", (26,)))[:1]
        asd = asi(o, h, l, c, *asp)
        out["asi"], out["asima"] = asd["asi"], asd["asima"]
        emp = tuple(p.get("emv", (14, 9)))[:2]
        em = emv(h, l, v, *emp)
        out["emv"], out["emv_ma"] = em["emv"], em["emv_ma"]
        wv = wvad(o, h, l, c, v, *tuple(p.get("wvad", (24,)))[:1])
        out["wvad"], out["wvad_ma"] = wv["wvad"], wv["wvad_ma"]
        crp = tuple(p.get("cr", (26,)))[:1]
        crd = cr(h, l, c, *crp)
        out["cr"], out["cr_ma1"], out["cr_ma2"], out["cr_ma3"] = \
            crd["cr"], crd["ma1"], crd["ma2"], crd["ma3"]
        out["vol_price_corr"] = vol_price_corr(c, v, 20)
        am = amount_ma(amt, (5, 20))
        out["amount_ma5"], out["amount_ma20"] = am["amount_ma5"], am["amount_ma20"]
        # ---- 成本 / 量能均线 --------------------------------------------- #
        vw = vwap_cost(h, l, c, v, int(p.get("vwap_window", 60)))
        out["vwap"] = vw["vwap"]
        out["vol_ma5"] = ma(v, 5)
        out["vol_ma20"] = ma(v, 20)
        if "turnover" in d.columns:
            out["turnover_ma5"] = ma(_as_series(d["turnover"], "turnover"), 5)
        else:
            out["turnover_ma5"] = pd.Series(np.full(len(d), _NAN), index=d.index)
        # ---- 波动 -------------------------------------------------------- #
        pc = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        out["atr14"] = smma(tr, 14)
        out["amp_pct"] = pd.Series(_safe_div((h - l).to_numpy(dtype=float),
                                             pc.to_numpy(dtype=float)) * 100.0, index=d.index)
    except Exception as exc:  # noqa: BLE001
        log.debug("indicator_matrix 部分指标失败: %s", exc)
    # 去掉全 NaN 的空列（例如无 volume 时的量能族）
    try:
        out = out.loc[:, ~out.isna().all(axis=0)]
    except Exception:  # noqa: BLE001
        pass
    return out


def _cross_code(cross: str) -> float:
    """交叉状态 -> 数值（1 金叉 / -1 死叉 / 0 无），便于进矩阵与 ML 特征。"""
    return 1.0 if cross == "gold" else (-1.0 if cross == "dead" else 0.0)


def _signal_code(signal: str) -> float:
    """DMI 信号 -> 数值（bull 1 / bear -1 / weak 0 / range 0）。"""
    return {"bull": 1.0, "bear": -1.0}.get(str(signal), 0.0)


# --------------------------------------------------------------------------- #
# 自测（离线合成数据 + 数值校验）
# --------------------------------------------------------------------------- #
def _iter_numeric(obj):
    """递归产出对象内的所有数值 ndarray，供"无 ±inf"检查。"""
    if isinstance(obj, pd.DataFrame):
        for col in obj.columns:
            if pd.api.types.is_numeric_dtype(obj[col]):
                yield str(col), obj[col].to_numpy(dtype=float, na_value=np.nan)
    elif isinstance(obj, pd.Series):
        yield "series", obj.to_numpy(dtype=float, na_value=np.nan)
    elif isinstance(obj, np.ndarray):
        yield "ndarray", obj.astype(float, copy=False)
    elif isinstance(obj, dict):
        for k, val in obj.items():
            for name, arr in _iter_numeric(val):
                yield "%s.%s" % (k, name), arr
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            for name, arr in _iter_numeric(val):
                yield "[%d].%s" % (i, name), arr
    elif isinstance(obj, (int, float, np.floating, np.integer)) and not isinstance(obj, bool):
        yield "scalar", np.asarray([float(obj)])


def _self_test() -> None:  # pragma: no cover - 自测块
    try:
        from .selftest_data import make_ohlcv
    except ImportError:
        from selftest_data import make_ohlcv  # type: ignore

    df = make_ohlcv(600, seed=1)
    df_backup = df.copy(deep=True)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v, amt = df["volume"], df["amount"]
    cdi = df.set_index("date")["close"]          # 带 DatetimeIndex 的收盘价（测日期路径）
    hdi, ldi = df.set_index("date")["high"], df.set_index("date")["low"]

    # ---------- 1. 基础口径校验 ---------- #
    m5 = ma(c, 5)
    assert np.allclose(m5.to_numpy(), c.rolling(5).mean().to_numpy(), equal_nan=True), "ma5 口径不一致"
    assert len(m5) == len(c) and m5.index.equals(c.index), "ma 长度/索引不一致"
    assert np.isnan(m5.iloc[:4]).all() and np.isfinite(m5.iloc[4]), "ma 头部应为 NaN"
    # smma 与通达信 SMA(X,N,1) 展开式逐项一致
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sm = smma(x, 3)
    ref2 = (3.0 + 2.0 * ((2.0 + 2.0 * 1.0) / 3.0)) / 3.0        # 17/9
    assert abs(sm.iloc[2] - ref2) < 1e-12, ("smma 递归口径错误", sm.tolist())
    assert np.isnan(sm.iloc[0]) and np.isnan(sm.iloc[1]), "smma 头部应为 NaN"
    e = ema(x, 3)                                               # alpha=0.5: 1, 1.5, 2.25
    assert abs(e.iloc[2] - 2.25) < 1e-12, e.tolist()
    # ndarray 输入 -> RangeIndex 输出
    arr_out = ma(np.asarray(c.to_numpy(dtype=float)), 20)
    assert isinstance(arr_out, pd.Series) and isinstance(arr_out.index, pd.RangeIndex)
    print("[1] 均线口径校验通过：ma5 == close.rolling(5).mean()；smma(=[1,2,3,4,5],3)=%.6f "
          "与通达信 SMA(X,N,1) 一致" % sm.iloc[2])

    # ---------- 2. 全指标调用 ---------- #
    res = {
        "expma": expma(c), "bbi": bbi(c), "dkx": dkx(o, h, l, c), "macd": macd(c),
        "kdj": kdj(h, l, c), "rsi": rsi(c, 14), "rsi_multi": rsi_multi(c),
        "boll": boll(c), "ene": ene(c), "sar": sar(h, l), "bias": bias(c),
        "cci": cci(h, l, c), "dmi": dmi(h, l, c), "wr": wr(h, l, c), "dma": dma(c),
        "trix": trix(c), "brar": brar(o, h, l, c), "vr": vr(c, v), "obv": obv(c, v),
        "obv_ma": obv_ma(c, v), "asi": asi(o, h, l, c), "emv": emv(h, l, v),
        "wvad": wvad(o, h, l, c, v), "roc": roc(c), "lon": lon(c), "psy": psy(c),
        "mtm": mtm(c), "cr": cr(h, l, c), "vwap_cost": vwap_cost(h, l, c, v),
        "chip": chip_distribution(h, l, c, v), "vol_price_corr": vol_price_corr(c, v),
        "amount_ma": amount_ma(amt), "turnover_rate": turnover_rate(v, 1.25e9),
        "gap_stats": gap_stats(o, h, l, c), "trend_strength": trend_strength(c, h, l),
        "sr": support_resistance(h, l, c, v), "channel": channel_fit(h, l, c),
        "fib": fib_levels(h, l), "golden_pit": golden_pit(h, l, c, v),
        "renko": renko(cdi, hdi, ldi), "equal_price": equal_kline(df, "price"),
        "equal_vol": equal_kline(df, "volume"), "weekly": resample_kline(df, "W"),
        "monthly": resample_kline(df, "M"),
    }
    # 无 ±inf
    bad = []
    for name, obj in res.items():
        for sub, arr in _iter_numeric(obj):
            if arr is None or arr.size == 0:
                continue
            if np.isinf(arr).any():
                bad.append("%s.%s" % (name, sub))
    assert not bad, "存在 ±inf: %s" % bad
    print("[2] 全部 %d 组指标无 ±inf，无异常中断" % len(res))

    # ---------- 3. 取值域校验 ---------- #
    r14 = res["rsi"]
    rv = r14.to_numpy(dtype=float)
    rv = rv[np.isfinite(rv)]
    assert rv.size > 0 and rv.min() >= -1e-9 and rv.max() <= 100 + 1e-9, "RSI 越界"
    kk, dd = res["kdj"]["k"], res["kdj"]["d"]
    kv = kk.to_numpy(dtype=float); kv = kv[np.isfinite(kv)]
    dv = dd.to_numpy(dtype=float); dv = dv[np.isfinite(dv)]
    assert kv.min() >= -1e-9 and kv.max() <= 100 + 1e-9, "K 越界"
    assert dv.min() >= -1e-9 and dv.max() <= 100 + 1e-9, "D 越界"
    bo = res["boll"]
    ok = np.isfinite(bo["mid"].to_numpy(dtype=float))
    assert (bo["upper"].to_numpy()[ok] >= bo["mid"].to_numpy()[ok] - 1e-9).all(), "BOLL upper<mid"
    assert (bo["mid"].to_numpy()[ok] >= bo["lower"].to_numpy()[ok] - 1e-9).all(), "BOLL mid<lower"
    wrv = res["wr"]["wr6"].to_numpy(dtype=float)
    wrv = wrv[np.isfinite(wrv)]
    assert wrv.min() >= -100 - 1e-9 and wrv.max() <= 0 + 1e-9, "WR 应为 -100~0"
    bb = res["bbi"].to_numpy(dtype=float); bb = bb[np.isfinite(bb)]
    assert bb.size > 0
    st = res["sar"]["trend"].to_numpy()
    assert set(np.unique(st)).issubset({-1, 0, 1}), "sar trend 取值异常"
    fl = res["sar"]["flip"].to_numpy()
    assert set(np.unique(fl)).issubset({-1, 0, 1}), "sar flip 取值异常"
    assert res["sar"]["flip"].abs().sum() >= 1, "SAR 长期未翻转，疑似实现错误"
    print("[3] 取值域校验通过：RSI∈[0,100]=[%.2f,%.2f]，K/D∈[%.2f,%.2f]，BOLL upper≥mid≥lower，"
          "WR∈[-100,0]，SAR 翻转 %d 次"
          % (rv.min(), rv.max(), kv.min(), kv.max(), int(res["sar"]["flip"].abs().sum())))

    # ---------- 4. 筹码 / 支撑压力 / 趋势强度 ---------- #
    chip = res["chip"]
    assert abs(float(chip["hist"].sum()) - 1.0) < 1e-9, "chip hist 未归一化"
    assert len(chip["prices"]) == len(chip["hist"]), "chip prices/hist 长度不一致"
    assert 0.0 <= chip["profit_ratio"] <= 1.0 + 1e-9, "profit_ratio 越界"
    sr = res["sr"]
    assert len(sr["supports"]) >= 1 and len(sr["resistances"]) >= 1, "支撑压力至少各 1 个"
    assert sr["position"] in ("near_support", "near_resistance", "mid"), "position 取值异常"
    assert 0.0 <= sr["supports"][0]["strength"] <= 1.0
    assert sr["supports"][0]["price"] <= sr["resistances"][0]["price"], "压力位低于支撑位"
    ts = res["trend_strength"]
    assert ts["direction"] in (-1, 0, 1) and 0.0 <= ts["r2"] <= 1.0 + 1e-9
    # 纯线性序列应给出 r2≈1、direction=+1
    lin = pd.Series(np.linspace(10.0, 30.0, 40))
    ts2 = trend_strength(lin, n=20)
    assert ts2["r2"] > 0.999 and ts2["direction"] == 1, ("线性趋势识别失败", ts2)
    print("[4] 筹码归一化 sum(hist)=%.6f，concentration=%.3f，peak=%.2f；"
          "支撑 %d 个 / 压力 %d 个（最近支撑 %.2f / 压力 %.2f，%s）；"
          "线性趋势 r2=%.4f dir=%d"
          % (chip["hist"].sum(), chip["concentration"], chip["peak_price"],
             len(sr["supports"]), len(sr["resistances"]),
             sr["nearest_support"], sr["nearest_resistance"], sr["position"],
             ts2["r2"], ts2["direction"]))

    # ---------- 5. dict 键完整性 ---------- #
    keys_expect = {
        "expma": {"expma1", "expma2"},
        "dkx": {"mid", "dkx", "trend_up", "cross"},
        "macd": {"dif", "dea", "macd", "cross", "hist_trend"},
        "kdj": {"k", "d", "j", "cross", "j_value"},
        "boll": {"mid", "upper", "lower", "width", "pct_b", "bandwidth_pct"},
        "ene": {"upper", "lower", "mid", "position"},
        "sar": {"sar", "trend", "flip"},
        "dmi": {"pdi", "mdi", "adx", "adxr", "signal"},
        "dma": {"dma", "ama"},
        "trix": {"trix", "trma", "cross"},
        "brar": {"ar", "br"},
        "obv_ma": {"obv", "obv_ma"},
        "asi": {"asi", "asima", "si"},
        "emv": {"emv", "emv_ma"},
        "wvad": {"wvad", "wvad_ma"},
        "roc": {"roc", "roc_ma"},
        "lon": {"lon", "lon_ma"},
        "mtm": {"mtm", "mtm_ma"},
        "cr": {"cr", "ma1", "ma2", "ma3"},
        "vwap_cost": {"vwap", "cost_60", "profit_ratio"},
        "chip": {"hist", "prices", "concentration", "avg_cost", "peak_price",
                 "main_peak", "profit_ratio"},
        "gap_stats": {"gaps", "up_gap_count", "down_gap_count", "last_gap"},
        "trend_strength": {"slope", "r2", "adx_like", "direction"},
        "sr": {"supports", "resistances", "nearest_support", "nearest_resistance", "position"},
        "channel": {"upper_slope", "lower_slope", "upper_intercept", "lower_intercept",
                    "upper_now", "lower_now", "channel", "position", "width_pct", "break"},
        "fib": {"high", "low", "direction", "levels", "retracement"},
        "golden_pit": {"detected", "pit_low", "pit_depth_pct", "recover_bars", "score"},
    }
    for name, need in keys_expect.items():
        missing = need - set(res[name].keys())
        assert not missing, "%-14s 缺少键: %s" % (name, missing)
    assert set(res["bias"].keys()) == {"bias6", "bias12", "bias24"}
    assert {"wr6", "wr10"}.issubset(res["wr"].keys())
    assert {"rsi6", "rsi12", "rsi24"}.issubset(res["rsi_multi"].keys())
    assert {"amount_ma5", "amount_ma20"}.issubset(res["amount_ma"].keys())
    print("[5] 全部 dict 返回键完整（%d 个函数）" % len(keys_expect))

    # ---------- 6. K 线重采样 / 砖图 ---------- #
    assert set(res["renko"].columns) == {"date", "open", "high", "low", "close", "brick_dir"}
    assert set(res["renko"]["brick_dir"].unique()).issubset({-1, 1})
    assert (res["renko"]["high"] >= res["renko"]["low"]).all()
    for key in ("equal_price", "equal_vol"):
        cols = set(res[key].columns)
        assert cols == {"bar_index", "open", "high", "low", "close", "volume", "amount",
                        "start_date", "end_date", "n_orig"}, cols
        assert int(res[key]["n_orig"].sum()) == 60, "等价/等量K线未覆盖全部窗口 K 线"
        assert (res[key]["high"] >= res[key]["low"]).all(), "聚合后 high<low"
    wk = res["weekly"]
    assert "pct_chg" in wk.columns and len(wk) > 0
    assert abs(int(wk["volume"].sum()) - int(v.sum())) < 1e-6, "周线成交量求和不一致"
    mo = res["monthly"]
    assert len(mo) > 0 and len(mo) < len(wk) < len(df), "周/月线聚合层数不合理"
    print("[6] 砖图 %d 块 / 等价K线 %d 根 / 等量K线 %d 根 / 周线 %d 根 / 月线 %d 根"
          % (len(res["renko"]), len(res["equal_price"]), len(res["equal_vol"]),
             len(wk), len(mo)))

    # ---------- 7. 涨停 / 缺口 / 黄金坑 ---------- #
    code = "600519"
    prev = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    cls = pd.Series([11.0, 11.0, 9.0, 10.5, 10.0, 10.0])       # 涨停 / 一字 / 跌停 / 炸板 / 平 / 平
    hi = pd.Series([11.0, 11.0, 9.0, 11.0, 10.0, 10.0])
    lo = pd.Series([10.4, 11.0, 9.0, 10.0, 10.0, 10.0])
    op = pd.Series([10.5, 11.0, 9.0, 10.1, 10.0, 10.0])
    ls = limit_status_full(op, hi, lo, cls, prev, code, "")
    assert ls.iloc[0] == "limit_up", ls.tolist()
    assert ls.iloc[1] == "one_word", ls.tolist()
    assert ls.iloc[2] == "limit_down", ls.tolist()
    assert ls.iloc[3] == "broken", ls.tolist()
    assert ls.iloc[4] == "" and ls.iloc[5] == "", ls.tolist()
    # 只给 close/prev_close 时退化为三态（文档签名仍可用）
    ls2 = limit_status(cls, prev, code)
    assert set(ls2.unique()).issubset({"limit_up", "limit_down", "one_word", "broken", ""})
    assert ls2.iloc[4] == "" and ls2.iloc[0] == "limit_up", ls2.tolist()
    # 直接给 OHLC 时五态齐全（一字板 / 炸板 可判定）
    ls3 = limit_status(cls, prev, code, "", high=hi, low=lo, open_=op)
    assert ls3.tolist() == ls.tolist(), ls3.tolist()
    # 创业板 19.5% 口径（10.00 -> 涨停价 11.95）
    ls4 = limit_status(pd.Series([11.95, 11.79]), pd.Series([10.0, 10.0]), "300750", "")
    assert ls4.iloc[0] == "limit_up", ls4.tolist()
    assert limit_status(pd.Series([10.98]), pd.Series([10.0]), "600519", "").iloc[0] == "limit_up"
    assert limit_status(pd.Series([10.5]), pd.Series([10.0]), "600519", "ST某某").iloc[0] == "limit_up"
    gap_df = pd.DataFrame({
        "open": [10.0, 10.6, 10.5, 11.2, 11.0],
        "high": [10.2, 10.8, 10.6, 11.4, 11.1],
        "low": [9.8, 10.5, 10.0, 11.15, 10.9],
        "close": [10.1, 10.7, 10.2, 11.3, 11.0],
    })
    gs = gap_stats(gap_df["open"], gap_df["high"], gap_df["low"], gap_df["close"])
    assert gs["up_gap_count"] >= 1 and gs["down_gap_count"] >= 1, gs
    assert gs["last_gap"] is not None and gs["last_gap"]["type"] in ("up", "down")
    # 构造一个标准黄金坑：震荡 -> 冲高 22.72（坑口）-> 5 根跌到 18.32（破前低 19.6）->
    # 5 根放量收复到 20.6（> 坑口 90% = 20.45）
    base = [19.8 + 1.2 * ((i % 8) / 7.0) for i in range(30)]
    pit_px = base + [22.5, 21.8, 21.0, 20.2, 19.4, 18.5,
                     18.9, 19.4, 19.9, 20.3, 20.6, 20.9, 21.0, 21.2]
    pit_px += [21.0] * (60 - len(pit_px))
    pit_vol = [1.0e6] * 30 + [1.0e6] * 6 + [2.6e6] * 5 + [1.0e6] * (60 - 41)
    pit_df = pd.DataFrame({"high": [p * 1.01 for p in pit_px],
                           "low": [p * 0.99 for p in pit_px],
                           "close": pit_px, "volume": pit_vol})
    pit = golden_pit(pit_df["high"], pit_df["low"], pit_df["close"], pit_df["volume"])
    assert pit["detected"], ("黄金坑未被识别", pit)
    assert pit["pit_depth_pct"] >= 8.0 and pit["recover_bars"] <= 15, pit
    assert pit["score"] > 0.3, pit
    pit_novol = golden_pit(pit_df["high"], pit_df["low"], pit_df["close"])   # 向后兼容调用
    assert set(pit_novol) >= {"detected", "pit_low", "pit_depth_pct", "recover_bars", "score"}
    print("[7] 涨停/跌停/一字板/炸板 判定全部通过；缺口 up=%d down=%d（最近 %s %.2f%%，%s）；"
          "黄金坑 detected=%s 深度=%.1f%% 收复=%d根 score=%.3f"
          % (gs["up_gap_count"], gs["down_gap_count"], gs["last_gap"]["type"],
             gs["last_gap"]["size_pct"], "已回补" if gs["last_gap"]["filled"] else "未回补",
             pit["detected"], pit["pit_depth_pct"], pit["recover_bars"], pit["score"]))

    # ---------- 8. indicator_matrix ---------- #
    mat = indicator_matrix(df)
    need_cols = ["ma5", "ma20", "macd_dif", "macd_dea", "macd_hist", "k", "d", "j", "rsi6",
                 "rsi14", "boll_mid", "boll_up", "boll_low", "boll_pctb", "ene_up", "ene_low",
                 "sar", "sar_trend", "dkx", "bbi", "expma1", "expma2", "bias6", "bias12",
                 "bias24", "cci", "pdi", "mdi", "adx", "wr6", "wr10", "dma", "ama", "trix",
                 "trma", "ar", "br", "vr", "obv", "obv_ma", "asi", "asima", "emv", "emv_ma",
                 "wvad", "roc", "lon", "psy", "mtm", "cr", "vwap", "vol_ma5", "vol_ma20",
                 "atr14", "amp_pct", "turnover_ma5"]
    missing = [x for x in need_cols if x not in mat.columns]
    assert not missing, "indicator_matrix 缺少列: %s" % missing
    assert mat.shape[0] == len(df), "indicator_matrix 行数不一致"
    assert mat.shape[1] >= 45, "列数不足 45"
    iv = mat.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    assert not np.isinf(iv).any(), "indicator_matrix 含 ±inf"
    print("[8] indicator_matrix(df).shape = %s，列数 = %d（要求 ≥45）" % (mat.shape, mat.shape[1]))
    print("    前 12 列:", list(mat.columns[:12]))
    print("    末 6 列 :", list(mat.columns[-6:]))

    # ---------- 9. 契约一致性 ---------- #
    assert indicator_matrix(df).shape == mat.shape, "indicator_matrix 非确定性"
    assert df.equals(df_backup), "入参 DataFrame 被修改"
    assert mat.index.equals(df.index), "indicator_matrix 索引与输入不一致"
    for nm in ("ma5", "k", "rsi14", "atr14"):
        arr = mat[nm].to_numpy(dtype=float)
        assert arr.size == len(df) and np.isnan(arr[:4]).all(), "%s 头部未按约定留 NaN" % nm
    assert np.isnan(mat["amp_pct"].to_numpy(dtype=float)[0]), "amp_pct 首根应为 NaN"
    print("[9] 入参 DataFrame 未被修改；index 对齐；指标头部保留 NaN（不 fillna、不裁剪）")

    print("indicators self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()

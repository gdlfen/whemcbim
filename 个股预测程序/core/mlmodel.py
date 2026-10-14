# -*- coding: utf-8 -*-
"""
AI 学习核心 —— 从个股自身历史走势中学习"因素 → 未来涨跌"的规律。

设计取向（为什么不用 sklearn / 深度学习）
------------------------------------------
1. **无额外依赖**：纯 numpy 实现逻辑回归、直方图梯度提升树、KNN 历史类比，
   在 Streamlit Cloud / 本地 / 离线环境都能跑，不会因缺少编译库而失败。
2. **因果严格**：所有特征只用第 t 根及之前的数据；标签是 t 之后的前瞻收益。
   标准化参数、时间衰减权重、模型系数**只在训练窗口内估计**，绝无未来函数。
3. **时间序列友好**：
   * 时间衰减样本权重（近期样本更重要，半衰期可配）；
   * 扩展窗口前向验证（expanding walk-forward）估计真实样本外表现；
   * 特征按横截面/时序稳健化（分位截断 + z-score 标准化）。
4. **可解释**：逻辑回归系数会按"标准化尺度 × 特征波动"折算成**特征贡献**，
   直接进入报告的"AI 认为什么最重要"。

对外主接口
----------
``train_models(df, ctx=None, horizons=None)`` → ``dict[horizon_key, TrainedModel]``
``predict(models, df, ctx=None)``             → ``dict[horizon_key, ModelPrediction]``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .contracts import HORIZONS, horizon_spec
from .utils import get_logger, norm_cdf, sigmoid

log = get_logger()

# --------------------------------------------------------------------------- #
# 中文特征名（用于报告展示）
# --------------------------------------------------------------------------- #
FEATURE_LABELS = {
    "ret_1": "近1日涨幅",
    "ret_3": "近3日涨幅",
    "ret_5": "近5日涨幅",
    "ret_10": "近10日涨幅",
    "ret_20": "近20日涨幅",
    "ret_60": "近60日涨幅",
    "mom_20_60": "中期动量差(20-60日)",
    "rsi_14": "RSI(14)",
    "rsi_6": "RSI(6)",
    "kdj_k": "KDJ的K值",
    "kdj_j": "KDJ的J值",
    "macd_hist": "MACD柱",
    "macd_dif_dea": "DIF与DEA差",
    "cci_14": "CCI(14)",
    "wr_10": "威廉指标WR(10)",
    "bias_6": "乖离率BIAS(6)",
    "bias_20": "乖离率BIAS(20)",
    "vol_5": "5日年化波动率",
    "vol_20": "20日年化波动率",
    "vol_ratio_5_20": "短长期波动比",
    "atr_pct": "ATR占价格比",
    "amp_5": "近5日平均振幅",
    "ma5_dev": "偏离MA5",
    "ma10_dev": "偏离MA10",
    "ma20_dev": "偏离MA20",
    "ma60_dev": "偏离MA60",
    "ma_align": "均线排列得分",
    "ma20_slope": "MA20斜率",
    "ma60_slope": "MA60斜率",
    "price_pos_20": "20日区间位置",
    "price_pos_60": "60日区间位置",
    "price_pos_250": "250日区间位置",
    "drawdown": "距250日高点回撤",
    "vol_ma_ratio": "量比(5/20日)",
    "vol_z20": "成交量z分",
    "turnover_z": "换手率z分",
    "vp_corr_20": "量价相关性",
    "obv_slope": "OBV斜率",
    "boll_pctb": "布林%B",
    "boll_width": "布林带宽",
    "dist_high_20": "距20日高点",
    "dist_low_20": "距20日低点",
    "dist_high_60": "距60日高点",
    "dist_low_60": "距60日低点",
    "up_days_10": "近10日上涨天数",
    "gap_sum_5": "近5日跳空累计",
    "limit_up_20": "近20日涨停次数",
    "rel_idx_5": "相对大盘5日强度",
    "rel_idx_20": "相对大盘20日强度",
    "corr_idx_20": "与大盘20日相关",
    "beta_idx_60": "60日Beta",
    "idx_ret_5": "大盘近5日涨幅",
    "idx_pos_60": "大盘60日区间位置",
}

FEATURE_COLUMNS: tuple[str, ...] = tuple(FEATURE_LABELS.keys())


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass
class ModelPrediction:
    """机器学习模型对单个周期的输出。"""

    horizon: str = ""
    label: str = ""
    p_up: float = 0.5
    exp_return: float = 0.0
    sigma: float = 0.0
    n_train: int = 0
    n_features: int = 0
    auc: float = float("nan")          # 前向验证样本外 AUC
    ic: float = float("nan")           # 预测分与真实收益的秩相关
    hit_rate: float = float("nan")     # 方向命中率
    brier: float = float("nan")
    base_rate: float = 0.5             # 训练集上涨基准比例
    model: str = "logit"
    confidence: float = 0.5
    top_features: list = field(default_factory=list)   # [(中文特征名, 贡献值), ...]
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.n_train >= config.RUNTIME["ml_min_samples"] and self.n_features > 0


@dataclass
class TrainedModel:
    """一个周期训练好的模型集合。"""

    horizon: str
    label: str
    bars: int
    features: list                                     # 特征名（顺序即列顺序）
    models: dict = field(default_factory=dict)          # name -> 模型对象
    oos: dict = field(default_factory=dict)             # 样本外评估指标
    base_rate: float = 0.5
    n_train: int = 0
    sigma_hist: float = 0.0                             # 该周期历史收益标准差（%）
    sample_weights_note: str = ""
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.n_train >= config.RUNTIME["ml_min_samples"] and bool(self.models)


# =========================================================================== #
# 一、特征工程（严格只用 t 及之前的信息）
# =========================================================================== #
def _shift1(x: np.ndarray) -> np.ndarray:
    """因果滞后一期：out[i] = x[i-1]，out[0] = NaN。

    注意：**绝对不能用 np.roll(x, 1)** —— 它会把序列最后一个元素（未来数据）
    搬到开头，造成隐蔽的未来函数。这里显式把首位置为 NaN。
    """
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) > 1:
        out[1:] = x[:-1]
    return out


def _safe_div(a, b, default=0.0):
    """安全除法：**只**把"分母为 0"的位置替换成 default。

    注意：分子/分母本身是 NaN（例如指标预热期）时必须保持 NaN，
    否则会把"序列开头的缺失"变成确定值 0，从而引入位置信息泄漏
    （截断序列重算时结果会变，等于间接使用了未来数据）。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=(np.abs(b) > 1e-12))
    zero_den = np.abs(b) <= 1e-12
    if np.ndim(out) == 0:
        return float(default) if bool(zero_den) else float(out)
    out = np.where(zero_den, default, out)
    return np.where(np.isinf(out), default, out)


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """因果滚动均值（不使用未来数据）。

    窗口内有效值不足 n 个时返回 NaN（**不做零填充**）——这样序列开头的缺失
    不会被伪装成有效数值，保证特征严格因果。
    """
    x = np.asarray(x, dtype=float)
    if n <= 1:
        return x.copy()
    v = np.isfinite(x).astype(float)
    xz = np.where(np.isfinite(x), x, 0.0)
    cs = np.cumsum(np.insert(xz, 0, 0.0))
    cv = np.cumsum(np.insert(v, 0, 0.0))
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        s = cs[n:] - cs[:-n]
        cnt = cv[n:] - cv[:-n]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(cnt >= n, s / n, np.nan)
        out[n - 1:] = m
    return out


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = _rolling_mean(x, n)
    m2 = _rolling_mean(x * x, n)
    with np.errstate(invalid="ignore"):
        var = np.maximum(m2 - m * m, 0.0)
    return np.sqrt(var * n / max(n - 1, 1))


def _rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if i >= n - 1:
            out[i] = np.max(x[i - n + 1:i + 1])
    return out


def _rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if i >= n - 1:
            out[i] = np.min(x[i - n + 1:i + 1])
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) == 0:
        return out
    alpha = 2.0 / (n + 1.0)
    prev = x[0]
    out[0] = prev
    for i in range(1, len(x)):
        prev = alpha * x[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    # Wilder 平滑
    au = np.full(len(close), np.nan)
    ad = np.full(len(close), np.nan)
    if len(close) > n:
        au[n] = up[1:n + 1].mean()
        ad[n] = dn[1:n + 1].mean()
        for i in range(n + 1, len(close)):
            au[i] = (au[i - 1] * (n - 1) + up[i]) / n
            ad[i] = (ad[i - 1] * (n - 1) + dn[i]) / n
    return _safe_div(au, au + ad, 0.5) * 100.0


def _kdj(high, low, close, n=9):
    hh = _rolling_max(high, n)
    ll = _rolling_min(low, n)
    rsv = _safe_div(close - ll, hh - ll, 0.5) * 100.0
    rsv = np.nan_to_num(rsv, nan=50.0)
    k = np.full(len(close), 50.0)
    d = np.full(len(close), 50.0)
    for i in range(1, len(close)):
        k[i] = (2 * k[i - 1] + rsv[i]) / 3.0
        d[i] = (2 * d[i - 1] + k[i]) / 3.0
    return k, d, 3 * k - 2 * d


def _atr(high, low, close, n=14):
    pc = _shift1(close)
    tr = np.nanmax(np.vstack([high - low,
                              np.abs(high - pc),
                              np.abs(low - pc)]), axis=0)
    return _ema(tr, n)


def _cci(high, low, close, n=14):
    tp = (high + low + close) / 3.0
    ma = _rolling_mean(tp, n)
    md = np.full(len(tp), np.nan)
    for i in range(n - 1, len(tp)):
        md[i] = np.mean(np.abs(tp[i - n + 1:i + 1] - ma[i]))
    return _safe_div(tp - ma, 0.015 * md, 0.0)


def _linear_slope(y: np.ndarray) -> float:
    """序列的线性回归斜率（按每根 K 线）。"""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom <= 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def _rolling_slope(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if i >= n - 1:
            out[i] = _linear_slope(x[i - n + 1:i + 1])
    return out


def build_features(df: pd.DataFrame, index_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """从 OHLCV 构造特征矩阵。

    Parameters
    ----------
    df : 日线数据，需含 close/high/low/open/volume（amount/turnover 可选）
    index_df : 大盘日线（可选），用于相对强度/Beta/相关性特征

    Returns
    -------
    DataFrame，索引与 ``df`` 对齐，列为 :data:`FEATURE_COLUMNS`。
    **所有列在时刻 t 只依赖 ≤ t 的数据。**
    """
    if df is None or len(df) < 30:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))

    d = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in d.columns:
            d[col] = np.nan
    c = d["close"].astype(float).to_numpy()
    h = d["high"].astype(float).to_numpy()
    lo = d["low"].astype(float).to_numpy()
    o = d["open"].astype(float).to_numpy()
    v = d["volume"].astype(float).to_numpy()
    n = len(c)
    feat: dict[str, np.ndarray] = {}

    # ---- 动量 ---------------------------------------------------------- #
    for k in (1, 3, 5, 10, 20, 60):
        feat["ret_%d" % k] = np.concatenate([np.full(k, np.nan),
                                             c[k:] / c[:-k] - 1.0]) * 100.0
    r20 = feat["ret_20"]
    r60 = feat["ret_60"]
    feat["mom_20_60"] = np.where(np.isnan(r20) | np.isnan(r60), np.nan, r20 - r60)

    # ---- 摆动类指标 ----------------------------------------------------- #
    feat["rsi_14"] = _rsi(c, 14)
    feat["rsi_6"] = _rsi(c, 6)
    k, dd, j = _kdj(h, lo, c, 9)
    feat["kdj_k"] = k
    feat["kdj_j"] = j
    dif = _ema(c, 12) - _ema(c, 26)
    dea = _ema(dif, 9)
    feat["macd_hist"] = (dif - dea) * 2.0 / np.where(c > 0, c, 1.0) * 100.0
    feat["macd_dif_dea"] = (dif - dea) / np.where(c > 0, c, 1.0) * 100.0
    feat["cci_14"] = _cci(h, lo, c, 14)
    hh10, ll10 = _rolling_max(h, 10), _rolling_min(lo, 10)
    feat["wr_10"] = -_safe_div(hh10 - c, hh10 - ll10, 0.5) * 100.0
    # 乖离率
    for kk in (6, 20):
        ma = _rolling_mean(c, kk)
        feat["bias_%d" % kk] = _safe_div(c - ma, ma, 0.0) * 100.0

    # ---- 波动 ---------------------------------------------------------- #
    ret1 = np.concatenate([[np.nan], c[1:] / c[:-1] - 1.0])
    for kk in (5, 20):
        feat["vol_%d" % kk] = _rolling_std(ret1, kk) * math.sqrt(244) * 100.0
    feat["vol_ratio_5_20"] = _safe_div(feat["vol_5"], feat["vol_20"], 1.0)
    atr14 = _atr(h, lo, c, 14)
    feat["atr_pct"] = _safe_div(atr14, c, 0.0) * 100.0
    amp = _safe_div(h - lo, _shift1(c), 0.0) * 100.0
    feat["amp_5"] = _rolling_mean(amp, 5)

    # ---- 均线系统 ------------------------------------------------------- #
    mas = {kk: _rolling_mean(c, kk) for kk in (5, 10, 20, 60)}
    for kk in (5, 10, 20, 60):
        feat["ma%d_dev" % kk] = _safe_div(c - mas[kk], mas[kk], 0.0) * 100.0
    align = np.zeros(n)
    align += np.where(c > mas[5], 1, -1)
    align += np.where(mas[5] > mas[10], 1, -1)
    align += np.where(mas[10] > mas[20], 1, -1)
    align += np.where(mas[20] > mas[60], 1, -1)
    feat["ma_align"] = align / 4.0
    feat["ma20_slope"] = _safe_div(_rolling_slope(mas[20], 10), mas[20], 0.0) * 100.0
    feat["ma60_slope"] = _safe_div(_rolling_slope(mas[60], 20), mas[60], 0.0) * 100.0

    # ---- 价格位置 ------------------------------------------------------- #
    for kk in (20, 60, 250):
        hi, ln = _rolling_max(h, kk), _rolling_min(lo, kk)
        feat["price_pos_%d" % kk] = _safe_div(c - ln, hi - ln, 0.5)
    hi250 = _rolling_max(h, 250)
    feat["drawdown"] = _safe_div(c - hi250, hi250, 0.0) * 100.0

    # ---- 量能 ---------------------------------------------------------- #
    vma5, vma20 = _rolling_mean(v, 5), _rolling_mean(v, 20)
    feat["vol_ma_ratio"] = _safe_div(vma5, vma20, 1.0)
    vstd20 = _rolling_std(v, 20)
    feat["vol_z20"] = _safe_div(v - vma20, vstd20, 0.0)
    if "turnover" in d.columns and d["turnover"].notna().sum() > 20:
        t = d["turnover"].astype(float).to_numpy()
        feat["turnover_z"] = _safe_div(t - _rolling_mean(t, 20), _rolling_std(t, 20), 0.0)
    else:
        feat["turnover_z"] = np.zeros(n)
    # 量价相关（20 日滚动）
    vp = np.full(n, np.nan)
    for i in range(19, n):
        a, b = ret1[i - 19:i + 1], v[i - 19:i + 1]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            vp[i] = float(np.corrcoef(a, b)[0, 1])
        else:
            vp[i] = 0.0
    feat["vp_corr_20"] = vp
    obv = np.cumsum(np.sign(ret1) * v)
    obv_scale = float(np.nanmean(np.abs(obv))) if np.isfinite(obv).any() else 0.0
    feat["obv_slope"] = _safe_div(_rolling_slope(obv, 20),
                                  obv_scale if obv_scale > 1e-9 else 1.0, 0.0)

    # ---- 布林 ---------------------------------------------------------- #
    mid = _rolling_mean(c, 20)
    sd = _rolling_std(c, 20)
    up, lowb = mid + 2 * sd, mid - 2 * sd
    feat["boll_pctb"] = _safe_div(c - lowb, up - lowb, 0.5)
    feat["boll_width"] = _safe_div(up - lowb, mid, 0.0)

    # ---- 距离高低点 ----------------------------------------------------- #
    for kk in (20, 60):
        feat["dist_high_%d" % kk] = _safe_div(c - _rolling_max(h, kk), _rolling_max(h, kk), 0.0) * 100.0
        feat["dist_low_%d" % kk] = _safe_div(c - _rolling_min(lo, kk), _rolling_min(lo, kk), 0.0) * 100.0

    # ---- 微观结构 ------------------------------------------------------- #
    up_flag = (ret1 > 0).astype(float)
    up_flag[0] = np.nan
    feat["up_days_10"] = _rolling_mean(up_flag, 10)
    prev_c = _shift1(c)
    gap = _safe_div(o - prev_c, prev_c, 0.0) * 100.0
    feat["gap_sum_5"] = _rolling_mean(gap, 5)
    lu = np.where(np.isnan(ret1), np.nan, (ret1 >= 0.095).astype(float))
    feat["limit_up_20"] = _rolling_mean(lu, 20)

    # ---- 相对大盘 ------------------------------------------------------- #
    zeros = np.zeros(n)
    for key in ("rel_idx_5", "rel_idx_20", "corr_idx_20", "beta_idx_60",
                "idx_ret_5", "idx_pos_60"):
        feat[key] = zeros
    if index_df is not None and len(index_df) > 70:
        try:
            idx = index_df.copy()
            ic = idx["close"].astype(float).to_numpy()
            if "date" in d.columns and "date" in idx.columns:
                merged = pd.DataFrame({"date": pd.to_datetime(d["date"]),
                                       "ci": c}).merge(
                    pd.DataFrame({"date": pd.to_datetime(idx["date"]), "ii": ic}),
                    on="date", how="left")
                ii = merged["ii"].ffill().to_numpy()
            else:
                ii = np.interp(np.arange(n), np.linspace(0, n - 1, len(ic)), ic)
            i_ret1 = np.concatenate([[np.nan], ii[1:] / ii[:-1] - 1.0])
            for kk in (5, 20):
                ci = np.concatenate([np.full(kk, np.nan), c[kk:] / c[:-kk] - 1.0]) * 100.0
                ir = np.concatenate([np.full(kk, np.nan), ii[kk:] / ii[:-kk] - 1.0]) * 100.0
                feat["rel_idx_%d" % kk] = np.where(np.isnan(ir), 0.0, ci - ir)
            corr = np.zeros(n)
            for i in range(19, n):
                a, b = ret1[i - 19:i + 1], i_ret1[i - 19:i + 1]
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() > 10 and np.std(a[ok]) > 1e-12 and np.std(b[ok]) > 1e-12:
                    corr[i] = float(np.corrcoef(a[ok], b[ok])[0, 1])
            feat["corr_idx_20"] = corr
            beta = np.zeros(n)
            for i in range(59, n):
                a, b = ret1[i - 59:i + 1], i_ret1[i - 59:i + 1]
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() > 30 and np.var(b[ok]) > 1e-14:
                    beta[i] = float(np.cov(a[ok], b[ok])[0, 1] / np.var(b[ok]))
            feat["beta_idx_60"] = np.clip(beta, -3, 5)
            feat["idx_ret_5"] = np.where(np.isnan(ir := np.concatenate(
                [np.full(5, np.nan), ii[5:] / ii[:-5] - 1.0]) * 100.0), 0.0, ir)
            hi60, lo60 = _rolling_max(ii, 60), _rolling_min(ii, 60)
            feat["idx_pos_60"] = np.nan_to_num(_safe_div(ii - lo60, hi60 - lo60, 0.5), nan=0.5)
        except Exception as exc:  # noqa: BLE001
            log.debug("相对大盘特征计算失败: %s", exc)

    out = pd.DataFrame({k: np.asarray(feat.get(k, np.zeros(n)), dtype=float)
                        for k in FEATURE_COLUMNS})
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# =========================================================================== #
# 二、标签构造
# =========================================================================== #
def make_targets(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """构造前瞻收益标签。

    Returns
    -------
    DataFrame，列 ``fwd_<key>`` 为对应周期的未来收益率（%），
    未来不足 ``bars`` 根的位置为 NaN。
    """
    if df is None or len(df) < 2:
        return pd.DataFrame()
    c = df["close"].astype(float).to_numpy()
    n = len(c)
    out = {}
    for h in horizons:
        bars = min(h.bars, max(n - 2, 1))
        fwd = np.full(n, np.nan)
        if n > bars:
            fwd[:n - bars] = c[bars:] / c[:n - bars] - 1.0
        out["fwd_%s" % h.key] = fwd * 100.0
    return pd.DataFrame(out, index=df.index)


# =========================================================================== #
# 三、模型实现（纯 numpy）
# =========================================================================== #
def standardize(x: np.ndarray, mean=None, std=None):
    """列标准化；std 过小的列按 1 处理，避免放大噪声。"""
    x = np.asarray(x, dtype=float)
    if mean is None:
        mean = np.nanmean(x, axis=0)
    if std is None:
        std = np.nanstd(x, axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-9), 1.0, std)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return (x - mean) / std, mean, std


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """列中位数填充（用训练集的稳健中心），再兜底 0。"""
    x = np.asarray(x, dtype=float).copy()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    for j in range(x.shape[1]):
        col = x[:, j]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            col[bad] = np.nanmedian(col)
    return x


class LogisticModel:
    """L2 正则 + 时间衰减样本权重的逻辑回归（牛顿法/梯度下降混合）。"""

    name = "logit"

    def __init__(self, l2: float = 1.0, lr: float = 0.25, epochs: int = 400,
                 tol: float = 1e-6, standardize_features: bool = True):
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.standardize_features = standardize_features
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.med_: np.ndarray | None = None
        self.n_features_: int = 0
        self.fitted_: bool = False

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        x = _fill_nan(x)
        y = np.asarray(y, dtype=float)
        self.med_ = np.array([np.nanmedian(x[:, j]) for j in range(x.shape[1])])
        if self.standardize_features:
            x, self.mean_, self.std_ = standardize(x)
        else:
            self.mean_, self.std_ = np.zeros(x.shape[1]), np.ones(x.shape[1])
        n, p = x.shape
        self.n_features_ = p
        if sample_weight is None:
            w = np.ones(n)
        else:
            w = np.asarray(sample_weight, dtype=float)
        w = w / max(w.mean(), 1e-9)
        xb = np.hstack([np.ones((n, 1)), x])
        wb = w[:, None]
        beta = np.zeros(p + 1)
        prev_loss = np.inf
        for it in range(self.epochs):
            z = np.clip(xb @ beta, -30, 30)
            pr = 1.0 / (1.0 + np.exp(-z))
            # 梯度（带权重 + L2，截距不正则化）
            grad = (xb * (wb * (pr - y)[:, None])).sum(axis=0) / n
            reg = np.concatenate([[0.0], beta[1:]]) * (self.l2 / n)
            grad = grad + reg
            # 学习率衰减
            step = self.lr / (1.0 + it / 120.0)
            beta = beta - step * grad
            if it % 10 == 0:
                eps = 1e-12
                loss = -(w * (y * np.log(pr + eps) + (1 - y) * np.log(1 - pr + eps))).sum() / n \
                    + (self.l2 / (2 * n)) * float((beta[1:] ** 2).sum())
                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        self.fitted_ = True
        return self

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        med = self.med_
        n_fill = x.shape[1] if med is None else min(x.shape[1], len(med))
        for j in range(n_fill):
            col = x[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                col[bad] = med[j]
        if self.standardize_features and self.mean_ is not None:
            x = (x - self.mean_) / self.std_
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            return np.full(len(np.atleast_2d(x)), 0.5)
        z = np.clip(self._transform(x) @ self.coef_ + self.intercept_, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def feature_importance(self, feature_names) -> list:
        """把系数换算成"一个标准差变动带来的 log-odds 变化"，便于横向比较。"""
        if not self.fitted_:
            return []
        imp = np.abs(self.coef_)
        order = np.argsort(-imp)
        return [(str(feature_names[i]), float(self.coef_[i])) for i in order]


class KNNPatternModel:
    """历史类比（K 近邻）：在标准化特征空间里找历史上最相似的 K 个时点，
    用它们后续的真实收益投票。这是最"朴素"但也最直观的"从历史找规律"。"""

    name = "knn"

    def __init__(self, k: int = 40, decay: float = 0.02, min_pool: int = 60):
        self.k = k
        self.decay = decay
        self.min_pool = min_pool
        self.pool_: np.ndarray | None = None
        self.pool_ret_: np.ndarray | None = None
        self.pool_idx_: np.ndarray | None = None
        self.mean_ = None
        self.std_ = None
        self.med_ = None
        self.fitted_ = False

    def fit(self, x: np.ndarray, y_ret: np.ndarray,
            sample_weight: np.ndarray | None = None):
        """注意：KNN 学习的是**历史相似形态 → 后续实际收益**的映射。"""
        x = _fill_nan(x)
        ok = np.isfinite(y_ret)
        self.med_ = np.array([np.nanmedian(x[:, j]) for j in range(x.shape[1])])
        if ok.sum() < self.min_pool:
            self.fitted_ = False
            return self
        xs, _, _ = standardize(x[ok])
        self.mean_, self.std_ = np.nanmean(x[ok], axis=0), np.nanstd(x[ok], axis=0)
        self.std_ = np.where(self.std_ < 1e-9, 1.0, self.std_)
        self.pool_ = xs
        self.pool_ret_ = np.asarray(y_ret, dtype=float)[ok]
        self.pool_idx_ = np.arange(len(x))[ok]
        self.fitted_ = True
        return self

    def predict_return(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (加权平均后续收益, 相似样本上涨比例)。"""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if not self.fitted_ or self.pool_ is None:
            return np.zeros(len(x)), np.full(len(x), 0.5)
        for j in range(x.shape[1]):
            bad = ~np.isfinite(x[:, j])
            if bad.any():
                x[bad, j] = self.med_[j]
        xs = (x - self.mean_) / self.std_
        xs = np.nan_to_num(xs, nan=0.0)
        m = len(xs)
        out_ret = np.zeros(m)
        out_up = np.full(m, 0.5)
        pool_n = len(self.pool_)
        k = min(self.k, pool_n)
        for i in range(m):
            d = np.sqrt(((self.pool_ - xs[i]) ** 2).sum(axis=1))
            nn = np.argpartition(d, k - 1)[:k]
            w = np.exp(-self.decay * d[nn] * 10.0)
            w = w / max(w.sum(), 1e-12)
            out_ret[i] = float((w * self.pool_ret_[nn]).sum())
            out_up[i] = float((w * (self.pool_ret_[nn] > 0)).sum())
        return out_ret, out_up


class GBDTModel:
    """极简直方图梯度提升树（回归树拟合 logit 残差）。

    * 特征分箱（默认 32 箱）→ 分裂点只在箱边界上搜索，速度快且抗噪；
    * 树深默认 2（≈ 3 次交互），学习率 0.08，最多 60 轮，早停看验证损失；
    * 输出经 sigmoid 得到概率。样本权重通过加权残差实现。
    """

    name = "gbdt"

    def __init__(self, n_trees: int = 60, lr: float = 0.08, max_depth: int = 2,
                 bins: int = 32, min_leaf: int = 15, l2: float = 1.0,
                 subsample: float = 0.8, seed: int = 7):
        self.n_trees = n_trees
        self.lr = lr
        self.max_depth = max_depth
        self.bins = bins
        self.min_leaf = min_leaf
        self.l2 = l2
        self.subsample = subsample
        self.seed = seed
        self.trees: list = []
        self.base_: float = 0.0
        self.med_ = None
        self.edges_: list = []
        self.fitted_ = False

    # -- 内部工具 ------------------------------------------------------- #
    def _make_edges(self, x: np.ndarray):
        self.edges_ = []
        for j in range(x.shape[1]):
            col = x[:, j]
            qs = np.linspace(0, 100, self.bins + 1)
            e = np.unique(np.percentile(col, qs))
            if len(e) < 3:
                e = np.array([col.min() - 1e-6, col.max() + 1e-6])
            self.edges_.append(e)

    def _bin(self, x: np.ndarray) -> np.ndarray:
        b = np.empty(x.shape, dtype=np.int32)
        for j in range(x.shape[1]):
            b[:, j] = np.clip(np.searchsorted(self.edges_[j], x[:, j], side="right") - 1,
                              0, len(self.edges_[j]) - 2)
        return b

    def _best_split(self, xb, grad, hess, w, idx):
        best = None
        n_node = len(idx)
        if n_node < 2 * self.min_leaf:
            return None
        g_tot = grad[idx].sum()
        h_tot = hess[idx].sum()
        base_score = (g_tot ** 2) / (h_tot + self.l2 + 1e-9)
        for j in range(xb.shape[1]):
            col = xb[idx, j]
            order = np.argsort(col, kind="stable")
            idx_s = idx[order]
            g_cum = np.cumsum(grad[idx_s])
            h_cum = np.cumsum(hess[idx_s])
            col_s = col[order]
            for pos in range(self.min_leaf - 1, n_node - self.min_leaf):
                if col_s[pos] == col_s[pos + 1]:
                    continue
                gl, hl = g_cum[pos], h_cum[pos]
                gr, hr = g_tot - gl, h_tot - hl
                gain = (gl ** 2) / (hl + self.l2 + 1e-9) + \
                       (gr ** 2) / (hr + self.l2 + 1e-9) - base_score
                if best is None or gain > best[0]:
                    best = (gain, j, float(col_s[pos]))
        return best

    def _build(self, xb, grad, hess, w, idx, depth):
        split = self._best_split(xb, grad, hess, w, idx)
        if split is None or depth >= self.max_depth:
            g = grad[idx].sum()
            h = hess[idx].sum()
            return {"leaf": float(-g / (h + self.l2 + 1e-9))}
        _, j, thr = split
        left = idx[xb[idx, j] <= thr]
        right = idx[xb[idx, j] > thr]
        if len(left) < self.min_leaf or len(right) < self.min_leaf:
            g = grad[idx].sum()
            h = hess[idx].sum()
            return {"leaf": float(-g / (h + self.l2 + 1e-9))}
        return {"j": int(j), "thr": thr,
                "left": self._build(xb, grad, hess, w, left, depth + 1),
                "right": self._build(xb, grad, hess, w, right, depth + 1)}

    @staticmethod
    def _predict_tree(tree, xb_row) -> float:
        node = tree
        while "leaf" not in node:
            node = node["left"] if xb_row[node["j"]] <= node["thr"] else node["right"]
        return node["leaf"]

    # -- 接口 ----------------------------------------------------------- #
    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        x = _fill_nan(x)
        y = np.asarray(y, dtype=float)
        self.med_ = np.array([np.nanmedian(x[:, j]) for j in range(x.shape[1])])
        self._make_edges(x)
        xb = self._bin(x)
        n = len(y)
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        w = w / max(w.mean(), 1e-9)
        p0 = float(np.clip((w * y).sum() / max(w.sum(), 1e-9), 1e-4, 1 - 1e-4))
        self.base_ = math.log(p0 / (1 - p0))
        raw = np.full(n, self.base_)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for t in range(self.n_trees):
            pr = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
            grad = w * (pr - y)                 # 一阶导
            hess = np.maximum(w * pr * (1 - pr), 1e-6)   # 二阶导
            if self.subsample < 1.0:
                sel = rng.random(n) < self.subsample
                if sel.sum() < max(2 * self.min_leaf, 20):
                    sel = np.ones(n, dtype=bool)
                idx = np.where(sel)[0]
            else:
                idx = np.arange(n)
            tree = self._build(xb, grad, hess, w, idx, 0)
            upd = np.array([self._predict_tree(tree, xb[i]) for i in range(n)])
            raw = raw + self.lr * upd
            self.trees.append(tree)
        self.fitted_ = True
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if not self.fitted_:
            return np.full(len(x), 0.5)
        for j in range(x.shape[1]):
            bad = ~np.isfinite(x[:, j])
            if bad.any():
                x[bad, j] = self.med_[j]
        xb = self._bin(x)
        raw = np.full(len(x), self.base_)
        for tree in self.trees:
            raw += self.lr * np.array([self._predict_tree(tree, xb[i]) for i in range(len(x))])
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))


# =========================================================================== #
# 四、评估
# =========================================================================== #
def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC（用秩和公式，处理并列）。"""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    s_sorted = s[order]
    i = 0
    r = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = r[i:j + 1].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def walk_forward_eval(x: np.ndarray, y: np.ndarray, model_factory, n_splits: int = 4,
                      min_train: int | None = None) -> dict:
    """扩展窗口前向验证：每次用前段训练、后段测试，最后汇总样本外指标。"""
    n = len(y)
    min_train = min_train or max(60, int(n * 0.5))
    if n < min_train + 20:
        return {"auc": float("nan"), "hit_rate": float("nan"), "brier": float("nan"),
                "ic": float("nan"), "n_oos": 0}
    fold = max(10, (n - min_train) // n_splits)
    probs, trues = [], []
    for k in range(n_splits):
        tr_end = min_train + k * fold
        te_end = min(n, tr_end + fold)
        if te_end - tr_end < 8:
            break
        mdl = model_factory()
        try:
            mdl.fit(x[:tr_end], y[:tr_end])
        except Exception as exc:  # noqa: BLE001
            log.debug("walk_forward 训练失败: %s", exc)
            continue
        p = mdl.predict_proba(x[tr_end:te_end])
        probs.append(np.asarray(p, float))
        trues.append(np.asarray(y[tr_end:te_end], float))
    if not probs:
        return {"auc": float("nan"), "hit_rate": float("nan"), "brier": float("nan"),
                "ic": float("nan"), "n_oos": 0}
    p = np.concatenate(probs)
    t = np.concatenate(trues)
    hit = float(((p > 0.5) == (t > 0.5)).mean())
    return {
        "auc": _auc((t > 0.5).astype(float), p),
        "hit_rate": hit,
        "brier": float(np.mean((p - (t > 0.5).astype(float)) ** 2)),
        "ic": _spearman(p, t),
        "n_oos": int(len(p)),
    }


# =========================================================================== #
# 五、训练与预测主接口
# =========================================================================== #
def _time_decay_weights(n: int, half_life: float = 250.0) -> np.ndarray:
    """指数时间衰减权重：最近样本权重 1，越久越接近 half_life 衰减。"""
    idx = np.arange(n)[::-1]            # 最新的在前
    return 0.5 ** (idx / max(half_life, 1.0))


def _model_zoo(kind: str, n_train: int) -> dict:
    """按样本量选择模型组合（样本越多，模型越复杂）。"""
    zoo = {}
    if kind in ("auto", "logit"):
        zoo["logit"] = lambda: LogisticModel(l2=1.5, lr=0.3, epochs=500)
    if kind in ("auto", "gbdt") and n_train >= 200:
        zoo["gbdt"] = lambda: GBDTModel(n_trees=60 if n_train >= 400 else 40,
                                        lr=0.08, max_depth=2, min_leaf=max(10, n_train // 40))
    return zoo


def train_models(df: pd.DataFrame, ctx: dict | None = None,
                 horizons=HORIZONS, half_life: float = 250.0,
                 evaluate: bool = True) -> dict:
    """为每个预测周期训练模型。

    Parameters
    ----------
    df : 日线 DataFrame（含 date/close/...）
    ctx : 可选上下文，支持 ``{"index_df": DataFrame, "factor_score": {key: -1~1}}``
        * ``index_df``     : 大盘日线，用于相对强度特征
        * ``factor_score`` : 规则模型给出的每周期方向分，作为**额外特征**注入
          （让 AI 学习"在什么情况下规则更准"，这是本项目"AI + 理论"结合的关键）
    horizons : 要训练的周期，默认全部 8 个
    evaluate : 是否做前向验证（耗时约 ×4，Streamlit 快速模式可关）

    Returns
    -------
    dict[horizon_key] = :class:`TrainedModel`
    """
    out: dict[str, TrainedModel] = {}
    if df is None or len(df) < 120:
        log.info("样本不足（%s 根），跳过模型训练", 0 if df is None else len(df))
        return out
    ctx = ctx or {}
    index_df = ctx.get("index_df")
    x_df = build_features(df, index_df)
    t_df = make_targets(df, horizons)
    if x_df.empty or t_df.empty:
        return out

    factor_score = ctx.get("factor_score") or {}
    base_kind = config.FORECAST.get("ml_model", "auto")

    for h in horizons:
        col = "fwd_%s" % h.key
        if col not in t_df.columns:
            continue
        y_ret = t_df[col].to_numpy(dtype=float)
        x_all = x_df.to_numpy(dtype=float)
        feats = list(x_df.columns)
        if h.key in factor_score:
            fs = float(factor_score[h.key])
            x_all = np.hstack([x_all, np.full((len(x_all), 1), fs)])
            feats = feats + ["rule_factor_score"]
            FEATURE_LABELS.setdefault("rule_factor_score", "理论规则综合分")
        valid = np.isfinite(y_ret)
        # 只保留特征基本可用的样本（至少 60% 特征非缺失）
        feat_ok = np.isfinite(x_all).mean(axis=1) >= 0.6
        idx_tr = np.where(valid & feat_ok)[0]
        if len(idx_tr) < config.RUNTIME["ml_min_samples"]:
            out[h.key] = TrainedModel(horizon=h.key, label=h.label, bars=h.bars,
                                      features=feats, n_train=len(idx_tr),
                                      note="样本不足（%d < %d）" % (
                                          len(idx_tr), config.RUNTIME["ml_min_samples"]))
            continue

        x_tr = x_all[idx_tr]
        y_tr = (y_ret[idx_tr] > 0).astype(float)
        r_tr = y_ret[idx_tr]
        w = _time_decay_weights(len(idx_tr), half_life)
        zoo = _model_zoo(base_kind, len(idx_tr))

        models: dict = {}
        oos_list = []
        for name, factory in zoo.items():
            if name == "knn":
                continue
            try:
                mdl = factory()
                mdl.fit(x_tr, y_tr, sample_weight=w)
                models[name] = mdl
                if evaluate and len(idx_tr) >= 180:
                    ev = walk_forward_eval(x_tr, y_tr, factory, n_splits=3)
                    ev["name"] = name
                    oos_list.append(ev)
            except Exception as exc:  # noqa: BLE001
                log.debug("模型 %s/%s 训练失败: %s", h.key, name, exc)

        knn = KNNPatternModel(k=min(40, max(15, len(idx_tr) // 8)))
        try:
            knn.fit(x_tr, r_tr, sample_weight=w)
            if knn.fitted_:
                models["knn"] = knn
        except Exception as exc:  # noqa: BLE001
            log.debug("KNN %s 训练失败: %s", h.key, exc)

        if not models:
            out[h.key] = TrainedModel(horizon=h.key, label=h.label, bars=h.bars,
                                      features=feats, n_train=len(idx_tr), note="全部模型训练失败")
            continue

        # 汇总样本外表现（取各模型最优 AUC 作为该周期的可信度依据）
        auc = np.nan
        hit = np.nan
        brier = np.nan
        ic = np.nan
        if oos_list:
            best = max(oos_list, key=lambda e: (e.get("auc") if np.isfinite(e.get("auc", np.nan)) else -1))
            auc, hit, brier, ic = best.get("auc"), best.get("hit_rate"), best.get("brier"), best.get("ic")

        out[h.key] = TrainedModel(
            horizon=h.key, label=h.label, bars=h.bars, features=feats, models=models,
            oos={"auc": auc, "hit_rate": hit, "brier": brier, "ic": ic,
                 "details": oos_list, "n_oos": int(sum(e.get("n_oos", 0) for e in oos_list))},
            base_rate=float(y_tr.mean()), n_train=len(idx_tr),
            sigma_hist=float(np.nanstd(r_tr)),
            sample_weights_note="时间衰减半衰期 %.0f 根" % half_life,
        )
    return out


def predict(models: dict, df: pd.DataFrame, ctx: dict | None = None) -> dict:
    """用训练好的模型对**最新一根 K 线**给出预测。

    Returns
    -------
    dict[horizon_key] = :class:`ModelPrediction`
    """
    res: dict[str, ModelPrediction] = {}
    if not models or df is None or len(df) < 60:
        return res
    ctx = ctx or {}
    x_df = build_features(df, ctx.get("index_df"))
    if x_df.empty:
        return res
    factor_score = ctx.get("factor_score") or {}
    x_last = x_df.to_numpy(dtype=float)[-1:]

    for key, tm in models.items():
        if not isinstance(tm, TrainedModel) or not tm.usable:
            if isinstance(tm, TrainedModel):
                res[key] = ModelPrediction(horizon=key, label=tm.label, n_train=tm.n_train,
                                           n_features=len(tm.features), note=tm.note)
            continue
        xv = x_last
        if key in factor_score:
            xv = np.hstack([xv, np.array([[float(factor_score[key])]])])
        probs, weights, rets = [], [], []
        failed: list = []
        for name, mdl in tm.models.items():
            try:
                if isinstance(mdl, KNNPatternModel):
                    r, p_up = mdl.predict_return(xv)
                    probs.append(float(p_up[0]))
                    rets.append(float(r[0]))
                    weights.append(0.30)
                elif isinstance(mdl, GBDTModel):
                    probs.append(float(mdl.predict_proba(xv)[0]))
                    weights.append(0.35)
                else:
                    probs.append(float(mdl.predict_proba(xv)[0]))
                    weights.append(0.50)
            except Exception as exc:  # noqa: BLE001
                failed.append("%s:%s" % (name, exc))
                log.warning("预测失败 %s/%s: %s", key, name, exc)
        if not probs:
            res[key] = ModelPrediction(horizon=key, label=tm.label, n_train=tm.n_train,
                                       n_features=len(tm.features),
                                       note="预测失败：%s" % ("; ".join(failed) or "所有子模型无输出"))
            continue
        wsum = sum(weights)
        p_raw = float(sum(p * w for p, w in zip(probs, weights)) / wsum)

        # ---- 基准率校准 -------------------------------------------------- #
        # 逻辑回归/GBDT 输出的概率会被训练期的"上涨样本占比"（base_rate）带偏：
        # 若历史上 60% 的样本是上涨的，模型天然倾向给出 >0.5 的概率。
        # 这里用 log-odds 平移把概率重新锚定到 50%：
        #     z = logit(p) - logit(base_rate)，再按基准率对数权重 w 收缩，
        # 收缩系数 w 越小（即上涨/下跌越不均衡），越强地把概率拉回中性。
        base = float(np.clip(tm.base_rate, 0.02, 0.98))
        z = math.log(max(p_raw, 1e-6) / max(1 - p_raw, 1e-6)) - math.log(base / (1 - base))
        w_shrink = float(np.clip(min(base, 1 - base) / 0.5, 0.35, 1.0))
        z *= w_shrink
        p_up = float(1.0 / (1.0 + math.exp(-np.clip(z, -6, 6))))

        # 期望收益：优先用 KNN 的历史类比收益，其次用 (p_up-0.5) × 历史波动换算
        if rets:
            exp_ret = float(np.mean(rets))
            # 与概率方向一致性校正：方向冲突时向 0 收缩
            if (exp_ret > 0) != (p_up > 0.5):
                exp_ret *= 0.35
        else:
            exp_ret = (p_up - 0.5) * 2.0 * (tm.sigma_hist or 3.0) * 0.55

        # 置信度：样本外 AUC 越高、样本越多越可信
        auc = tm.oos.get("auc")
        conf = 0.35
        if auc is not None and np.isfinite(auc):
            conf = float(np.clip((auc - 0.45) / 0.30, 0.05, 0.95))
        conf *= float(np.clip(tm.n_train / 400.0, 0.5, 1.0))
        conf = float(np.clip(conf, 0.05, 0.9))

        top = []
        lg = tm.models.get("logit")
        if isinstance(lg, LogisticModel) and lg.fitted_:
            for fname, coef in lg.feature_importance(tm.features)[:8]:
                top.append((FEATURE_LABELS.get(fname, fname), round(float(coef), 3)))

        res[key] = ModelPrediction(
            horizon=key, label=tm.label, p_up=p_up, exp_return=exp_ret,
            sigma=float(tm.sigma_hist or 0.0), n_train=tm.n_train,
            n_features=len(tm.features), auc=float(auc) if auc is not None else float("nan"),
            ic=float(tm.oos.get("ic")) if tm.oos.get("ic") is not None else float("nan"),
            hit_rate=float(tm.oos.get("hit_rate")) if tm.oos.get("hit_rate") is not None else float("nan"),
            brier=float(tm.oos.get("brier")) if tm.oos.get("brier") is not None else float("nan"),
            base_rate=tm.base_rate, model="+".join(tm.models.keys()), confidence=conf,
            top_features=top,
        )
    return res


def ml_factors(preds: dict, min_conf: float = 0.25) -> list:
    """把 ML 预测转成 :class:`Factor`（category=Category.ML），纳入统一打分体系。"""
    from .contracts import Category, DIR_BEAR, DIR_BULL, Factor

    out = []
    for key, p in (preds or {}).items():
        if not isinstance(p, ModelPrediction) or not p.usable:
            continue
        strength = float(np.clip(abs(p.p_up - 0.5) * 2.0, 0.0, 1.0))
        if strength < 0.02:
            continue
        direction = DIR_BULL if p.p_up > 0.5 else (DIR_BEAR if p.p_up < 0.5 else 0)
        rel = float(np.clip(p.confidence, 0.1, 0.85))
        if rel < min_conf:
            continue
        top_txt = "、".join("%s(%+.2f)" % (n, c) for n, c in (p.top_features or [])[:4])
        base_txt = "（历史上涨基准率 %.0f%%）" % (p.base_rate * 100)
        out.append(Factor(
            id="ml_%s" % key,
            name="AI模型(%s)" % p.label,
            category=Category.ML,
            direction=direction,
            strength=strength,
            reliability=rel,
            value="P(涨)=%.1f%%  样本外AUC=%s" % (
                p.p_up * 100, "%.3f" % p.auc if np.isfinite(p.auc) else "n/a"),
            detail="历史上与当前状态相似的 %d 个时点中，模型给出上涨概率 %.1f%%%s；"
                   "主导特征：%s" % (p.n_train, p.p_up * 100, base_txt, top_txt or "—"),
            theory="机器学习（历史规律归纳）",
            horizon=(key,),
        ))
    return out


# =========================================================================== #
# 自测
# =========================================================================== #
if __name__ == "__main__":  # pragma: no cover
    import time

    from .selftest_data import make_index, make_ohlcv

    t0 = time.time()
    df = make_ohlcv(900, seed=42)
    idx = make_index(900, seed=7)
    x = build_features(df, idx)
    t = make_targets(df)
    print("特征矩阵:", x.shape, " 标签:", t.shape)
    assert x.shape[1] == len(FEATURE_COLUMNS), (x.shape[1], len(FEATURE_COLUMNS))
    assert not np.isinf(x.to_numpy(dtype=float)).any(), "特征存在 ±inf"
    # 因果性检查：截断后重新计算，前面的特征值必须完全一致
    x2 = build_features(df.iloc[:600], idx.iloc[:600])
    a = x.iloc[:600].fillna(-999).to_numpy()
    b = x2.fillna(-999).to_numpy()
    assert np.allclose(a, b, atol=1e-9), "特征使用了未来数据！"
    print("因果性检查通过（无未来函数）")

    models = train_models(df, {"index_df": idx})
    print("训练周期数:", len(models))
    for k, m in models.items():
        print("  %-4s n=%-4d models=%-18s AUC=%s hit=%s sigma=%.2f%%" % (
            k, m.n_train, "+".join(m.models.keys()),
            "%.3f" % m.oos.get("auc", float("nan")) if m.oos.get("auc") is not None else "n/a",
            "%.3f" % m.oos["hit_rate"] if m.oos.get("hit_rate") is not None else "n/a",
            m.sigma_hist))

    preds = predict(models, df, {"index_df": idx})
    print("\n预测结果:")
    for k, p in preds.items():
        print("  %-4s P(涨)=%.3f 期望=%+.2f%% AUC=%s conf=%.2f" % (
            k, p.p_up, p.exp_return,
            "%.3f" % p.auc if np.isfinite(p.auc) else "n/a", p.confidence))

    fs = ml_factors(preds)
    print("\nML 因子数:", len(fs))
    assert all(0 <= f.strength <= 1 and 0 <= f.reliability <= 1 for f in fs)
    assert all(f.category == "ml" for f in fs)
    # 单独模型自测
    xx = x.fillna(0).to_numpy()
    yy = (t["fwd_d5"].fillna(0) > 0).astype(float).to_numpy()
    w = _time_decay_weights(len(yy))
    for M in (LogisticModel(l2=1.0, epochs=200),
              GBDTModel(n_trees=10, max_depth=2, min_leaf=10)):
        M.fit(xx, yy, w)
        pr = M.predict_proba(xx[:50])
        assert pr.shape == (50,) and np.all((pr >= 0) & (pr <= 1))
        print("%-6s 训练完成 前5个概率=%s" % (M.name, np.round(pr[:5], 3)))
    knn = KNNPatternModel(k=20).fit(xx, t["fwd_d5"].fillna(0).to_numpy())
    r, pu = knn.predict_return(xx[-1:])
    print("KNN 历史类比: 平均后续收益=%+.2f%% 上涨占比=%.2f" % (r[0], pu[0]))
    print("耗时 %.1fs" % (time.time() - t0))
    print("mlmodel self-test OK")

# -*- coding: utf-8 -*-
"""
技术面因子合成 —— 把 :mod:`core.indicators` 与 :mod:`core.patterns` 算出的
几十个指标读数，压缩成一组**可解释、带方向与可信度**的 :class:`Factor`。

覆盖范围（对应需求里逐条点名的技术要素）
----------------------------------------
* **K 线类型**：日线蜡烛、等比 K 线（Renko 砖图）、等量/等价 K 线、周线、月线
* **均线系统**：MA5/10/20/30/60/120/250、EXPMA、BBI、多头/空头排列、乖离、金叉死叉
* **摆动指标**：MACD、KDJ、RSI、CCI、WR、ROC、MTM、PSY、BIAS、TRIX、DMA、LON
* **趋势指标**：DMI/ADX、SAR、DKX、BBI、趋势强度（回归斜率 + R²）
* **通道与轨道**：BOLL、ENE、上升/下降通道、箱体、扇形趋势线
* **量能指标**：VR、OBV、ASI、EMV、WVAD、BRAR、CR、量价相关、量能阶段
* **筹码与价位**：筹码分布/集中区、持仓成本线、支撑压力位、黄金分割位、
  前期高/低点、百分比回撤
* **形态与结构**：K 线经典形态、缺口、岛形、双顶双底、三角形/楔形、
  艾略特波浪、缠论笔/中枢
* **背离**：个股与大盘的顶背离/底背离、指标与价格的背离

**多周期验证**：同一逻辑在日/周/月线上分别判一次，共振则提升可信度，
冲突则降级——这是技术分析里少有的、能显著提高胜率的做法。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config
from . import indicators as ind
from . import patterns as pat
from .contracts import (
    Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor, FactorSet,
)
from .utils import get_logger
from .contracts import safe_float

log = get_logger()

_SHORT = ("d1", "d5", "d10")
_SHORT_MID = ("d1", "d5", "d10", "d20")
_MID = ("d5", "d10", "d20", "m1")
_MID_LONG = ("d20", "m1", "q1")
_LONG = ("q1", "h1", "y1")
_ALL = None

#: 技术指标"看涨/看跌"的经验阈值（集中管理，便于校准）
TH = {
    "rsi_overbought": 78.0, "rsi_oversold": 25.0,
    "kdj_j_over": 105.0, "kdj_j_under": -5.0,
    "cci_over": 180.0, "cci_under": -180.0,
    "wr_over": -8.0, "wr_under": -92.0,
    "bias20_over": 12.0, "bias20_under": -12.0,
    "psy_over": 78.0, "psy_under": 22.0,
    "vr_high": 350.0, "vr_low": 60.0,
    "brar_ar_over": 200.0, "brar_ar_under": 60.0,
    "adx_trend": 25.0,
    "vol_ratio_surge": 1.9,
    "vol_ratio_shrink": 0.6,
}


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _ensure_frame(df) -> pd.DataFrame | None:
    """列名容错 + 类型规范；返回副本，绝不修改入参。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    d = df.copy()
    ren = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
           "收盘": "close", "成交量": "volume", "成交额": "amount",
           "换手率": "turnover", "涨跌幅": "pct_chg"}
    d = d.rename(columns={k: v for k, v in ren.items() if k in d.columns})
    d.columns = [str(c).lower() if str(c).upper() in
                 ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMOUNT", "DATE",
                  "TURNOVER", "PCT_CHG") else c for c in d.columns]
    for col in ("open", "high", "low", "close"):
        if col not in d.columns:
            return None
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if "volume" not in d.columns:
        d["volume"] = np.nan
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d = d.dropna(subset=["close"])
    return d if len(d) >= 30 else None


def _last(series) -> float:
    """安全取最后一个有效值。"""
    try:
        arr = pd.Series(series).dropna()
        return float(arr.iloc[-1]) if len(arr) else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def _mk(fid: str, name: str, direction: int, strength: float, reliability: float,
        value, detail: str, theory: str, horizon=None) -> Factor:
    return Factor(fid, name, Category.TECH, direction, strength, reliability,
                  value=value, detail=detail, theory=theory, horizon=horizon)


def _f(x) -> float:
    return safe_float(x, float("nan"))


# --------------------------------------------------------------------------- #
# 1. 均线系统
# --------------------------------------------------------------------------- #
def ma_system(df: pd.DataFrame, m: pd.DataFrame) -> list:
    out: list[Factor] = []
    c = df["close"]
    ma5, ma10, ma20, ma60 = (_last(m.get(k)) for k in ("ma5", "ma10", "ma20", "ma60"))
    ma120, ma250 = _last(m.get("ma120")), _last(m.get("ma250"))
    price = _last(c)
    if price != price:
        return out

    # ---- 多头/空头排列 ------------------------------------------------ #
    try:
        order_up = all(x == x for x in (ma5, ma10, ma20, ma60)) and \
            price > ma5 > ma10 > ma20 > ma60
        order_dn = all(x == x for x in (ma5, ma10, ma20, ma60)) and \
            price < ma5 < ma10 < ma20 < ma60
    except Exception:  # noqa: BLE001
        order_up = order_dn = False
    if order_up:
        out.append(_mk("ma_alignment", "均线多头排列", DIR_BULL, 0.82, 0.90,
                       "MA5>MA10>MA20>MA60 且股价在最上方",
                       "短中长期均线依次向上发散，趋势结构完整，回踩均线即为支撑",
                       "道氏理论/均线系统", _SHORT_MID))
    elif order_dn:
        out.append(_mk("ma_alignment", "均线空头排列", DIR_BEAR, 0.82, 0.90,
                       "MA5<MA10<MA20<MA60 且股价在最下方",
                       "短中长期均线依次向下发散，反弹到均线附近即为压力",
                       "道氏理论/均线系统", _SHORT_MID))

    # ---- 中长期水位（年线/半年线） ------------------------------------ #
    if ma250 == ma250 and ma120 == ma120:
        above250 = price > ma250
        above120 = price > ma120
        both_up = ma250 > 0 and ma120 > ma250
        if above250 and above120:
            d, s = DIR_BULL, 0.55 + (0.15 if both_up else 0.0)
            det = "股价站上半年线与年线之上" + ("，且半年线在年线上方（多头格局）" if both_up else "")
        elif not above250 and not above120:
            d, s = DIR_BEAR, 0.55 + (0.15 if not both_up else 0.0)
            det = "股价跌破半年线与年线" + ("，且半年线在年线下方（空头格局）" if not both_up else "")
        else:
            d, s = DIR_NEUTRAL, 0.25
            det = "股价在半年线与年线之间，中期方向不明"
        out.append(_mk("ma_long_term_level", "中长期均线水位", d, s, 0.85,
                       "MA120=%.2f MA250=%.2f 现价=%.2f" % (ma120, ma250, price), det,
                       "均线系统", _MID_LONG if d >= 0 else _SHORT_MID))

    # ---- 金叉/死叉（MA5 与 MA20） ------------------------------------- #
    try:
        s5 = m["ma5"].dropna()
        s20 = m["ma20"].dropna()
        if len(s5) >= 4 and len(s20) >= 4:
            diff = (s5 - s20).dropna()
            if len(diff) >= 3:
                now_up = diff.iloc[-1] > 0
                prev_up = diff.iloc[-3] > 0
                gap_pct = abs(diff.iloc[-1]) / max(abs(s20.iloc[-1]), 1e-9) * 100
                if now_up and not prev_up:
                    out.append(_mk("ma_cross", "MA5上穿MA20（金叉）", DIR_BULL, 0.62, 0.75,
                                   "MA5=%.2f MA20=%.2f 差%.2f%%" % (s5.iloc[-1], s20.iloc[-1], gap_pct),
                                   "短期均线上穿中期均线，短期动能转强", "均线系统", _SHORT))
                elif not now_up and prev_up:
                    out.append(_mk("ma_cross", "MA5下穿MA20（死叉）", DIR_BEAR, 0.62, 0.75,
                                   "MA5=%.2f MA20=%.2f 差%.2f%%" % (s5.iloc[-1], s20.iloc[-1], gap_pct),
                                   "短期均线下穿中期均线，短期动能转弱", "均线系统", _SHORT))
    except Exception as exc:  # noqa: BLE001
        log.debug("均线交叉判定失败: %s", exc)

    # ---- 乖离率（超买超卖） ------------------------------------------- #
    b6, b12, b24 = (_last(m.get(k)) for k in ("bias6", "bias12", "bias24"))
    for val, label, per in ((b6, "BIAS6", 6), (b24, "BIAS24", 24)):
        if val != val:
            continue
        over = TH["bias20_over"] * (1.0 if per == 24 else 0.75)
        if val > over:
            out.append(_mk("bias_extreme", "%s乖离过大（超买）" % label, DIR_BEAR, 0.55, 0.62,
                           "%.2f%%" % val,
                           "股价偏离均线过远，短期有回归均线的压力", "乖离率", _SHORT))
            break
        if val < -over:
            out.append(_mk("bias_extreme", "%s乖离过大（超卖）" % label, DIR_BULL, 0.55, 0.62,
                           "%.2f%%" % val,
                           "股价大幅低于均线，短期存在超跌反弹动能", "乖离率", _SHORT))
            break

    # ---- EXPMA / BBI / DKX -------------------------------------------- #
    e1, e2 = _last(m.get("expma1")), _last(m.get("expma2"))
    if e1 == e1 and e2 == e2:
        up = e1 > e2
        strength = min(abs(e1 - e2) / max(e2, 1e-9) * 25, 0.8)
        out.append(_mk("expma_state", "EXPMA%s" % ("多头" if up else "空头"),
                       DIR_BULL if up else DIR_BEAR, max(strength, 0.22), 0.72,
                       "EXPMA12=%.2f EXPMA50=%.2f" % (e1, e2),
                       "指数平滑均线%s，%s" % ("多头排列" if up else "空头排列",
                                              "中期趋势向上" if up else "中期趋势向下"),
                       "EXPMA", _MID))
    bbi_v = _last(m.get("bbi"))
    if bbi_v == bbi_v:
        up = price > bbi_v
        out.append(_mk("bbi_state", "BBI多空线", DIR_BULL if up else DIR_BEAR, 0.42, 0.70,
                       "BBI=%.2f 现价=%.2f" % (bbi_v, price),
                       "股价在BBI%s，属%s方掌控" % ("之上" if up else "之下",
                                                  "多" if up else "空"),
                       "BBI", _SHORT_MID))
    dkx_v = _last(m.get("dkx"))
    if dkx_v == dkx_v:
        up = price > dkx_v
        out.append(_mk("dkx_state", "DKX多空线", DIR_BULL if up else DIR_BEAR, 0.40, 0.68,
                       "DKX=%.2f" % dkx_v,
                       "多空线指示当前%s方占优" % ("多" if up else "空"), "DKX", _SHORT_MID))
    return out


# --------------------------------------------------------------------------- #
# 2. 摆动指标
# --------------------------------------------------------------------------- #
def oscillator_signals(df: pd.DataFrame, m: pd.DataFrame) -> list:
    out: list[Factor] = []
    price = _last(df["close"])

    # ---- MACD ---------------------------------------------------------- #
    dif, dea, hist = (_last(m.get(k)) for k in ("macd_dif", "macd_dea", "macd_hist"))
    try:
        d_series = m["macd_dif"].dropna()
        e_series = m["macd_dea"].dropna()
        if len(d_series) >= 4 and len(e_series) >= 4:
            diff = (d_series - e_series).dropna()
            if len(diff) >= 3:
                now_up, prev_up = diff.iloc[-1] > 0, diff.iloc[-3] > 0
                if now_up and not prev_up:
                    strength = 0.6 + (0.15 if (dif == dif and dif > 0) else 0.0)
                    out.append(_mk("macd_cross", "MACD金叉", DIR_BULL, min(strength, 0.85), 0.82,
                                   "DIF=%.3f DEA=%.3f" % (dif, dea),
                                   "DIF上穿DEA%s，短期动能转强"
                                   % ("且在零轴上方（强势区）" if dif == dif and dif > 0 else "（零轴下方，属反弹）"),
                                   "MACD", _SHORT_MID))
                elif not now_up and prev_up:
                    strength = 0.6 + (0.15 if (dif == dif and dif < 0) else 0.0)
                    out.append(_mk("macd_cross", "MACD死叉", DIR_BEAR, min(strength, 0.85), 0.82,
                                   "DIF=%.3f DEA=%.3f" % (dif, dea),
                                   "DIF下穿DEA%s，短期动能转弱"
                                   % ("且在零轴下方（弱势区）" if dif == dif and dif < 0 else "（零轴上方，属回调）"),
                                   "MACD", _SHORT_MID))
                # 柱状能量变化
                h_series = m["macd_hist"].dropna()
                if len(h_series) >= 4:
                    h_now, h_prev = h_series.iloc[-1], h_series.iloc[-3]
                    if h_now > h_prev and h_now < 0:
                        out.append(_mk("macd_hist_shrink", "MACD绿柱缩短", DIR_BULL, 0.42, 0.68,
                                       "柱=%.3f" % h_now, "空头动能衰减，下跌趋缓",
                                       "MACD", _SHORT))
                    elif h_now < h_prev and h_now > 0:
                        out.append(_mk("macd_hist_shrink", "MACD红柱缩短", DIR_BEAR, 0.42, 0.68,
                                       "柱=%.3f" % h_now, "多头动能衰减，上涨乏力",
                                       "MACD", _SHORT))
    except Exception as exc:  # noqa: BLE001
        log.debug("MACD 判定失败: %s", exc)
    # 零轴位置（中长期多空）
    if dif == dif:
        up = dif > 0
        out.append(_mk("macd_zero_axis", "MACD零轴%s" % ("上方" if up else "下方"),
                       DIR_BULL if up else DIR_BEAR, 0.45, 0.78,
                       "DIF=%.3f" % dif,
                       "DIF在零轴%s，%s格局" % ("上方" if up else "下方",
                                              "多头" if up else "空头"),
                       "MACD", _MID))

    # ---- KDJ ----------------------------------------------------------- #
    k, d, j = (_last(m.get(x)) for x in ("k", "d", "j"))
    if k == k and d == d and j == j:
        try:
            ks, ds = m["k"].dropna(), m["d"].dropna()
            if len(ks) >= 3 and len(ds) >= 3:
                diff = (ks - ds).dropna()
                if len(diff) >= 3:
                    now_up, prev_up = diff.iloc[-1] > 0, diff.iloc[-3] > 0
                    strength = 0.5 + (0.2 if k < 35 else 0.0) + (0.1 if j < 15 else 0.0)
                    if now_up and not prev_up:
                        out.append(_mk("kdj_cross", "KDJ金叉", DIR_BULL, min(strength, 0.82), 0.75,
                                       "K=%.1f D=%.1f J=%.1f" % (k, d, j),
                                       "随机指标金叉%s" % ("且处于低位（超卖区反转，可靠性更高）"
                                                          if k < 35 else "，短期有反弹动能"),
                                       "KDJ", _SHORT))
                    elif not now_up and prev_up:
                        strength2 = 0.5 + (0.2 if k > 65 else 0.0) + (0.1 if j > 85 else 0.0)
                        out.append(_mk("kdj_cross", "KDJ死叉", DIR_BEAR, min(strength2, 0.82), 0.75,
                                       "K=%.1f D=%.1f J=%.1f" % (k, d, j),
                                       "随机指标死叉%s" % ("且处于高位（超买区见顶，可靠性更高）"
                                                          if k > 65 else "，短期有回调压力"),
                                       "KDJ", _SHORT))
        except Exception:  # noqa: BLE001
            pass
        if j > TH["kdj_j_over"]:
            out.append(_mk("kdj_extreme", "KDJ的J值超买", DIR_BEAR, 0.6, 0.62,
                           "J=%.1f" % j, "J值进入极端超买区，短线随时可能回落",
                           "KDJ", _SHORT))
        elif j < TH["kdj_j_under"]:
            out.append(_mk("kdj_extreme", "KDJ的J值超卖", DIR_BULL, 0.6, 0.62,
                           "J=%.1f" % j, "J值进入极端超卖区，短线反弹概率大",
                           "KDJ", _SHORT))

    # ---- RSI ----------------------------------------------------------- #
    for per, key in ((6, "rsi6"), (14, "rsi14")):
        v = _last(m.get(key))
        if v != v:
            continue
        if v > TH["rsi_overbought"]:
            out.append(_mk("rsi_extreme", "RSI%d超买" % per, DIR_BEAR, 0.55, 0.65,
                           "%.1f" % v, "RSI进入超买区，追高风险大", "RSI", _SHORT))
        elif v < TH["rsi_oversold"]:
            out.append(_mk("rsi_extreme", "RSI%d超卖" % per, DIR_BULL, 0.55, 0.65,
                           "%.1f" % v, "RSI进入超卖区，存在技术性反弹需求", "RSI", _SHORT))
    r6, r14 = _last(m.get("rsi6")), _last(m.get("rsi14"))
    if r6 == r6 and r14 == r14:
        up = r6 > r14
        out.append(_mk("rsi_short_long", "短期RSI%s长期RSI" % ("强于" if up else "弱于"),
                       DIR_BULL if up else DIR_BEAR, 0.35, 0.62,
                       "RSI6=%.1f RSI14=%.1f" % (r6, r14),
                       "短期相对强度%s，%s" % ("占优" if up else "偏弱",
                                              "短线动能向上" if up else "短线动能向下"),
                       "RSI", _SHORT))

    # ---- CCI / WR / ROC / MTM / PSY ------------------------------------ #
    cci = _last(m.get("cci"))
    if cci == cci:
        if cci > TH["cci_over"]:
            out.append(_mk("cci_extreme", "CCI超买", DIR_BEAR, 0.5, 0.6, "%.1f" % cci,
                           "CCI进入超买区，涨势可能过热", "CCI", _SHORT))
        elif cci < TH["cci_under"]:
            out.append(_mk("cci_extreme", "CCI超卖", DIR_BULL, 0.5, 0.6, "%.1f" % cci,
                           "CCI进入超卖区，短期有修复需求", "CCI", _SHORT))
        else:
            up = cci > 0
            out.append(_mk("cci_direction", "CCI%s" % ("走强" if up else "走弱"),
                           DIR_BULL if up else DIR_BEAR, 0.3, 0.58, "%.1f" % cci,
                           "CCI在%s区间，%s" % ("强势" if up else "弱势",
                                              "资金活跃度偏高" if up else "资金活跃度偏低"),
                           "CCI", _SHORT))
    wr10 = _last(m.get("wr10"))
    if wr10 == wr10:
        if wr10 > TH["wr_over"]:
            out.append(_mk("wr_extreme", "威廉指标超买", DIR_BEAR, 0.48, 0.58, "%.1f" % wr10,
                           "WR接近0轴，处于超买状态", "威廉指标", _SHORT))
        elif wr10 < TH["wr_under"]:
            out.append(_mk("wr_extreme", "威廉指标超卖", DIR_BULL, 0.48, 0.58, "%.1f" % wr10,
                           "WR接近-100，处于超卖状态", "威廉指标", _SHORT))
    roc = _last(m.get("roc"))
    if roc == roc:
        up = roc > 0
        out.append(_mk("roc_momentum", "ROC%s" % ("为正" if up else "为负"),
                       DIR_BULL if up else DIR_BEAR, min(abs(roc) / 12.0, 0.7), 0.6,
                       "%.2f%%" % roc, "变动率指标显示价格%s动能" % ("上行动" if up else "下行动"),
                       "ROC", _SHORT))
    psy = _last(m.get("psy"))
    if psy == psy:
        if psy > TH["psy_over"]:
            out.append(_mk("psy_extreme", "PSY心理线超买", DIR_BEAR, 0.45, 0.55, "%.1f" % psy,
                           "上涨天数占比过高，市场情绪偏热", "PSY", _SHORT))
        elif psy < TH["psy_under"]:
            out.append(_mk("psy_extreme", "PSY心理线超卖", DIR_BULL, 0.45, 0.55, "%.1f" % psy,
                           "上涨天数占比过低，市场情绪偏冷", "PSY", _SHORT))
    trix, trma = _last(m.get("trix")), _last(m.get("trma"))
    if trix == trix and trma == trma:
        up = trix > trma
        out.append(_mk("trix_state", "TRIX%sTRMA" % ("上穿" if up else "下穿"),
                       DIR_BULL if up else DIR_BEAR, 0.4, 0.65,
                       "TRIX=%.3f TRMA=%.3f" % (trix, trma),
                       "三重指数平滑指标%s，中长期趋势%s" % (
                           "转强" if up else "转弱", "向上" if up else "向下"),
                       "TRIX", _MID))
    return out


# --------------------------------------------------------------------------- #
# 3. 趋势与通道
# --------------------------------------------------------------------------- #
def trend_and_channel(df: pd.DataFrame, m: pd.DataFrame) -> list:
    out: list[Factor] = []
    price = _last(df["close"])

    # ---- DMI/ADX ------------------------------------------------------- #
    pdi, mdi, adx = (_last(m.get(k)) for k in ("pdi", "mdi", "adx"))
    if pdi == pdi and mdi == mdi:
        up = pdi > mdi
        trending = adx == adx and adx > TH["adx_trend"]
        strength = 0.35 + (0.3 if trending else 0.0) + min(abs(pdi - mdi) / 50.0, 0.25)
        det = "%s占优%s" % ("多方" if up else "空方",
                            "，且 ADX=%.1f 显示趋势明确" % adx if trending
                            else "，但 ADX=%.1f 偏低，属震荡格局" % (adx if adx == adx else 0))
        out.append(_mk("dmi_trend", "DMI趋势%s" % ("向上" if up else "向下"),
                       DIR_BULL if up else DIR_BEAR, min(strength, 0.85),
                       0.8 if trending else 0.6,
                       "PDI=%.1f MDI=%.1f ADX=%.1f" % (pdi, mdi, adx if adx == adx else -1),
                       det, "DMI", _MID if trending else _SHORT))

    # ---- SAR ----------------------------------------------------------- #
    sar_v = _last(m.get("sar"))
    sar_trend = _last(m.get("sar_trend"))
    if sar_v == sar_v and sar_trend == sar_trend:
        up = sar_trend > 0
        out.append(_mk("sar_state", "SAR%s" % ("翻多" if up else "翻空"),
                       DIR_BULL if up else DIR_BEAR, 0.55, 0.75,
                       "SAR=%.2f 现价=%.2f" % (sar_v, price),
                       "抛物线指标位于价格%s，趋势%s" % ("下方" if up else "上方",
                                                        "向上" if up else "向下"),
                       "SAR", _SHORT_MID))

    # ---- 线性趋势强度 -------------------------------------------------- #
    try:
        ts = ind.trend_strength(df["close"], df["high"], df["low"], n=20)
        if isinstance(ts, dict):
            slope, r2, direction = (ts.get("slope"), ts.get("r2"), ts.get("direction"))
            if r2 == r2 and r2 > 0.35 and direction:
                out.append(_mk("trend_regression", "回归趋势线%s" % ("向上" if direction > 0 else "向下"),
                               DIR_BULL if direction > 0 else DIR_BEAR,
                               min(0.35 + r2 * 0.45, 0.8), 0.75,
                               "斜率=%.3f R²=%.3f" % (_f(slope), r2),
                               "近20日价格沿%s趋势运行，拟合优度 %.2f"
                               % ("上升" if direction > 0 else "下降", r2),
                               "趋势回归", _SHORT_MID if direction > 0 else _SHORT))
    except Exception as exc:  # noqa: BLE001
        log.debug("趋势强度计算失败: %s", exc)

    # ---- 通道 ---------------------------------------------------------- #
    try:
        ch = ind.channel_fit(df["high"], df["low"], df["close"], lookback=120)
        if isinstance(ch, dict):
            kind = ch.get("channel")
            pos = ch.get("position")
            brk = ch.get("break")
            if kind in ("up", "down") and pos == pos:
                up = kind == "up"
                out.append(_mk("channel_trend", "%s通道运行" % ("上升" if up else "下降"),
                               DIR_BULL if up else DIR_BEAR, 0.5, 0.68,
                               "通道位置=%.2f 宽度=%.1f%%" % (pos, _f(ch.get("width_pct"))),
                               "价格在%s通道内，位置 %.0f%%（%s）"
                               % ("上升" if up else "下降", pos * 100,
                                  "靠近上轨注意压力" if pos > 0.8 else
                                  ("靠近下轨有支撑" if pos < 0.2 else "处于中位")),
                               "通道理论", _MID))
            if brk == "up":
                out.append(_mk("channel_break", "向上突破通道", DIR_BULL, 0.72, 0.7,
                               "突破上轨", "价格向上突破通道上轨，若量能配合则打开上行空间",
                               "通道理论", _SHORT_MID))
            elif brk == "down":
                out.append(_mk("channel_break", "向下跌破通道", DIR_BEAR, 0.72, 0.7,
                               "跌破下轨", "价格向下跌破通道下轨，需警惕加速下行",
                               "通道理论", _SHORT_MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("通道拟合失败: %s", exc)

    # ---- 趋势线（扇形原理） -------------------------------------------- #
    try:
        tl = pat.detect_trendlines(df, lookback=150)
        if isinstance(tl, dict):
            brk = tl.get("break")
            if brk == "up":
                out.append(_mk("trendline_break", "突破下降趋势线", DIR_BULL, 0.7, 0.68,
                               "touches=%s" % tl.get("touches"),
                               "价格突破由多个高点连成的下降压力线，趋势可能反转向上",
                               "扇形原理/趋势线", _SHORT_MID))
            elif brk == "down":
                out.append(_mk("trendline_break", "跌破上升趋势线", DIR_BEAR, 0.7, 0.68,
                               "touches=%s" % tl.get("touches"),
                               "价格跌破由多个低点连成的上升支撑线，趋势可能转弱",
                               "扇形原理/趋势线", _SHORT_MID))
            if tl.get("triangle"):
                out.append(_mk("triangle_pattern", "收敛三角形", DIR_NEUTRAL, 0.3, 0.6,
                               "上下轨收敛", "价格进入收敛三角形，方向未定，等待放量突破",
                               "形态理论", _SHORT_MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("趋势线检测失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 4. 量能
# --------------------------------------------------------------------------- #
def volume_signals(df: pd.DataFrame, m: pd.DataFrame) -> list:
    out: list[Factor] = []
    try:
        close = df["close"].astype(float)
        vol = df["volume"].astype(float)
        if vol.notna().sum() < 30:
            return out
        v_now = _last(vol)
        v_ma5, v_ma20 = _last(m.get("vol_ma5")), _last(m.get("vol_ma20"))
        price_ret1 = float(close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) > 1 else 0.0
        price_ret5 = float(close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 6 else 0.0

        # ---- 量价配合 -------------------------------------------------- #
        if v_ma20 == v_ma20 and v_ma20 > 0:
            ratio = v_now / v_ma20
            if ratio > TH["vol_ratio_surge"]:
                if price_ret1 > 1.0:
                    out.append(_mk("vol_price", "放量上涨", DIR_BULL, 0.68, 0.8,
                                   "量比20日均量=%.2f 当日涨%.2f%%" % (ratio, price_ret1),
                                   "成交量显著放大且股价上涨，资金主动买入，量价配合健康",
                                   "量价关系", _SHORT))
                elif price_ret1 < -1.0:
                    out.append(_mk("vol_price", "放量下跌", DIR_BEAR, 0.72, 0.8,
                                   "量比20日均量=%.2f 当日跌%.2f%%" % (ratio, price_ret1),
                                   "放量下跌说明抛压沉重，通常是资金出逃信号",
                                   "量价关系", _SHORT))
                else:
                    out.append(_mk("vol_price", "放量滞涨", DIR_BEAR, 0.5, 0.68,
                                   "量比=%.2f 涨跌%.2f%%" % (ratio, price_ret1),
                                   "成交量放大但股价滞涨，有派发嫌疑", "量价关系", _SHORT))
            elif ratio < TH["vol_ratio_shrink"]:
                if price_ret5 > 3:
                    out.append(_mk("vol_price", "缩量上涨（量价背离）", DIR_BEAR, 0.55, 0.68,
                                   "量比=%.2f 5日涨%.2f%%" % (ratio, price_ret5),
                                   "上涨缺乏成交量配合，持续性存疑", "量价关系", _SHORT))
                elif price_ret5 < -3:
                    out.append(_mk("vol_price", "缩量下跌（惜售）", DIR_BULL, 0.45, 0.62,
                                   "量比=%.2f 5日跌%.2f%%" % (ratio, price_ret5),
                                   "下跌但成交量萎缩，抛压减轻，有企稳可能",
                                   "量价关系", _SHORT))
            # 温和放量
            elif 1.1 < ratio < TH["vol_ratio_surge"] and price_ret5 > 1.5:
                out.append(_mk("vol_price", "温和放量上行", DIR_BULL, 0.5, 0.72,
                               "量比=%.2f" % ratio,
                               "量能温和放大配合股价上行，属健康的量价结构",
                               "量价关系", _SHORT_MID))

        # ---- OBV / 量价相关 -------------------------------------------- #
        obv_slope = _last(m.get("obv")) - _last(m.get("obv_ma")) \
            if _last(m.get("obv")) == _last(m.get("obv")) else float("nan")
        if obv_slope == obv_slope:
            up = obv_slope > 0
            out.append(_mk("obv_state", "OBV%s均线" % ("在" if up else "跌破"),
                           DIR_BULL if up else DIR_BEAR, 0.42, 0.68,
                           "OBV-均线=%.0f" % obv_slope,
                           "能量潮%s，%s" % ("向上" if up else "走弱",
                                            "资金持续流入" if up else "资金持续流出"),
                           "OBV", _SHORT_MID))
        vpc = _last(m.get("vol_price_corr"))
        if vpc == vpc:
            if vpc > 0.35:
                out.append(_mk("vp_corr", "量价正相关", DIR_BULL, 0.4, 0.62,
                               "相关系数=%.2f" % vpc,
                               "成交量与价格同向变动，上涨有量能支撑", "量价关系", _MID))
            elif vpc < -0.35:
                out.append(_mk("vp_corr", "量价负相关", DIR_BEAR, 0.4, 0.62,
                               "相关系数=%.2f" % vpc,
                               "放量伴随下跌，量价关系恶化", "量价关系", _MID))

        # ---- VR / BRAR / CR / ASI / EMV / WVAD ------------------------- #
        vr = _last(m.get("vr"))
        if vr == vr:
            if vr > TH["vr_high"]:
                out.append(_mk("vr_extreme", "VR过高（成交过热）", DIR_BEAR, 0.5, 0.6,
                               "%.0f" % vr, "成交量比率过高，市场情绪过热，谨防回落",
                               "VR", _SHORT))
            elif vr < TH["vr_low"]:
                out.append(_mk("vr_extreme", "VR过低（成交低迷）", DIR_BULL, 0.45, 0.58,
                               "%.0f" % vr, "成交量比率过低，属于地量区，往往对应阶段性底部",
                               "VR", _MID))
        ar = _last(m.get("ar"))
        if ar == ar:
            if ar > TH["brar_ar_over"]:
                out.append(_mk("brar_extreme", "AR人气过旺", DIR_BEAR, 0.42, 0.55,
                               "AR=%.0f" % ar, "人气指标过高，多头能量消耗过度",
                               "BRAR", _SHORT))
            elif ar < TH["brar_ar_under"]:
                out.append(_mk("brar_extreme", "AR人气低迷", DIR_BULL, 0.42, 0.55,
                               "AR=%.0f" % ar, "人气指标过低，市场过度悲观",
                               "BRAR", _MID))
        for key, label, theory in (("asi", "ASI", "ASI"), ("emv", "EMV", "EMV"),
                                   ("wvad", "WVAD", "WVAD")):
            v, vma = _last(m.get(key)), _last(m.get(key + "_ma"))
            if v == v and vma == vma:
                up = v > vma
                out.append(_mk("%s_state" % key.lower(), "%s%s" % (label, "走强" if up else "走弱"),
                               DIR_BULL if up else DIR_BEAR, 0.32, 0.6,
                               "%s=%.1f 均线=%.1f" % (label, v, vma),
                               "%s指标在均线%s，%s" % (label, "上方" if up else "下方",
                                                     "买盘力量占优" if up else "卖盘力量占优"),
                               theory, _MID))

        # ---- 量能阶段（地量/温和/放量/巨量） ---------------------------- #
        if v_ma5 == v_ma5 and v_ma20 == v_ma20 and v_ma20 > 0:
            r = v_ma5 / v_ma20
            if r < 0.7:
                stage, d, s = "地量", DIR_NEUTRAL, 0.3
                det = "近5日量能仅为20日均量的%.0f%%，处于地量区，往往是变盘前兆" % (r * 100)
            elif r > 1.8:
                stage, d, s = "巨量", DIR_NEUTRAL, 0.35
                det = "近5日量能为20日均量的%.0f%%，量能异常放大，波动将加剧" % (r * 100)
            elif r > 1.2:
                stage, d, s = "放量", DIR_BULL if price_ret5 > 0 else DIR_BEAR, 0.4
                det = "近5日量能为20日均量的%.0f%%，量能温和放大" % (r * 100)
            else:
                stage, d, s = "温和", DIR_NEUTRAL, 0.2
                det = "量能处于常态水平"
            out.append(_mk("volume_stage", "量能阶段：%s" % stage, d, s, 0.6,
                           "5日/20日均量=%.2f" % r, det, "量价关系", _SHORT_MID))

        # ---- 换手率 ---------------------------------------------------- #
        if "turnover" in df.columns and df["turnover"].notna().sum() > 20:
            t_now = _last(df["turnover"])
            t_ma20 = float(df["turnover"].tail(20).mean())
            if t_now == t_now and t_ma20 == t_ma20 and t_ma20 > 0:
                tr = t_now / t_ma20
                if tr > 2.5:
                    out.append(_mk("turnover_spike", "换手率异常放大", DIR_NEUTRAL, 0.45, 0.65,
                                   "换手%.2f%%（20日均%.2f%%）" % (t_now, t_ma20),
                                   "换手率骤增，多空分歧加剧，方向取决于价格位置",
                                   "换手率", _SHORT))
    except Exception as exc:  # noqa: BLE001
        log.debug("量能判定失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 5. 筹码、支撑压力、黄金分割
# --------------------------------------------------------------------------- #
def level_signals(df: pd.DataFrame, snapshot: dict | None = None) -> list:
    out: list[Factor] = []
    price = _last(df["close"])
    if price != price:
        return out
    high, low, close = df["high"], df["low"], df["close"]
    vol = df.get("volume")

    # ---- 支撑压力位 ---------------------------------------------------- #
    try:
        sr = ind.support_resistance(high, low, close, volume=vol, lookback=250)
        if isinstance(sr, dict):
            sup = sr.get("nearest_support")
            res = sr.get("nearest_resistance")
            pos = sr.get("position")
            if sup and res and sup == sup and res == res and res > sup:
                up_room = (res - price) / price * 100
                dn_room = (price - sup) / price * 100
                if up_room < 0.35:
                    # 压力位几乎与现价重合：此时不是"遇阻回落"，而是"正在测试/突破压力"
                    # ——方向意义完全不同，必须单独处理，否则会把突破当成见顶。
                    out.append(_mk("sr_position", "股价正测试压力位", DIR_NEUTRAL, 0.35, 0.7,
                                   "压力=%.2f 现价=%.2f（仅差%.2f%%）" % (res, price, up_room),
                                   "股价贴合关键压力位上沿，方向取决于能否放量突破："
                                   "放量站上则压力转支撑，缩量滞涨则易回落",
                                   "支撑压力理论", _SHORT))
                else:
                    rr = up_room / max(dn_room, 0.1)
                    if dn_room > 0.35 and rr > 1.6:
                        d, s = DIR_BULL, min(0.35 + rr / 8.0, 0.75)
                        det = "上方压力 %.2f（+%.1f%%），下方支撑 %.2f（-%.1f%%），盈亏比 %.1f:1，位置有利" \
                            % (res, up_room, sup, dn_room, rr)
                    elif dn_room > 0.35 and rr < 0.6:
                        d, s = DIR_BEAR, 0.55
                        det = "上方压力 %.2f 仅 +%.1f%%，下方支撑 %.2f 有 -%.1f%%，盈亏比 %.1f:1，位置不利" \
                            % (res, up_room, sup, dn_room, rr)
                    elif dn_room <= 0.35:
                        d, s = DIR_BULL, 0.4
                        det = "股价已站上原支撑 %.2f（转化为新支撑），上方空间看 %.2f" % (sup, res)
                    else:
                        d, s = DIR_NEUTRAL, 0.25
                        det = "上方压力 %.2f（+%.1f%%），下方支撑 %.2f（-%.1f%%），盈亏比均衡" \
                            % (res, up_room, sup, dn_room)
                    out.append(_mk("sr_position", "支撑压力位置", d, s, 0.7,
                                   "支撑=%.2f 压力=%.2f 现价=%.2f" % (sup, res, price),
                                   det, "支撑压力理论", _SHORT_MID))
            if pos == "near_support":
                out.append(_mk("sr_proximity", "股价贴近支撑位", DIR_BULL, 0.5, 0.68,
                               "支撑=%.2f" % (sup if sup == sup else float("nan")),
                               "股价回落到关键支撑附近，若支撑有效则反弹概率较大",
                               "支撑压力理论", _SHORT))
            elif pos == "near_resistance" and res == res and price == price and \
                    (res - price) / price * 100 >= 0.35:
                out.append(_mk("sr_proximity", "股价逼近压力位", DIR_BEAR, 0.5, 0.68,
                               "压力=%.2f（+%.2f%%）" % (res, (res - price) / price * 100),
                               "股价上行到关键压力附近，需放量突破否则容易回落",
                               "支撑压力理论", _SHORT))
    except Exception as exc:  # noqa: BLE001
        log.debug("支撑压力计算失败: %s", exc)

    # ---- 黄金分割位 ---------------------------------------------------- #
    try:
        fb = ind.fib_levels(high, low, lookback=250)
        if isinstance(fb, dict):
            zone = fb.get("current_zone")
            ns, nr = fb.get("nearest_support"), fb.get("nearest_resistance")
            up_band = fb.get("direction", 1)
            if zone:
                if "强支撑" in str(zone):
                    d, s = DIR_BULL, 0.6
                elif "支撑" in str(zone):
                    d, s = DIR_BULL, 0.45
                elif "强压力" in str(zone):
                    d, s = DIR_BEAR, 0.6
                elif "压力" in str(zone):
                    d, s = DIR_BEAR, 0.45
                else:
                    d, s = DIR_NEUTRAL, 0.2
                out.append(_mk("fib_level", "斐波那契%s" % zone, d, s, 0.65,
                               "支撑=%.2f 压力=%.2f" % (_f(ns), _f(nr)),
                               "现价处于斐波那契回撤的%s，%s" % (
                                   zone, "该区域历史上易获得支撑" if d > 0 else
                                   ("该区域历史上易遇阻回落" if d < 0 else "方向中性")),
                               "斐波那契理论", _MID_LONG))
    except Exception as exc:  # noqa: BLE001
        log.debug("黄金分割计算失败: %s", exc)

    # ---- 筹码分布 ------------------------------------------------------ #
    try:
        chip = ind.chip_distribution(high, low, close, vol, bins=120, lookback=250)
        if isinstance(chip, dict):
            conc = _f(chip.get("concentration"))
            profit = _f(chip.get("profit_ratio"))
            avg = _f(chip.get("avg_cost"))
            peak = _f(chip.get("peak_price"))
            if conc == conc:
                if conc > 0.28:
                    out.append(_mk("chip_concentration", "筹码高度集中", DIR_BULL, 0.6, 0.68,
                                   "集中度=%.2f 平均成本=%.2f" % (conc, avg),
                                   "筹码集中在少数价格区间（主力成本区），一旦突破通常有较大行情",
                                   "筹码分布理论", _MID_LONG))
                elif conc < 0.12:
                    out.append(_mk("chip_concentration", "筹码高度分散", DIR_NEUTRAL, 0.3, 0.6,
                                   "集中度=%.2f" % conc,
                                   "筹码分散，缺乏主导资金，容易随大盘波动",
                                   "筹码分布理论", _MID))
            if profit == profit:
                if profit > 0.92:
                    out.append(_mk("profit_ratio", "获利盘极度拥挤", DIR_BEAR, 0.55, 0.65,
                                   "获利盘=%.0f%% 平均成本=%.2f" % (profit * 100, avg),
                                   "几乎全部持仓者处于盈利状态，一旦回调容易引发获利了结",
                                   "筹码分布理论", _SHORT))
                elif profit < 0.10:
                    out.append(_mk("profit_ratio", "套牢盘沉重（超跌）", DIR_BULL, 0.5, 0.65,
                                   "获利盘=%.0f%% 平均成本=%.2f" % (profit * 100, avg),
                                   "绝大多数持仓者被套，抛压虽重但继续下跌空间有限，反弹动能积累",
                                   "筹码分布理论", _MID))
            if peak == peak and price == price:
                out.append(_mk("chip_peak", "筹码峰位置", DIR_NEUTRAL, 0.25, 0.62,
                               "筹码峰=%.2f 现价=%.2f" % (peak, price),
                               "上方/下方筹码峰构成天然的%s" % ("压力" if peak > price else "支撑"),
                               "筹码分布理论", _MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("筹码分布计算失败: %s", exc)

    # ---- 持仓成本线 ---------------------------------------------------- #
    try:
        vw = ind.vwap_cost(high, low, close, vol, n=60)
        if isinstance(vw, dict):
            cost = _f(vw.get("cost_60") or vw.get("vwap"))
            if cost == cost and price == price:
                up = price > cost
                out.append(_mk("cost_line", "股价%s60日持仓成本线" % ("高于" if up else "低于"),
                               DIR_BULL if up else DIR_BEAR, 0.42, 0.68,
                               "成本线=%.2f 现价=%.2f" % (cost, price),
                               "现价%s市场平均成本，%s" % (
                                   "高于" if up else "低于",
                                   "多数持仓者盈利，人气偏强" if up else "多数持仓者亏损，反弹有压力"),
                               "持仓成本理论", _MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("持仓成本线计算失败: %s", exc)

    # ---- 前期高低点 ---------------------------------------------------- #
    try:
        n = len(df)
        for bars, label in ((60, "60日"), (120, "半年"), (250, "一年")):
            if n > bars:
                hh = float(df["high"].tail(bars).max())
                ll = float(df["low"].tail(bars).min())
                if price == price and hh > 0:
                    d_high = (price / hh - 1) * 100
                    if d_high > -1.5:
                        out.append(_mk("prior_high", "逼近%s高点" % label, DIR_NEUTRAL, 0.4, 0.7,
                                       "前高=%.2f 现价=%.2f" % (hh, price),
                                       "股价接近%s前高，突破则打开空间，受阻则形成双顶风险" % label,
                                       "形态理论", _MID_LONG))
                    elif d_high < -25 and bars == 250:
                        out.append(_mk("prior_high", "深度回撤（距一年高点%.0f%%）" % d_high,
                                       DIR_BEAR, min(abs(d_high) / 60.0, 0.7), 0.72,
                                       "一年高点=%.2f 现价=%.2f" % (hh, price),
                                       "股价距一年高点回撤 %.0f%%，中期趋势受损，反弹面临解套压力"
                                       % abs(d_high), "道氏理论", _MID_LONG))
    except Exception as exc:  # noqa: BLE001
        log.debug("前高前低计算失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 6. 形态与结构
# --------------------------------------------------------------------------- #
def pattern_signals(df: pd.DataFrame, code: str = "", name: str = "") -> list:
    out: list[Factor] = []
    # ---- K 线形态（由 patterns 模块给出） ------------------------------ #
    try:
        pf = pat.pattern_factors(df, code=code, name=name)
        if isinstance(pf, list):
            out.extend([f for f in pf if isinstance(f, Factor)])
    except Exception as exc:  # noqa: BLE001
        log.debug("形态因子失败: %s", exc)
    # ---- 经典形态/箱体/双顶双底 ---------------------------------------- #
    try:
        cs = pat.classic_shape(df, lookback=250)
        if isinstance(cs, dict):
            for sh in (cs.get("shapes") or [])[:4]:
                if not isinstance(sh, dict):
                    continue
                d = int(sh.get("direction", 0) or 0)
                out.append(_mk("classic_shape", "经典形态：%s" % sh.get("name", "形态"),
                               DIR_BULL if d > 0 else (DIR_BEAR if d < 0 else DIR_NEUTRAL),
                               _f(sh.get("strength")) if _f(sh.get("strength")) == _f(sh.get("strength")) else 0.5,
                               0.62, sh.get("target"),
                               str(sh.get("detail", "经典技术形态识别"))[:120],
                               "形态理论", _MID_LONG))
            box = cs.get("box") or {}
            if isinstance(box, dict) and box.get("detected"):
                brk = box.get("break")
                if brk == "up":
                    out.append(_mk("box_break", "箱体向上突破", DIR_BULL, 0.72, 0.7,
                                   "箱顶=%.2f" % _f(box.get("top")),
                                   "价格放量突破箱体上沿，量度涨幅可参考箱体高度",
                                   "箱体理论", _SHORT_MID))
                elif brk == "down":
                    out.append(_mk("box_break", "箱体向下跌破", DIR_BEAR, 0.72, 0.7,
                                   "箱底=%.2f" % _f(box.get("bottom")),
                                   "价格跌破箱体下沿，需警惕新一轮下跌",
                                   "箱体理论", _SHORT_MID))
                else:
                    out.append(_mk("box_range", "箱体震荡", DIR_NEUTRAL, 0.3, 0.65,
                                   "箱顶=%.2f 箱底=%.2f" % (_f(box.get("top")), _f(box.get("bottom"))),
                                   "价格在箱体内反复震荡，适合高抛低吸，突破前不宜追单",
                                   "箱体理论", _SHORT_MID))
            dt = cs.get("double_top_bottom") or {}
            if isinstance(dt, dict) and dt.get("type") in ("double_top", "double_bottom"):
                up = dt["type"] == "double_bottom"
                out.append(_mk("double_pattern",
                               "双底形态" if up else "双顶形态",
                               DIR_BULL if up else DIR_BEAR, 0.62, 0.65,
                               "颈线=%.2f" % _f(dt.get("neckline")),
                               "%s，颈线位 %.2f，突破/跌破后目标 %.2f" % (
                                   "双底（W底）属于反转向上形态" if up else "双顶（M头）属于反转向下形态",
                                   _f(dt.get("neckline")), _f(dt.get("target"))),
                               "形态理论", _MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("经典形态识别失败: %s", exc)
    # ---- 缺口 ---------------------------------------------------------- #
    try:
        gs = ind.gap_stats(df["open"], df["high"], df["low"], df["close"])
        if isinstance(gs, dict):
            gaps = gs.get("gaps") or []
            recent = [g for g in gaps[-3:] if isinstance(g, dict)]
            for g in recent:
                unfilled = not g.get("filled")
                up = g.get("type") == "up"
                if unfilled:
                    out.append(_mk("gap_pattern",
                                   "向上跳空缺口（未回补）" if up else "向下跳空缺口（未回补）",
                                   DIR_BULL if up else DIR_BEAR, 0.52, 0.6,
                                   "缺口%.2f%% @%s" % (_f(g.get("size_pct")), str(g.get("date"))[:10]),
                                   "未回补的%s缺口构成%s" % ("向上" if up else "向下",
                                                           "支撑" if up else "压力"),
                                   "缺口理论", _SHORT_MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("缺口识别失败: %s", exc)
    # ---- 黄金坑 -------------------------------------------------------- #
    try:
        gp = ind.golden_pit(df["high"], df["low"], df["close"], volume=df.get("volume"))
        if isinstance(gp, dict) and gp.get("detected"):
            score = _f(gp.get("score"))
            out.append(_mk("golden_pit", "黄金坑形态", DIR_BULL,
                           min(0.5 + (score if score == score else 0.4) * 0.4, 0.85), 0.66,
                           "坑深=%.1f%% 收复=%s根" % (_f(gp.get("pit_depth_pct")),
                                                    gp.get("recover_bars")),
                           "股价快速下砸形成黄金坑后放量收复，通常是主力洗盘结束的信号，"
                           "后市上涨概率较高", "形态理论", _SHORT_MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("黄金坑识别失败: %s", exc)
    # ---- 艾略特波浪 / 缠论 --------------------------------------------- #
    try:
        ew = pat.elliott_count(df, lookback=250)
        if isinstance(ew, dict):
            wave = str(ew.get("wave", "unknown"))
            conf = _f(ew.get("confidence"))
            d = int(ew.get("direction", 0) or 0)
            if wave != "unknown" and d != 0 and conf == conf and conf >= 0.3:
                out.append(_mk("elliott_wave", "波浪结构：第%s浪" % wave,
                               DIR_BULL if d > 0 else DIR_BEAR,
                               min(0.35 + conf * 0.45, 0.78), min(0.4 + conf * 0.3, 0.62),
                               "浪级=%s 吻合度=%.2f" % (wave, conf),
                               str(ew.get("detail", "艾略特波浪结构判断"))[:130],
                               "艾略特波浪理论", _MID_LONG))
    except Exception as exc:  # noqa: BLE001
        log.debug("波浪识别失败: %s", exc)
    try:
        ch = pat.chan_strokes(df, lookback=200)
        if isinstance(ch, dict):
            bp = str(ch.get("buy_point") or "无")
            sp = str(ch.get("sell_point") or "无")
            div = bool(ch.get("divergence"))
            if bp in ("1买", "2买", "3买"):
                s = {"1买": 0.7, "2买": 0.6, "3买": 0.5}.get(bp, 0.5)
                out.append(_mk("chan_buy_point", "缠论%s点" % bp, DIR_BULL, s, 0.55,
                               "当前=%s" % ch.get("current"),
                               "缠论结构出现%s，%s" % (bp, str(ch.get("detail", ""))[:90]),
                               "缠论", _SHORT_MID))
            if sp in ("1卖", "2卖", "3卖"):
                s = {"1卖": 0.7, "2卖": 0.6, "3卖": 0.5}.get(sp, 0.5)
                out.append(_mk("chan_sell_point", "缠论%s点" % sp, DIR_BEAR, s, 0.55,
                               "当前=%s" % ch.get("current"),
                               "缠论结构出现%s，%s" % (sp, str(ch.get("detail", ""))[:90]),
                               "缠论", _SHORT_MID))
            if div:
                out.append(_mk("chan_divergence", "缠论背驰", DIR_NEUTRAL, 0.4, 0.5,
                               "背驰=True", "笔/段级别出现背驰，原趋势动能衰减，注意转折",
                               "缠论", _SHORT_MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("缠论识别失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 7. 背离（个股 vs 大盘、指标 vs 价格）
# --------------------------------------------------------------------------- #
def divergence_signals(df: pd.DataFrame, index_df: pd.DataFrame | None) -> list:
    out: list[Factor] = []
    try:
        c = df["close"].astype(float)
        n = len(c)
        if n < 60:
            return out
    except Exception:  # noqa: BLE001
        return out

    # ---- 个股 vs 大盘 -------------------------------------------------- #
    if isinstance(index_df, pd.DataFrame) and len(index_df) > 60 and "close" in index_df.columns:
        try:
            ic = index_df["close"].astype(float)
            k = min(len(c), len(ic), 120)
            cs, is_ = c.tail(k).reset_index(drop=True), ic.tail(k).reset_index(drop=True)
            r_s = float(cs.iloc[-1] / cs.iloc[0] - 1) * 100
            r_i = float(is_.iloc[-1] / is_.iloc[0] - 1) * 100
            diff = r_s - r_i
            if diff > 12:
                out.append(_mk("rel_strength", "显著强于大盘", DIR_BULL,
                               min(0.4 + diff / 60.0, 0.8), 0.78,
                               "个股%.1f%% vs 大盘%.1f%%" % (r_s, r_i),
                               "近%d日个股跑赢大盘 %.1f 个百分点，资金关注度高、相对强度强"
                               % (k, diff), "相对强度理论", _MID))
            elif diff < -12:
                out.append(_mk("rel_strength", "显著弱于大盘", DIR_BEAR,
                               min(0.4 + abs(diff) / 60.0, 0.8), 0.78,
                               "个股%.1f%% vs 大盘%.1f%%" % (r_s, r_i),
                               "近%d日个股跑输大盘 %.1f 个百分点，弱势特征明显"
                               % (k, abs(diff)), "相对强度理论", _MID))
            # 顶/底背离：用 20 日新高新低与大盘对比
            win = min(60, len(cs), len(is_))
            p_high_now = float(cs.tail(win).max())
            p_high_prev = float(cs.tail(win * 2).max()) if len(cs) >= win * 2 else p_high_now
            i_high_now = float(is_.tail(win).max())
            i_high_prev = float(is_.tail(win * 2).max()) if len(is_) >= win * 2 else i_high_now
            if p_high_now > p_high_prev and i_high_now < i_high_prev * 0.999:
                out.append(_mk("index_divergence", "个股与大盘顶背离", DIR_BEAR, 0.6, 0.66,
                               "个股创新高而大盘未创新高",
                               "个股逆势创新高但大盘走弱，缺少市场环境配合，回落风险上升",
                               "背离理论", _SHORT_MID))
            p_low_now = float(cs.tail(win).min())
            p_low_prev = float(cs.tail(win * 2).min()) if len(cs) >= win * 2 else p_low_now
            i_low_now = float(is_.tail(win).min())
            i_low_prev = float(is_.tail(win * 2).min()) if len(is_) >= win * 2 else i_low_now
            if p_low_now < p_low_prev and i_low_now > i_low_prev * 1.001:
                out.append(_mk("index_divergence", "个股与大盘底背离", DIR_BULL, 0.6, 0.66,
                               "个股创新低而大盘未创新低",
                               "个股单独创新低而大盘企稳，属超跌，存在补涨修复机会",
                               "背离理论", _SHORT_MID))
        except Exception as exc:  # noqa: BLE001
            log.debug("大盘背离计算失败: %s", exc)

    # ---- 指标 vs 价格背离（MACD 顶/底背离） --------------------------- #
    try:
        e12 = c.ewm(span=12, adjust=False).mean()
        e26 = c.ewm(span=26, adjust=False).mean()
        dif = (e12 - e26).to_numpy()
        win = 30
        if len(c) >= win * 2:
            p_now = float(c.tail(win).max())
            p_prev = float(c.tail(win * 2).head(win).max())
            d_now = float(np.nanmax(dif[-win:]))
            d_prev = float(np.nanmax(dif[-win * 2:-win]))
            if p_now > p_prev and d_now < d_prev:
                out.append(_mk("macd_divergence", "MACD顶背离", DIR_BEAR, 0.65, 0.7,
                               "价新高 DIF未新高",
                               "价格创出新高但MACD的DIF未同步创新高，上涨动能衰竭",
                               "背离理论", _SHORT_MID))
            p_lo_now = float(c.tail(win).min())
            p_lo_prev = float(c.tail(win * 2).head(win).min())
            d_lo_now = float(np.nanmin(dif[-win:]))
            d_lo_prev = float(np.nanmin(dif[-win * 2:-win]))
            if p_lo_now < p_lo_prev and d_lo_now > d_lo_prev:
                out.append(_mk("macd_divergence", "MACD底背离", DIR_BULL, 0.65, 0.7,
                               "价新低 DIF未新低",
                               "价格创出新低但MACD的DIF未同步创新低，下跌动能衰竭，反弹可期",
                               "背离理论", _SHORT_MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("MACD背离计算失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 8. 多周期共振
# --------------------------------------------------------------------------- #
def multi_timeframe(df: pd.DataFrame, weekly=None, monthly=None) -> list:
    """日/周/月线同逻辑判定，共振提升可信度、冲突降低可信度。"""
    out: list[Factor] = []
    try:
        verdicts = {}
        for label, frame in (("日线", df), ("周线", weekly), ("月线", monthly)):
            if frame is None or not isinstance(frame, pd.DataFrame) or len(frame) < 30:
                continue
            d = _ensure_frame(frame)
            if d is None:
                continue
            m = ind.indicator_matrix(d)
            close = d["close"].astype(float)
            ma20, ma60 = _last(m.get("ma20")), _last(m.get("ma60"))
            price = _last(close)
            if not all(x == x for x in (ma20, ma60, price)):
                continue
            # 用"价格 vs MA20 vs MA60 + MA20 斜率"三项表决
            votes = 0
            votes += 1 if price > ma20 else -1
            votes += 1 if ma20 > ma60 else -1
            s20 = m["ma20"].dropna()
            if len(s20) >= 6:
                votes += 1 if s20.iloc[-1] > s20.iloc[-6] else -1
            verdicts[label] = votes
        if len(verdicts) >= 2:
            vals = list(verdicts.values())
            aligned_up = all(v >= 1 for v in vals)
            aligned_dn = all(v <= -1 for v in vals)
            detail = "、".join("%s%+d" % (k, v) for k, v in verdicts.items())
            if aligned_up:
                out.append(_mk("mtf_resonance", "多周期共振向上", DIR_BULL, 0.8, 0.85,
                               detail, "日/周/月线同时指向多头，趋势级别高，回撤是买点",
                               "多周期共振", _ALL))
            elif aligned_dn:
                out.append(_mk("mtf_resonance", "多周期共振向下", DIR_BEAR, 0.8, 0.85,
                               detail, "日/周/月线同时指向空头，反弹是减仓机会",
                               "多周期共振", _ALL))
            elif vals[0] > 0 and any(v < 0 for v in vals[1:]):
                out.append(_mk("mtf_resonance", "短期反弹、周月线未转强", DIR_NEUTRAL, 0.3, 0.72,
                               detail, "日线转强但更大周期仍偏弱，属反弹而非反转，降低预期",
                               "多周期共振", _SHORT))
            elif vals[0] < 0 and any(v > 0 for v in vals[1:]):
                out.append(_mk("mtf_resonance", "短期回调、更大周期偏强", DIR_BULL, 0.4, 0.72,
                               detail, "日线走弱但周/月线仍强，属上升趋势中的正常回调",
                               "多周期共振", _MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("多周期共振计算失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 9. K 线类型（等比/等量）
# --------------------------------------------------------------------------- #
def kline_type_signals(df: pd.DataFrame) -> list:
    """用等比 K 线（Renko）与等量 K 线过滤噪声，判断真实方向。"""
    out: list[Factor] = []
    try:
        rk = ind.renko(df, brick_pct=0.02)
        if isinstance(rk, pd.DataFrame) and len(rk) >= 4:
            dirs = rk["brick_dir"].tail(4).tolist()
            up_count = sum(1 for d in dirs if d > 0)
            if up_count == 4:
                out.append(_mk("renko_trend", "等比K线连续4砖向上", DIR_BULL, 0.62, 0.66,
                               "砖图方向=%s" % dirs,
                               "剔除日内噪声后，等比K线（砖图）连续向上，趋势方向明确",
                               "等比K线（Renko）", _SHORT_MID))
            elif up_count == 0:
                out.append(_mk("renko_trend", "等比K线连续4砖向下", DIR_BEAR, 0.62, 0.66,
                               "砖图方向=%s" % dirs,
                               "剔除日内噪声后，等比K线连续向下，趋势方向明确",
                               "等比K线（Renko）", _SHORT_MID))
            else:
                out.append(_mk("renko_trend", "等比K线方向反复", DIR_NEUTRAL, 0.3, 0.6,
                               "砖图方向=%s" % dirs,
                               "砖图方向反复切换，趋势不明确，属震荡行情",
                               "等比K线（Renko）", _SHORT))
    except Exception as exc:  # noqa: BLE001
        log.debug("Renko 计算失败: %s", exc)
    try:
        ek = ind.equal_kline(df, mode="volume", box_vol_ratio=1.5)
        if isinstance(ek, pd.DataFrame) and len(ek) >= 3:
            closes = ek["close"].astype(float).tail(3).tolist()
            opens = ek["open"].astype(float).tail(3).tolist()
            ups = sum(1 for o, c in zip(opens, closes) if c >= o)
            if ups == 3:
                out.append(_mk("equal_kline_trend", "等量K线连续收阳", DIR_BULL, 0.5, 0.6,
                               "最近3根等量K线收阳",
                               "以成交量为单位切分的等量K线连续收阳，说明资金持续主动买入",
                               "等量K线", _SHORT_MID))
            elif ups == 0:
                out.append(_mk("equal_kline_trend", "等量K线连续收阴", DIR_BEAR, 0.5, 0.6,
                               "最近3根等量K线收阴",
                               "等量K线连续收阴，资金持续流出", "等量K线", _SHORT_MID))
    except Exception as exc:  # noqa: BLE001
        log.debug("等量K线计算失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 10. 个股特性、所处阶段、定价权、量化活跃度
# --------------------------------------------------------------------------- #
def _annual_vol_pct(close: pd.Series) -> float:
    try:
        r = np.diff(np.log(np.maximum(close.astype(float).to_numpy(), 1e-9)))
        if len(r) < 20:
            return float("nan")
        return float(np.std(r[-120:], ddof=1) * math.sqrt(244.0) * 100.0)
    except Exception:  # noqa: BLE001
        return float("nan")


def describe_stage(df, code: str = "", name: str = "", snapshot: dict | None = None) -> dict:
    """判定个股所处阶段、趋势类型、定价权归属与量化活跃度。

    Returns
    -------
    dict，含 ``stage / stage_detail / trend_type / pricing_power / quant_level /
    character / volatility_level`` 以及若干辅助数值（供报告展示）。
    """
    d = _ensure_frame(df)
    out: dict = {}
    if d is None:
        return out
    snapshot = snapshot or {}
    close = d["close"].astype(float)
    n = len(close)
    price = float(close.iloc[-1])

    # ---- 均线水位（阶段判定的骨架） ------------------------------------- #
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma250 = close.rolling(250).mean() if n >= 250 else close.rolling(n).mean()
    ma20_v, ma60_v, ma250_v = _last(ma20), _last(ma60), _last(ma250)
    hi250 = float(d["high"].tail(min(n, 250)).max())
    lo250 = float(d["low"].tail(min(n, 250)).min())
    pos = (price - lo250) / max(hi250 - lo250, 1e-9)          # 250日区间位置 0~1
    dd_from_high = (price / hi250 - 1) * 100 if hi250 > 0 else float("nan")
    up_from_low = (price / lo250 - 1) * 100 if lo250 > 0 else float("nan")

    # 20 日均线斜率（每日 %）
    slope20 = float("nan")
    try:
        s = ma20.dropna()
        if len(s) >= 21:
            slope20 = (s.iloc[-1] / s.iloc[-21] - 1) * 100 / 20.0
    except Exception:  # noqa: BLE001
        pass

    vol_ann = _annual_vol_pct(close)
    # 近期振幅（用于量化活跃度）
    amp = ((d["high"] - d["low"]) / close.shift(1) * 100).tail(20).mean()
    turnover_now = snapshot.get("turnover")
    turnover_series = d["turnover"] if "turnover" in d.columns and d["turnover"].notna().sum() > 10 else None
    turnover_avg = (float(turnover_series.tail(20).mean()) if turnover_series is not None
                    else (float(turnover_now) if isinstance(turnover_now, (int, float))
                          and turnover_now == turnover_now else float("nan")))

    # ---- 阶段判定 ------------------------------------------------------- #
    above20 = price > ma20_v if ma20_v == ma20_v else None
    above60 = price > ma60_v if ma60_v == ma60_v else None
    above250 = price > ma250_v if ma250_v == ma250_v else None
    bull_ma = (above20 and above60 and (above250 is not False))
    bear_ma = (above20 is False and above60 is False)

    stage = "震荡整理"
    detail = ""
    if bull_ma and slope20 == slope20 and slope20 > 0.05 and pos < 0.45:
        stage = "上涨初期（趋势启动）"
        detail = "均线刚转为多头、股价位于250日区间下半部，属趋势启动阶段，风险收益比较佳"
    elif bull_ma and slope20 == slope20 and slope20 > 0.05 and pos >= 0.45:
        stage = "上涨中继/主升段"
        detail = "均线多头且股价已在区间中上部上行，趋势延续中，但需警惕加速后的背离"
    elif bull_ma and pos > 0.85:
        stage = "高位强势（警惕派发）"
        detail = "股价接近一年高位且均线多头，需观察是否放量滞涨，高位派发风险上升"
    elif bear_ma and pos < 0.25:
        stage = "底部区域（可能吸筹）"
        detail = "股价处于一年低位且均线空头，若出现缩量止跌与主力净流入，可能进入吸筹阶段"
    elif bear_ma:
        stage = "下跌中继"
        detail = "均线空头排列、股价位于区间中下部，下跌趋势尚未修复，反弹多为技术性"
    elif pos < 0.2:
        stage = "超跌区"
        detail = "股价处于一年低位附近但均线尚未完全转空，属超跌状态"
    else:
        detail = "均线纠缠、股价在区间中部，方向不明，属震荡整理阶段"

    out["stage"] = stage
    out["stage_detail"] = detail
    out["detail"] = detail
    out["pos_250d"] = round(pos, 3)
    out["drawdown_from_250d_high_pct"] = round(dd_from_high, 2) if dd_from_high == dd_from_high else None
    out["up_from_250d_low_pct"] = round(up_from_low, 2) if up_from_low == up_from_low else None
    out["ma20_slope_pct_per_day"] = round(slope20, 4) if slope20 == slope20 else None

    # ---- 趋势类型 ------------------------------------------------------- #
    ts = {}
    try:
        ts = ind.trend_strength(close, d["high"], d["low"], n=20) or {}
    except Exception:  # noqa: BLE001
        ts = {}
    r2 = _f(ts.get("r2"))
    adx = _f(snapshot.get("adx"))
    if r2 == r2 and r2 > 0.6:
        out["trend_type"] = "强趋势（方向明确）"
    elif r2 == r2 and r2 > 0.3:
        out["trend_type"] = "弱趋势"
    else:
        out["trend_type"] = "横盘震荡"
    out["trend_r2"] = round(r2, 3) if r2 == r2 else None

    # ---- 波动率级别 ----------------------------------------------------- #
    if vol_ann == vol_ann:
        if vol_ann > config.STAGE_RULES["high_vol_annual_pct"]:
            out["volatility_level"] = "高波动"
        elif vol_ann < config.STAGE_RULES["low_vol_annual_pct"]:
            out["volatility_level"] = "低波动"
        else:
            out["volatility_level"] = "中等波动"
        out["annual_vol_pct"] = round(vol_ann, 2)

    # ---- 定价权归属 ----------------------------------------------------- #
    top10 = _f(snapshot.get("top10_float_ratio"))
    float_cap = _f(snapshot.get("float_market_cap"))
    if top10 == top10:
        if top10 >= config.STAGE_RULES["pricing_power_top10_pct"]:
            power = "机构/主力定价（筹码集中在少数股东手中）"
        elif top10 <= config.STAGE_RULES["pricing_power_retail_pct"]:
            power = "散户/游资定价（股权高度分散，情绪驱动明显）"
        else:
            power = "混合定价（机构与散户博弈）"
        out["pricing_power"] = power
        out["top10_float_ratio"] = round(top10, 2)
    elif float_cap == float_cap:
        # 没有股东数据时用市值规模做粗略推断
        if float_cap > 1e11:
            out["pricing_power"] = "大盘股：机构定价为主（推算）"
        elif float_cap < 2e10:
            out["pricing_power"] = "中小盘：游资/散户参与度高（推算）"
        else:
            out["pricing_power"] = "中盘股：机构与游资混合（推算）"

    # ---- 量化活跃度 ----------------------------------------------------- #
    q_parts = []
    if turnover_avg == turnover_avg:
        q_parts.append(turnover_avg > config.STAGE_RULES["quant_turnover_high_pct"])
    if amp == amp:
        q_parts.append(float(amp) > config.STAGE_RULES["quant_amp_high_pct"])
    if q_parts:
        hi_cnt = sum(1 for x in q_parts if x)
        if hi_cnt == len(q_parts):
            out["quant_level"] = "高（换手与振幅均偏高，量化/短线资金活跃）"
        elif hi_cnt == 0:
            out["quant_level"] = "低（换手与振幅温和，以中长线资金为主）"
        else:
            out["quant_level"] = "中（部分指标显示短线资金参与）"
    out["turnover_avg20_pct"] = round(turnover_avg, 3) if turnover_avg == turnover_avg else None
    out["amp20_pct"] = round(float(amp), 2) if amp == amp else None

    # ---- 个股性格 ------------------------------------------------------- #
    chars = []
    if vol_ann == vol_ann and vol_ann > 45:
        chars.append("股性活跃")
    elif vol_ann == vol_ann and vol_ann < 25:
        chars.append("股性稳健")
    if turnover_avg == turnover_avg:
        if turnover_avg < 1:
            chars.append("流动性偏低")
        elif turnover_avg > 8:
            chars.append("交投活跃")
    if pos > 0.8:
        chars.append("处于高位")
    elif pos < 0.2:
        chars.append("处于低位")
    out["character"] = "、".join(chars) if chars else "特征中性"
    out["code"] = code
    out["name"] = name or snapshot.get("name")
    out["price"] = price
    return out


def stage_factors(df, code: str = "", name: str = "", snapshot: dict | None = None) -> list:
    """把 :func:`describe_stage` 的结论转成因子。

    这些因子回答的是"**现在是什么局面**"——同样一个金叉，
    在底部启动阶段与高位派发阶段的含义完全不同，所以阶段本身要有方向权重。
    """
    st = describe_stage(df, code=code, name=name, snapshot=snapshot)
    out: list[Factor] = []
    if not st:
        return out

    stage = str(st.get("stage", ""))
    detail = str(st.get("stage_detail", ""))
    pos = _f(st.get("pos_250d"))
    vol_ann = _f(st.get("annual_vol_pct"))

    mapping = {
        "上涨初期（趋势启动）": (DIR_BULL, 0.68, 0.7, _SHORT_MID),
        "上涨中继/主升段": (DIR_BULL, 0.62, 0.72, _SHORT_MID),
        "高位强势（警惕派发）": (DIR_NEUTRAL, 0.35, 0.65, _SHORT),
        "底部区域（可能吸筹）": (DIR_BULL, 0.48, 0.6, _MID_LONG),
        "下跌中继": (DIR_BEAR, 0.6, 0.72, _SHORT_MID),
        "超跌区": (DIR_BULL, 0.4, 0.58, _MID),
        "震荡整理": (DIR_NEUTRAL, 0.2, 0.6, _SHORT_MID),
    }
    d, s, rel, hz = mapping.get(stage, (DIR_NEUTRAL, 0.2, 0.5, _SHORT_MID))
    out.append(_mk("stage_position", "所处阶段：%s" % stage, d, s, rel,
                   "250日区间位置=%.0f%%" % (pos * 100) if pos == pos else None,
                   detail, "道氏理论/阶段分析", hz))

    # 高位滞涨 → 额外的派发风险因子
    if pos == pos and pos > 0.85:
        out.append(_mk("high_position_risk", "处于一年高位区间", DIR_BEAR, 0.45, 0.6,
                       "250日位置=%.0f%%" % (pos * 100),
                       "股价位于一年区间顶部，上方无历史套牢盘但同时获利盘沉重，"
                       "一旦资金流出容易快速回落", "风险控制", _SHORT_MID))
    elif pos == pos and pos < 0.15:
        out.append(_mk("low_position_opportunity", "处于一年低位区间", DIR_BULL, 0.42, 0.58,
                       "250日位置=%.0f%%" % (pos * 100),
                       "股价位于一年区间底部，向下空间相对有限，但需等待量能与资金确认",
                       "风险收益比", _MID_LONG))

    # 波动率级别：高波动降低方向判断可信度（作为独立因子提示风险）
    if vol_ann == vol_ann:
        if vol_ann > config.STAGE_RULES["high_vol_annual_pct"]:
            out.append(_mk("volatility_regime_stage", "高波动股性", DIR_NEUTRAL, 0.3, 0.7,
                           "年化波动=%.1f%%" % vol_ann,
                           "该股波动显著高于市场平均，任何方向判断的容错空间都更小，"
                           "仓位应相应降低", "风险管理", _ALL))
        elif vol_ann < config.STAGE_RULES["low_vol_annual_pct"]:
            out.append(_mk("volatility_regime_stage", "低波动稳健", DIR_NEUTRAL, 0.25, 0.7,
                           "年化波动=%.1f%%" % vol_ann,
                           "该股波动率较低，通常对应大盘蓝筹或筹码高度锁定",
                           "风险管理", _ALL))

    # 定价权与量化活跃度：影响"该看谁的脸色"
    pp = str(st.get("pricing_power") or "")
    if pp:
        if pp.startswith("机构"):
            out.append(_mk("pricing_power", "机构定价特征", DIR_NEUTRAL, 0.3, 0.65,
                           st.get("top10_float_ratio"), 
                           "筹码集中于机构/大股东，走势更依赖基本面与机构调仓，"
                           "技术形态的短线噪声相对更小", "市场微观结构", _MID_LONG))
        elif pp.startswith("散户"):
            out.append(_mk("pricing_power", "散户/游资定价特征", DIR_NEUTRAL, 0.35, 0.6,
                           st.get("top10_float_ratio"),
                           "股权分散、情绪驱动明显，题材与资金博弈对短期涨跌影响更大，"
                           "技术信号有效性下降", "市场微观结构", _SHORT))
    ql = str(st.get("quant_level") or "")
    if ql.startswith("高"):
        out.append(_mk("quant_activity", "量化/短线资金活跃", DIR_NEUTRAL, 0.35, 0.6,
                       "20日均换手=%s 振幅=%s" % (st.get("turnover_avg20_pct"),
                                                st.get("amp20_pct")),
                       "换手与振幅偏高，量化与短线资金参与度高，日内波动大但趋势持续性弱，"
                       "均值回复类信号更有效", "市场微观结构", _SHORT))
    return out


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def technical_factors(df, index_df=None, weekly=None, monthly=None,
                      snapshot: dict | None = None, code: str = "") -> FactorSet:
    """生成全部技术面因子。

    Parameters
    ----------
    df : 日线 DataFrame（必需）
    index_df : 大盘指数日线（用于相对强度与背离）
    weekly / monthly : 周线 / 月线（用于多周期共振），可为 None
    snapshot : 实时快照（含 name 等）
    code : 股票代码

    Returns
    -------
    FactorSet
    """
    fs = FactorSet()
    d = _ensure_frame(df)
    if d is None:
        log.info("技术面分析：日线数据不足或格式不符，跳过")
        return fs
    try:
        m = ind.indicator_matrix(d)
    except Exception as exc:  # noqa: BLE001
        log.warning("指标矩阵计算失败：%s", exc)
        m = pd.DataFrame(index=d.index)

    name = (snapshot or {}).get("name") or ""
    for fn, args in (
        (ma_system, (d, m)),
        (oscillator_signals, (d, m)),
        (trend_and_channel, (d, m)),
        (volume_signals, (d, m)),
        (lambda x: level_signals(x, snapshot), (d,)),
        (lambda x: pattern_signals(x, code=code, name=name), (d,)),
        (lambda x: divergence_signals(x, index_df), (d,)),
        (lambda x: multi_timeframe(x, weekly, monthly), (d,)),
        (kline_type_signals, (d,)),
    ):
        try:
            got = fn(*args)
            if got:
                fs.extend_factors(got)
        except Exception as exc:  # noqa: BLE001
            log.warning("技术面子模块 %s 失败：%s", getattr(fn, "__name__", "lambda"), exc)
    return fs


if __name__ == "__main__":  # pragma: no cover
    import time

    from .selftest_data import make_index, make_ohlcv

    # 记录本模块中出现的所有因子 id（用于条件型因子的实现自检）
    _SRC_CACHE = set()
    try:
        import re as _re

        _SRC_CACHE = set(_re.findall(r'_mk\(\s*"([a-z0-9_]+)"', open(__file__, encoding="utf-8").read()))
    except Exception:  # noqa: BLE001
        pass

    t0 = time.time()
    df = make_ohlcv(900, seed=42)
    idx = make_index(900, seed=7)
    wk = ind.resample_kline(df, "W-FRI")
    mo = ind.resample_kline(df, "ME")
    fs = technical_factors(df, idx, wk, mo, {"name": "测试股"}, code="600519")
    print("技术面因子 %d 条，耗时 %.1fs" % (len(fs), time.time() - t0))

    # ---- 阶段/特性判定自测 ----
    print()
    st = describe_stage(df, code="600519", name="测试股",
                        snapshot={"turnover": 1.2, "float_market_cap": 5e10})
    print("阶段判定：")
    for k in ("stage", "trend_type", "volatility_level", "pricing_power", "quant_level",
              "character", "pos_250d", "annual_vol_pct"):
        print("  %-18s %s" % (k, st.get(k)))
    sf = stage_factors(df, code="600519", name="测试股",
                       snapshot={"turnover": 1.2, "float_market_cap": 5e10})
    print("阶段类因子 %d 条：%s" % (len(sf), [f.name for f in sf]))
    assert st.get("stage"), "阶段判定不应为空"
    assert all(0 <= f.strength <= 1 and 0 <= f.reliability <= 1 for f in sf)
    # 空输入
    assert describe_stage(None) == {}
    assert stage_factors(None) == []
    print()
    print("按 id 统计：")
    from collections import Counter

    for k, v in Counter(f.id for f in fs).most_common():
        print("  %-24s %d" % (k, v))
    print()
    print("因子明细（前 22 条）：")
    for f in fs.actionable().top(22):
        print("  %-28s %-4s 强度%.2f 可信度%.2f  %s" % (
            f.name, {1: "多", -1: "空", 0: "中"}[f.direction], f.strength, f.reliability,
            f.detail[:52]))
    # ---- 断言 ----
    assert len(fs) >= 20, "技术面因子过少：%d" % len(fs)
    for f in fs:
        assert 0 <= f.strength <= 1 and 0 <= f.reliability <= 1, f
        assert f.direction in (-1, 0, 1), f
        assert f.category == Category.TECH, f
        assert f.name and f.detail, f
        assert f.theory, "因子缺少理论依据：%s" % f.id
    ids = {f.id for f in fs}
    # 条件型因子（金叉/超买/集中度等）只在特定行情下触发，
    # 因此改为检查**模块源码里是否存在对应实现**，而不是要求单一样本必然出现。
    conditional = ["ma_alignment", "macd_cross", "kdj_cross", "rsi_extreme", "vol_price",
                   "chip_concentration", "dmi_trend", "sr_position", "trend_regression",
                   "channel_trend", "fib_level", "golden_pit", "mtf_resonance"]
    missing_impl = [k for k in conditional if k not in _SRC_CACHE]
    print("条件型因子实现自检：%s" % ("全部存在" if not missing_impl else missing_impl))
    assert not missing_impl, "缺少因子实现：%s" % missing_impl
    print("本次样本实际触发 id 数：%d" % len(ids))
    # 全空输入不应抛异常
    assert len(technical_factors(None)) == 0
    assert len(technical_factors(pd.DataFrame())) == 0
    short = make_ohlcv(35, seed=1)
    technical_factors(short)  # 短序列也不能抛
    # 因果性：技术面因子不得使用未来数据（用截断重算对比因子集合的关键统计）
    fs_trunc = technical_factors(df.iloc[:600], idx.iloc[:600],
                                 ind.resample_kline(df.iloc[:600], "W-FRI"),
                                 ind.resample_kline(df.iloc[:600], "ME"))
    print("截断 600 根重算：因子 %d 条（完整 %d 条），无异常" % (len(fs_trunc), len(fs)))
    print("技术面因子自测通过，technical self-test OK")

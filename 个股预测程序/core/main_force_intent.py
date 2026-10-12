# -*- coding: utf-8 -*-
"""
主力操盘意图分析 —— 把"资金动作 + 价格结果 + 量能结构"翻译成主力的意图。

为什么需要它
------------
单看"主力净流入 +1 亿"没有意义：同样是净流入，**股价温和上涨**说明主力在
悄悄吃货，**股价大涨**可能是拉高出货的前奏，**股价下跌**反而是洗盘。
所以必须把三者放在一起看：

    意图 = f(主力资金方向, 价格表现, 量能变化, 股价位置, 筹码结构, 盘口特征)

本模块识别的意图（每种都给强度、可信度与中文证据）
--------------------------------------------------
================  ==========================================================
意图              典型特征
================  ==========================================================
**建仓/吃货**     资金持续净流入 + 价格温和（不涨或微跌）+ 换手温和放大
                  或 缩量横盘但资金偷偷流入（低吸）
**拉升**          资金大幅净流入 + 价格突破 + 放量 + 位于中低位
**洗盘**          价格快速下探但资金未大幅流出 + 缩量 + 随后快速收复
**对倒/制造成交** 换手异常放大但价格几乎不动 + 资金净额接近 0
**出货/派发**     价格高位 + 资金净流出 + 放量滞涨（量增价平/价跌）
**打压吸筹**      价格下跌 + 资金净流入（越跌越买）+ 底部区域
**试盘**          盘中大幅冲高回落 / 长上影 + 量能突增 + 随后回归
**护盘**          大盘大跌而个股抗跌 + 资金净流入 + 关键支撑位
**观望**          资金与价格均无显著信号
================  ==========================================================

另外输出一个 **-1~1 的意图分**（正=偏多意图、负=偏空意图），
以及"意图的持续性"（连续多少天同向），作为因子纳入统一打分体系。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor, safe_float
from .utils import get_logger

log = get_logger()

#: 意图 → (中文名, 方向, 基础强度, 可信度)
INTENT_SPEC = {
    "accumulate":      ("建仓/吃货", DIR_BULL, 0.62, 0.70),
    "accumulate_dip":  ("打压吸筹", DIR_BULL, 0.66, 0.66),
    "markup":          ("拉升", DIR_BULL, 0.72, 0.72),
    "washout":         ("洗盘", DIR_BULL, 0.55, 0.62),
    "support":         ("护盘", DIR_BULL, 0.50, 0.64),
    "probe":           ("试盘", DIR_NEUTRAL, 0.35, 0.55),
    "churn":           ("对倒/对敲", DIR_NEUTRAL, 0.35, 0.58),
    "distribute":      ("出货/派发", DIR_BEAR, 0.70, 0.70),
    "dump":            ("恐慌抛售", DIR_BEAR, 0.72, 0.68),
    "watch":           ("观望", DIR_NEUTRAL, 0.15, 0.50),
}


def _col(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns:
            return df[n].astype(float)
    return None


def analyze_main_force(df: pd.DataFrame, flow: pd.DataFrame | None,
                       frame: pd.DataFrame | None = None,
                       snapshot: dict | None = None,
                       lookback: int = 60) -> dict:
    """识别主力当前操盘意图。

    Parameters
    ----------
    df : 个股日线（用于价格位置、量能、K线形态）
    flow : 资金流表（``main_net`` / ``super_net`` / ``big_net`` / ``main_pct`` …）
    frame : 同 df，仅为兼容 engine 的调用签名
    snapshot : 实时快照（换手率、量比等）
    lookback : 参与判定的交易日数

    Returns
    -------
    dict：``{"intent","intent_cn","direction","strength","confidence","score",
    "evidence":[...],"detail","series":[...],"persistence":n,"next_action":...}``
    """
    df = frame if (df is None and frame is not None) else df
    out: dict = {"intent": "watch", "intent_cn": "观望", "direction": DIR_NEUTRAL,
                 "strength": 0.15, "confidence": 0.4, "score": 0.0,
                 "evidence": [], "detail": "数据不足，无法判定主力意图",
                 "series": [], "persistence": 0, "next_action": ""}
    price = _col(df, "close") if isinstance(df, pd.DataFrame) else None
    if price is None or len(price) < 25:
        return out

    price = price.dropna()
    n = len(price)
    cur = float(price.iloc[-1])
    high = _col(df, "high")
    low = _col(df, "low")
    vol = _col(df, "volume")
    turnover = _col(df, "turnover")

    # ---- 价格特征 ------------------------------------------------------ #
    ret1 = float(price.iloc[-1] / price.iloc[-2] - 1) * 100 if n > 1 else 0.0
    ret5 = float(price.iloc[-1] / price.iloc[-6] - 1) * 100 if n > 5 else 0.0
    ret20 = float(price.iloc[-1] / price.iloc[-21] - 1) * 100 if n > 20 else 0.0
    win = min(n, 250)
    hi, lo = float(price.tail(win).max()), float(price.tail(win).min())
    pos = (cur - lo) / max(hi - lo, 1e-9)          # 0~1 区间位置
    # 波动/振幅
    amp = float(((high - low) / price.shift(1) * 100).tail(20).mean()) \
        if (high is not None and low is not None) else float("nan")
    vol_ma5 = float(vol.tail(5).mean()) if vol is not None and len(vol) >= 5 else float("nan")
    vol_ma20 = float(vol.tail(20).mean()) if vol is not None and len(vol) >= 20 else float("nan")
    vol_ratio = vol_ma5 / vol_ma20 if (vol_ma20 == vol_ma20 and vol_ma20) else float("nan")
    # 上影线（试盘特征）
    upper_shadow = float("nan")
    if high is not None and low is not None and n:
        rng = float(high.iloc[-1] - low.iloc[-1])
        if rng > 0:
            upper_shadow = float(high.iloc[-1] - max(price.iloc[-1], float(df["open"].iloc[-1])
                                                     if "open" in df.columns else price.iloc[-1])) / rng

    # ---- 资金特征 ------------------------------------------------------ #
    cash: dict = {}
    if isinstance(flow, pd.DataFrame) and not flow.empty and "main_net" in flow.columns:
        f = flow.copy()
        f["main_net"] = pd.to_numeric(f["main_net"], errors="coerce")
        recent = f.tail(20)
        cash["net_5"] = float(recent["main_net"].tail(5).sum())
        cash["net_10"] = float(recent["main_net"].tail(10).sum())
        cash["net_20"] = float(recent["main_net"].sum())
        cash["net_1"] = float(recent["main_net"].iloc[-1])
        cash["inflow_days"] = int((recent["main_net"] > 0).sum())
        cash["outflow_days"] = int((recent["main_net"] < 0).sum())
        # 连续同向天数
        sgn = np.sign(recent["main_net"].to_numpy())
        run = 0
        for v in sgn[::-1]:
            if v == 0:
                break
            if run == 0 or np.sign(run) == v:
                run = v if run == 0 else run + v
            else:
                break
        cash["run"] = int(run)
        pct = _col(f, "main_pct")
        if pct is not None:
            cash["main_pct_5"] = float(pct.tail(5).mean())
        # 单日结构：超大单/大单 vs 中小单
        for c, k in (("super_net", "super_1"), ("big_net", "big_1"),
                     ("mid_net", "mid_1"), ("small_net", "small_1")):
            col = _col(f, c)
            if col is not None:
                cash[k] = float(col.iloc[-1])
        # 与价格的背离：资金净额方向 vs 价格方向
        if "close" in f.columns:
            fc = pd.to_numeric(f["close"], errors="coerce")
            cash["price_5"] = float(fc.iloc[-1] / fc.iloc[-6] - 1) * 100 if len(fc) > 5 else float("nan")

    # ---- 逐条判定 ------------------------------------------------------ #
    ev: list[str] = []
    scores: dict[str, float] = {}

    net5 = cash.get("net_5", float("nan"))
    net20 = cash.get("net_20", float("nan"))
    scale = abs(net20) if net20 == net20 and abs(net20) > 1e-6 else \
        (abs(net5) if net5 == net5 and abs(net5) > 1e-6 else 1.0)

    if net5 == net5:
        ev.append("近5日主力净额 %s，近20日 %s（20日中 %d 天净流入）"
                  % (_amt(net5), _amt(net20) if net20 == net20 else "—",
                     cash.get("inflow_days", 0)))
    if vol_ratio == vol_ratio:
        ev.append("5日/20日量比 %.2f%s" % (vol_ratio, _vol_word(vol_ratio)))
    ev.append("股价位于一年区间 %.0f%% 位置，近5日 %+.2f%%、近20日 %+.2f%%"
              % (pos * 100, ret5, ret20))

    inflow = net5 == net5 and net5 > 0
    outflow = net5 == net5 and net5 < 0
    strong_in = net5 == net5 and abs(net5) > scale * 0.25
    flat_price = abs(ret5) < 2.5
    up_price = ret5 > 2.5
    down_price = ret5 < -2.5
    low_pos = pos < 0.40
    high_pos = pos > 0.70
    big_vol = vol_ratio == vol_ratio and vol_ratio > 1.35
    small_vol = vol_ratio == vol_ratio and vol_ratio < 0.80

    # ① 建仓/吃货：资金进 + 价格没怎么涨（悄悄吸）
    if inflow and flat_price and low_pos:
        s = 0.45 + min(abs(net5) / max(scale, 1e-9), 1.0) * 0.25
        scores["accumulate"] = s
        ev.append("资金净流入而股价温和（未大幅拉升）→ 典型的低调吸筹特征")
    # ② 打压吸筹：价格跌 + 资金进
    if inflow and down_price:
        scores["accumulate_dip"] = 0.60 + (0.08 if low_pos else 0.0)
        ev.append("股价下跌但主力资金反而净流入 → 越跌越买，意在低位收集筹码")
    # ③ 拉升：资金大进 + 上涨（放量是加分项，不是必要条件）
    #    注意：不要用"位置不高"把强势股排除掉——一只已经涨了 20% 的股票
    #    位置自然到高位，但它正处在拉升过程中。位置只用来调整**强度**与风险提示。
    sustained = ret20 == ret20 and ret20 > 8          # 20日累计涨幅明显
    if inflow and up_price and (big_vol or strong_in or sustained):
        s = 0.55 + (0.12 if big_vol else 0.0) + (0.06 if sustained else 0.0)
        if high_pos:
            s -= 0.06                      # 高位拉升，追高风险上升，略降强度
        if low_pos:
            s += 0.08
        scores["markup"] = max(s, 0.45)
        ev.append("资金大幅流入且股价上行%s → 处于主动拉升阶段%s"
                  % ("、量能配合" if big_vol else "", "（位置已偏高，注意追高风险）"
                     if high_pos else ""))
    # ④ 洗盘：快速下探但资金没大幅流出 + 缩量或长下影
    if down_price and (not outflow or abs(net5) < scale * 0.15) and (small_vol or
                                                                   (upper_shadow == upper_shadow and upper_shadow > 0.12)):
        scores["washout"] = 0.50 + (0.10 if small_vol else 0.0)
        ev.append("股价快速回落但资金未明显流出、量能萎缩 → 更像是洗盘而非出货")
    # ⑤ 出货/派发：位置高 + 资金流出 + 放量滞涨
    if outflow and high_pos and (big_vol or flat_price):
        scores["distribute"] = 0.58 + (0.12 if high_pos and big_vol else 0.0)
        ev.append("股价处于高位区间而主力资金净流出%s → 派发嫌疑大" %
                  ("、且放量滞涨" if big_vol and flat_price else ""))
    # ⑥ 恐慌抛售：下跌 + 大幅流出
    if outflow and down_price and big_vol:
        scores["dump"] = 0.55 + min(abs(net5) / max(scale, 1e-9), 1.0) * 0.18
        ev.append("放量下跌且资金大幅流出 → 抛压沉重，属被动或恐慌性卖出")
    # ⑦ 护盘：位置在支撑附近 + 资金进 + 大盘弱（此处用振幅收敛代理）
    if inflow and low_pos and amp == amp and amp < 3.0:
        scores["support"] = 0.42
        ev.append("低位窄幅震荡伴随资金净流入 → 有资金在关键位置承接（护盘/吸筹）")
    # ⑧ 对倒：换手高但价格不动 + 资金净额接近 0
    if cash.get("net_5") == cash.get("net_5") and abs(net5) < scale * 0.08 and big_vol and flat_price:
        scores["churn"] = 0.50
        ev.append("成交明显放大但股价几乎不动、资金净额接近零 → 存在对倒/制造成交量的特征")
    # ⑨ 试盘：长上影 + 量能突增
    if upper_shadow == upper_shadow and upper_shadow > 0.30 and big_vol:
        scores["probe"] = 0.45
        ev.append("出现长上影且量能突增 → 主力在试探上方抛压（试盘）")

    # ---- 选取得分最高的意图 -------------------------------------------- #
    if scores:
        best = max(scores.items(), key=lambda kv: kv[1])
        intent, raw_score = best
        name, direction, base_s, base_c = INTENT_SPEC[intent]
        strength = min(base_s + (raw_score - 0.5) * 0.5, 0.9)
        confidence = min(base_c + 0.05 * (len(scores) == 1), 0.85)
    else:
        intent = "watch"
        name, direction, base_s, base_c = INTENT_SPEC["watch"]
        strength, confidence = base_s, base_c
        ev.append("资金与价格均无显著信号，暂判定为观望")

    # 竞争意图：把次高分意图也报出来（意图常有混合）
    rivals = sorted(((k, v) for k, v in scores.items() if k != intent),
                    key=lambda kv: -kv[1])[:2]
    mixed = "、".join("%s(%.2f)" % (INTENT_SPEC[k][0], v) for k, v in rivals)

    # ---- 意图持续性 ---------------------------------------------------- #
    persistence = abs(cash.get("run", 0)) if cash else 0

    # ---- 意图分（-1~1）------------------------------------------------- #
    score = 0.0
    if direction != DIR_NEUTRAL:
        score = direction * strength * confidence * 1.6
    score = max(-1.0, min(1.0, score))
    # 资金方向与价格背离时，给意图分打一点折扣（信号不够纯）
    if cash and cash.get("net_5") == cash.get("net_5"):
        if np.sign(cash["net_5"]) != np.sign(score) and abs(score) > 0.1:
            score *= 0.6
            ev.append("注意：资金方向与价格方向不一致，意图判断的确定性下降")

    out.update({
        "intent": intent, "intent_cn": name, "direction": direction,
        "strength": round(strength, 3), "confidence": round(confidence, 3),
        "score": round(score, 3), "evidence": ev,
        "persistence": int(persistence),
        "mixed": mixed,
        "next_action": _next_action(intent),
        "detail": _compose_detail(name, direction, ev, persistence, mixed),
        "cash": {k: (round(v, 2) if isinstance(v, float) and v == v else v)
                 for k, v in (cash or {}).items()},
        "metrics": {"pos": round(pos, 3), "ret5": round(ret5, 2), "ret20": round(ret20, 2),
                    "vol_ratio": round(vol_ratio, 3) if vol_ratio == vol_ratio else None,
                    "amp20": round(amp, 3) if amp == amp else None,
                    "upper_shadow": round(upper_shadow, 3) if upper_shadow == upper_shadow else None},
    })
    return out


def _amt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:
        return "—"
    if abs(f) >= 1e8:
        return "%+.2f亿" % (f / 1e8)
    if abs(f) >= 1e4:
        return "%+.2f万" % (f / 1e4)
    return "%+.0f" % f


def _vol_word(r: float) -> str:
    if r != r:
        return ""
    if r > 1.8:
        return "（显著放量）"
    if r > 1.35:
        return "（放量）"
    if r < 0.7:
        return "（地量）"
    if r < 0.85:
        return "（缩量）"
    return "（量能平稳）"


def _next_action(intent: str) -> str:
    return {
        "accumulate": "主力在建仓期，股价通常被压制，适合逢低跟随、不宜追高",
        "accumulate_dip": "下跌中资金仍在吸筹，可在企稳信号出现后关注，不宜盲目抄底",
        "markup": "处于拉升阶段，可持有但需设好止盈，注意放量滞涨信号",
        "washout": "洗盘特征明显，急跌往往是机会，但要等资金转为净流入再确认",
        "support": "有资金护盘，下方支撑相对可靠，可观察能否放量突破",
        "probe": "主力在试盘，方向未定，等待后续量价确认",
        "churn": "对倒制造成交量，信号质量差，建议降低仓位、观望为主",
        "distribute": "高位派发嫌疑，反弹是减仓机会，不宜追高",
        "dump": "抛压沉重，注意控制风险，等待缩量止跌",
        "watch": "无明显主力动作，跟随大盘与行业波动为主",
    }.get(intent, "")


def _compose_detail(name: str, direction: int, ev: list, persistence: int,
                    mixed: str) -> str:
    head = "主力当前意图判定为「%s」" % name
    if persistence >= 2:
        head += "（已连续 %d 日同向）" % persistence
    body = "；".join(ev[:4])
    tail = ""
    if mixed:
        tail = "。同时存在次要意图：%s" % mixed
    return "%s。依据：%s%s" % (head, body, tail)


def intent_series(df: pd.DataFrame, flow: pd.DataFrame | None,
                  window: int = 40, min_bars: int = 25) -> list:
    """滚动判定最近 ``window`` 天每天的意图（用于看"意图变化过程"）。

    ``min_bars`` 是判定所需的最少 K 线数（与 :func:`analyze_main_force` 一致），
    数据太短时直接返回空列表，避免"看起来跑了但结果为空"的困惑。
    """
    out: list = []
    if not isinstance(df, pd.DataFrame) or len(df) < min_bars + 2:
        return out
    n = len(df)
    start = max(min_bars, n - window)
    for i in range(start, n + 1):
        sub = df.iloc[:i]
        fsub = None
        if isinstance(flow, pd.DataFrame) and not flow.empty:
            try:
                fsub = flow[flow["date"] <= sub["date"].iloc[-1]]
                if len(fsub) < 5:
                    fsub = flow.tail(i)
            except Exception:  # noqa: BLE001
                fsub = flow
        try:
            r = analyze_main_force(sub, fsub)
            out.append({"date": str(pd.to_datetime(sub["date"].iloc[-1]))[:10],
                        "intent": r["intent"], "intent_cn": r["intent_cn"],
                        "score": r["score"]})
        except Exception:  # noqa: BLE001
            continue
    return out


def intent_factors(df: pd.DataFrame, flow: pd.DataFrame | None,
                   frame: pd.DataFrame | None = None,
                   snapshot: dict | None = None) -> list:
    """把主力意图转成因子。"""
    r = analyze_main_force(df, flow, frame, snapshot)
    out: list[Factor] = []
    intent = r.get("intent", "watch")
    if intent == "watch" or r.get("confidence", 0) < 0.35:
        return out
    out.append(Factor(
        id="main_intent", name="主力操盘意图：%s" % r["intent_cn"],
        category=Category.CAPITAL,
        direction=r["direction"], strength=r["strength"],
        reliability=r["confidence"],
        value="意图分 %+.2f｜持续性 %d 日" % (r["score"], r.get("persistence", 0)),
        detail=r["detail"], theory="主力行为学/博弈分析",
        horizon=("d1", "d5", "d10", "d20", "m1")))
    # 意图的后续操作提示单独成一条弱因子（帮助用户理解，不参与方向）
    na = r.get("next_action")
    if na:
        out.append(Factor(
            id="main_intent_action", name="主力意图对应操作提示",
            category=Category.CAPITAL, direction=DIR_NEUTRAL, strength=0.2,
            reliability=0.4, value=r["intent_cn"], detail=na,
            theory="主力行为学", horizon=("d5", "d10", "d20")))
    return out


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    from .selftest_data import make_ohlcv

    print("=== 离线自测：构造四种典型主力行为 ===")

    def _mk(price_path, main_nets, seed=1, vol_boost=1.0):
        """按给定价格路径与主力净额构造数据，检验意图识别是否正确。

        ``vol_boost`` 用于放大**后段**成交量，模拟"放量"场景。
        """
        n = len(price_path)
        df = make_ohlcv(n, seed=seed, start_price=20.0)
        c = np.array(price_path, dtype=float)
        df["close"] = c
        df["open"] = c * 0.995
        df["high"] = c * 1.01
        df["low"] = c * 0.99
        v = df["volume"].to_numpy(dtype=float).copy()
        if vol_boost != 1.0:
            # 用前 75% 的均量作基准做"逐级放大"，确保 5日均量真的高于 20日均量
            cut = int(n * 0.75)
            base = float(np.mean(v[:cut])) if cut > 0 else float(np.mean(v))
            ramp = np.linspace(vol_boost, vol_boost * 2.0, n - cut)
            v[cut:] = base * ramp
        df["volume"] = v
        df["amount"] = v * c
        flow = pd.DataFrame({
            "date": df["date"],
            "main_net": main_nets,
            "super_net": np.array(main_nets) * 0.6,
            "big_net": np.array(main_nets) * 0.4,
            "mid_net": np.array(main_nets) * -0.3,
            "small_net": np.array(main_nets) * -0.7,
            "main_pct": np.array(main_nets) / 1e7,
            "close": c,
        })
        return df, flow

    cases = []

    # ① 建仓：价格横盘微跌 + 资金持续流入 + 低位
    p = list(np.linspace(30, 20, 60)) + list(np.linspace(20, 20.4, 20))
    f = list(np.linspace(0, 0, 60)) + [3e6, 4e6, 2e6, 5e6, 3e6] * 4
    cases.append(("建仓/吃货", _mk(p, f, 3)))

    # ② 拉升：价格上行 + 资金大额流入 + 放量
    p = list(np.linspace(20, 20, 60)) + list(np.linspace(20, 24, 20))
    f = list(np.linspace(0, 0, 60)) + [8e6, 1.2e7, 9e6, 1.1e7, 7e6] * 4
    cases.append(("拉升", _mk(p, f, 5, vol_boost=2.5)))
    # 也验证"缩量但持续大涨"能被识别为拉升（放量不是必要条件）
    cases.append(("拉升(缩量)", _mk(p, f, 5, vol_boost=1.0)))

    # ③ 出货：高位 + 资金流出 + 放量滞涨
    p = list(np.linspace(15, 32, 60)) + list(np.linspace(32, 31.5, 20))
    f = list(np.linspace(0, 0, 60)) + [-6e6, -8e6, -5e6, -9e6, -7e6] * 4
    cases.append(("出货/派发", _mk(p, f, 7)))

    # ④ 打压吸筹：价格下跌 + 资金流入
    p = list(np.linspace(30, 22, 60)) + list(np.linspace(22, 20, 20))
    f = list(np.linspace(0, 0, 60)) + [5e6, 6e6, 4e6, 7e6, 5e6] * 4
    cases.append(("打压吸筹", _mk(p, f, 9)))

    ok = 0
    for label, (df, flow) in cases:
        r = analyze_main_force(df, flow)
        hit = label.split("/")[0] in r["intent_cn"] or r["intent_cn"] in label
        ok += 1 if hit else 0
        print("\n  【%s】→ 判定「%s」%s  意图分 %+.2f (强度%.2f 可信度%.2f)"
              % (label, r["intent_cn"], "✓" if hit else "✗", r["score"],
                 r["strength"], r["confidence"]))
        print("     依据：%s" % r["detail"][:150])
        print("     提示：%s" % r["next_action"])
    print("\n典型主力行为识别正确 %d/%d" % (ok, len(cases)))

    # 因子（cases 的元素是 (意图标签, (df, flow))，这里取第一组的 df/flow）
    _label0, (_df0, _flow0) = cases[0]
    fs = intent_factors(_df0, _flow0)
    print("\n生成因子 %d 条：%s" % (len(fs), [f.name for f in fs]))
    assert fs, "建仓案例应生成因子"
    for f in fs:
        assert 0 <= f.strength <= 1 and 0 <= f.reliability <= 1
        assert f.theory and f.detail
    # 空输入
    assert analyze_main_force(None, None)["intent"] == "watch"
    assert intent_factors(None, None) == []
    # 序列（同样要用解包后的 df/flow）
    ser = intent_series(_df0, _flow0, window=15)
    print("\n意图序列（最近 %d 天）：%s" % (len(ser), [x["intent_cn"] for x in ser[-6:]]))
    assert len(ser) > 0, "意图序列应有结果"
    assert ok >= 3, "四种典型行为至少应识别正确 3 种"
    print("\nmain_force_intent self-test OK")

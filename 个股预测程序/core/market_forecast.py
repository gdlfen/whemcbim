# -*- coding: utf-8 -*-
"""
大盘与板块的概率预测 —— 把个股那套预测引擎扩展到指数与行业板块。

为什么可以复用
--------------
预测的核心是"因子打分 → logit 概率 → 波动率定区间"，
这套机制与标的类型无关。指数/板块同样有 K 线、同样适用技术面与量价因子。
所以这里**不重写一套**，而是：
  1. 为指数/板块装配因子（技术面 + 关键点 + 量价 + 均值回归/位置）
  2. 复用 :func:`core.forecast.build_forecasts` 得到 8 个周期的概率与区间

与个股的差别（必须说明）
------------------------
* 指数/板块**没有** PE/PB、龙虎榜、股东户数、业绩预告 —— 这些维度直接缺席，
  因子数比个股少，因此证据充分度低、置信度会自动降低（这是正确行为）。
* 板块除技术面外，额外用**当日强弱**（涨跌幅、上涨家数占比）作为情绪证据。

数据源
------
* 指数历史：``global_market.fetch_index_history``（腾讯/新浪，已验证）
* 板块历史：同花顺 ``stock_board_industry_index_ths``（akshare，已验证 436 行）
  —— 东财 push2his 在本机网络长期不可达，同花顺是稳定替代源。

诚实降级
--------
板块历史取不到时**不编造**：该板块显示"历史数据不可用，无法预测"，
只保留当日强弱信息。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .contracts import AnalysisBundle, Category, DIR_BEAR, DIR_BULL, Factor, FactorSet
from .utils import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
# 一、为指数/板块补充特有因子
# --------------------------------------------------------------------------- #
def _extra_factors(df: pd.DataFrame, name: str, kind: str) -> list:
    """补充"位置/均值回归/趋势强度"这类对指数与板块最有效的因子。"""
    out: list = []
    try:
        c = df["close"].astype(float).dropna()
        n = len(c)
        if n < 60:
            return out
        cur = float(c.iloc[-1])
        win = min(n, 250)
        hi, lo = float(c.tail(win).max()), float(c.tail(win).min())
        pos = (cur - lo) / max(hi - lo, 1e-9)
        ma20 = float(c.tail(20).mean())
        ma60 = float(c.tail(60).mean())
        ma250 = float(c.tail(250).mean()) if n >= 250 else ma60
        ret20 = (cur / float(c.iloc[-21]) - 1) * 100 if n > 21 else 0.0

        # 均线排列（指数/板块的趋势最可靠的判据）
        if ma20 > ma60 > ma250:
            out.append(Factor("mf_ma_align", "均线多头排列", Category.TECH,
                              DIR_BULL, 0.62, 0.72,
                              value="MA20>MA60>MA250",
                              detail="均线呈多头排列，中期趋势向上，"
                                     "这是%s最可靠的趋势判据。" % ("指数" if kind == "index" else "板块"),
                              theory="均线系统/趋势跟踪",
                              horizon=("d5", "d10", "d20", "m1")))
        elif ma20 < ma60 < ma250:
            out.append(Factor("mf_ma_align", "均线空头排列", Category.TECH,
                              DIR_BEAR, 0.62, 0.72,
                              value="MA20<MA60<MA250",
                              detail="均线呈空头排列，中期趋势向下。",
                              theory="均线系统/趋势跟踪",
                              horizon=("d5", "d10", "d20", "m1")))

        # 区间位置 + 均值回归（长周期最有效）
        if pos >= 0.85:
            out.append(Factor("mf_position", "处于年内高位", Category.TECH,
                              DIR_BEAR, 0.55, 0.62,
                              value="一年区间 %.0f%% 位置" % (pos * 100),
                              detail="已处于一年区间 %.0f%% 高位，"
                                     "均值回归压力上升——涨多了必跌是A股最稳定的规律之一。"
                                     % (pos * 100),
                              theory="均值回归/区间理论",
                              horizon=("d20", "m1", "q1")))
        elif pos <= 0.15:
            out.append(Factor("mf_position", "处于年内低位", Category.TECH,
                              DIR_BULL, 0.55, 0.62,
                              value="一年区间 %.0f%% 位置" % (pos * 100),
                              detail="处于一年区间 %.0f%% 低位，"
                                     "向下空间有限、反弹概率上升。" % (pos * 100),
                              theory="均值回归/区间理论",
                              horizon=("d20", "m1", "q1")))

        # 20 日动量（趋势延续）
        if abs(ret20) >= 3.0:
            out.append(Factor("mf_momentum", "20日动量%s" % ("向上" if ret20 > 0 else "向下"),
                              Category.TECH,
                              DIR_BULL if ret20 > 0 else DIR_BEAR,
                              min(abs(ret20) / 25.0 + 0.2, 0.75), 0.64,
                              value="%+.2f%%" % ret20,
                              detail="近20日涨跌 %+.2f%%，%s" % (
                                  ret20,
                                  "动量向上，趋势有延续性" if ret20 > 0
                                  else "动量向下，反弹前先要止跌"),
                              theory="动量效应", horizon=("d5", "d10", "d20")))
    except Exception as exc:  # noqa: BLE001
        log.debug("%s 补充因子失败: %s", name, exc)
    return out


def _board_strength_factors(row: dict) -> list:
    """板块当日强弱（来自同花顺实时摘要）。"""
    out: list = []
    try:
        pct = float(row.get("涨跌幅") or 0)
        up = float(row.get("上涨家数") or 0)
        dn = float(row.get("下跌家数") or 0)
        net = float(row.get("净流入") or 0)
        breadth = up / max(up + dn, 1) * 100
        if abs(pct) < 0.01 and breadth == 0:
            return out
        out.append(Factor("mf_board_today", "板块当日 %.2f%%" % pct,
                          Category.SENTIMENT,
                          DIR_BULL if pct > 0.5 else (DIR_BEAR if pct < -0.5 else 0),
                          min(abs(pct) / 6.0 + 0.2, 0.8), 0.6,
                          value="%+.2f%%｜%d涨/%d跌｜净流入 %.2f亿"
                                % (pct, int(up), int(dn), net),
                          detail="板块内 %d 只上涨、%d 只下跌（宽度 %.0f%%），"
                                 "净流入 %.2f 亿元。宽度高说明是整体性上涨，"
                                 "而非个别龙头拉动。" % (int(up), int(dn), breadth, net),
                          theory="板块宽度/资金流", horizon=("d1", "d5")))
    except Exception as exc:  # noqa: BLE001
        log.debug("板块强弱因子失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 二、单个标的的预测
# --------------------------------------------------------------------------- #
def forecast_series(df: pd.DataFrame, name: str, kind: str = "index",
                    extra_factors: list | None = None) -> list:
    """对一段行情给出 8 个周期的概率预测。

    Returns
    -------
    list[dict]：每项含 ``horizon/label/p_up/exp_return/ci_low/ci_high/
    confidence/trend/big_move_threshold``
    """
    if df is None or not len(df) or "close" not in df.columns:
        return []
    try:
        from . import technical as T
        from . import forecast as F

        factors: list = []
        try:
            factors.extend(T.technical_factors(df) or [])
        except Exception as exc:  # noqa: BLE001
            log.debug("%s 技术因子失败: %s", name, exc)
        try:
            from . import keypoint as KP

            kp = KP.detect_key_events(df)
            factors.extend(KP.keypoint_factors(df, kp) or [])
        except Exception as exc:  # noqa: BLE001
            log.debug("%s 关键点因子失败: %s", name, exc)
        factors.extend(_extra_factors(df, name, kind))
        factors.extend(extra_factors or [])
        # 指数专属：估值分位 / 两融余额 / 北向资金 / 新增开户数
        # （这是大盘的"基本面替代物"，个股没有对应维度）
        if kind == "index":
            try:
                from . import market_fundamental as MF

                key = next((k for k, v in INDEX_LABELS.items() if v == name), "hs300")
                factors.extend(MF.fundamental_factors(key) or [])
            except Exception as exc:  # noqa: BLE001
                log.debug("%s 大盘基本面替代物失败: %s", name, exc)
        if not factors:
            return []

        bundle = AnalysisBundle(
            symbol=str(name), name=str(name),
            as_of=str(pd.to_datetime(df["date"].iloc[-1]))[:10]
            if "date" in df.columns else "",
            factors=FactorSet(factors))
        bundle.ohlcv = df
        res = F.build_forecasts(bundle, use_ml=False)
        rows = []
        for f in res.forecasts:
            rows.append({
                "horizon": f.horizon, "label": f.label, "bars": f.bars,
                "p_up": round(float(f.p_up), 4),
                "p_down": round(float(f.p_down), 4),
                "exp_return": round(float(f.exp_return), 3),
                "ci_low": round(float(f.ci_low), 2),
                "ci_high": round(float(f.ci_high), 2),
                "confidence": round(float(f.confidence), 3),
                "trend": f.trend,
                "big_move_threshold": float(getattr(f, "big_move_threshold", 0) or 0),
            })
        return rows
    except Exception as exc:  # noqa: BLE001
        log.warning("%s 概率预测失败: %s", name, exc)
        return []


# --------------------------------------------------------------------------- #
# 三、批量：指数
# --------------------------------------------------------------------------- #
INDEX_LABELS = {
    "sh": "上证指数", "sz": "深证成指", "cyb": "创业板指",
    "hs300": "沪深300", "zz500": "中证500", "kc50": "科创50",
}


def forecast_indices(frames: dict, max_workers: int = 4) -> dict:
    """对多个指数并发给出概率预测。"""
    out: dict = {}
    tasks = [(k, v) for k, v in (frames or {}).items()
             if k in INDEX_LABELS and v is not None and len(v)]
    if not tasks:
        return out

    def _one(item):
        key, df = item
        return key, forecast_series(df, INDEX_LABELS.get(key, key), "index")

    with ThreadPoolExecutor(max_workers=max(1, min(len(tasks), max_workers))) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for fu in as_completed(futs):
            try:
                key, rows = fu.result()
                if rows:
                    out[key] = rows
            except Exception as exc:  # noqa: BLE001
                log.debug("指数预测失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 四、批量：行业板块（同花顺源）
# --------------------------------------------------------------------------- #
def fetch_ths_board_summary() -> list:
    """同花顺行业板块当日摘要（涨跌幅/上涨家数/净流入/领涨股）。"""
    try:
        import akshare as ak

        df = ak.stock_board_industry_summary_ths()
        if df is None or not len(df):
            return []
        return df.to_dict("records")
    except Exception as exc:  # noqa: BLE001
        log.debug("同花顺板块摘要失败: %s", exc)
        return []


def fetch_ths_board_history(name: str, days: int = 400) -> pd.DataFrame | None:
    """同花顺单板块历史K线 —— 复用 market_board 的实现（单一来源）。"""
    try:
        from .market_board import fetch_ths_board_history as _impl

        return _impl(name, days=days)   # 复用 market_board，避免两份实现漂移
    except Exception:  # noqa: BLE001
        pass
    try:
        import akshare as ak

        df = ak.stock_board_industry_index_ths(symbol=name)
        if df is None or not len(df):
            return None
        ren = {"日期": "date", "开盘价": "open", "最高价": "high",
               "最低价": "low", "收盘价": "close", "成交量": "volume"}
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
        if "date" not in df.columns or "close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df.tail(max(days, 120))
    except Exception as exc:  # noqa: BLE001
        log.debug("同花顺板块 %s 历史失败: %s", name, exc)
        return None


def forecast_boards(boards: list, max_boards: int = 12,
                    max_workers: int = 3) -> list:
    """对行业板块给出概率预测。

    ``boards`` 为同花顺摘要行（含 ``板块``/``涨跌幅``/``上涨家数``/``下跌家数``/
    ``净流入``）。为控制耗时，只对**当日最强与最弱各若干**个做预测。
    """
    if not boards:
        return []
    # 按涨跌幅排序，取最强与最弱若干（真正需要判断"还能不能追/能不能抄"的）
    srt = sorted(boards, key=lambda r: -(float(r.get("涨跌幅") or 0)))
    half = max(1, max_boards // 2)
    picked = srt[:half] + (srt[-half:] if len(srt) > half else [])
    seen, uniq = set(), []
    for r in picked:
        nm = str(r.get("板块") or "")
        if nm and nm not in seen:
            seen.add(nm)
            uniq.append(r)

    def _one(row):
        nm = str(row.get("板块") or "")
        if not nm:
            return None
        hist = fetch_ths_board_history(nm)
        # ---- 历史新鲜度校验 ----
        # 实测同花顺板块历史最新只到 2024-01-08（距今 2 年多），
        # 用它算出的"概率预测"是拿旧数据冒充现在，必须拦下。
        if hist is not None and len(hist) and "date" in hist.columns:
            try:
                from . import freshness as _FR

                r = _FR.validate("board_history", hist["date"].max())
                if r.get("stale"):
                    item_note = ("历史数据过期（最新 %s，距今 %.0f 天，上限 %d 天），"
                                 "**不给出概率预测**——用旧数据算出来的概率没有意义"
                                 % (r.get("date"), r.get("age_days") or 0,
                                    r.get("limit") or 5))
                    return {"name": nm, "pct": float(row.get("涨跌幅") or 0),
                            "up": row.get("上涨家数"), "down": row.get("下跌家数"),
                            "net_inflow": row.get("净流入"),
                            "leader": row.get("领涨股"),
                            "forecasts": [], "note": item_note}
            except Exception:  # noqa: BLE001
                pass
        item = {
            "name": nm,
            "pct": float(row.get("涨跌幅") or 0),
            "up": row.get("上涨家数"), "down": row.get("下跌家数"),
            "net_inflow": row.get("净流入"),
            "leader": row.get("领涨股"),
            "forecasts": [], "note": "",
        }
        if hist is None or len(hist) < 120:
            item["note"] = "历史数据不可用，无法给出概率预测（仅显示当日强弱）"
            return item
        # 板块额外汇总当日强弱作为情绪证据
        extra = _board_strength_factors(row)
        item["forecasts"] = forecast_series(hist, nm, "board", extra)
        if not item["forecasts"]:
            item["note"] = "因子不足，未能给出预测"
        return item

    out: list = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for r in pool.map(_one, uniq):
            if r:
                out.append(r)
    out.sort(key=lambda x: -abs(float(x.get("pct") or 0)))
    return out


# --------------------------------------------------------------------------- #
# 五、一站式
# --------------------------------------------------------------------------- #
def analyze(frames: dict | None = None, *, with_boards: bool = True,
            max_boards: int = 12) -> dict:
    """大盘 + 板块的概率预测。"""
    out: dict = {"indices": {}, "boards": [], "detail": "", "source": {}}
    idx = forecast_indices(frames or {})
    out["indices"] = idx
    if idx:
        out["source"]["indices"] = "指数历史（腾讯/新浪）"
    if with_boards:
        summary = fetch_ths_board_summary()
        if summary:
            out["source"]["boards"] = "同花顺行业板块（%d 个）" % len(summary)
            out["boards"] = forecast_boards(summary, max_boards=max_boards)
        else:
            out["source"]["boards"] = "同花顺板块数据不可用"
    n_idx = len(idx)
    n_bd = len([b for b in out["boards"] if b.get("forecasts")])
    bits = []
    if n_idx:
        bits.append("已预测 %d 个指数" % n_idx)
    if out["boards"]:
        bits.append("%d 个板块中 %d 个给出了概率预测"
                    % (len(out["boards"]), n_bd))
    out["detail"] = "；".join(bits) or "未能生成大盘/板块预测"
    return out


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    from .selftest_data import make_ohlcv

    print("=== 1. 合成序列：持续上涨 ===")
    n = 400
    up = np.linspace(3000, 3600, n) + np.sin(np.arange(n) / 10) * 12
    df = make_ohlcv(n, seed=3, start_price=3000.0)
    df["close"] = up
    df["high"] = up * 1.008
    df["low"] = up * 0.992
    rows = forecast_series(df, "合成指数", "index")
    print("  %d 个周期" % len(rows))
    for r in rows[:4]:
        print("     %-16s P(涨)=%5.1f%% 期望%+6.2f%% conf=%.2f"
              % (r["label"], r["p_up"] * 100, r["exp_return"], r["confidence"]))
    assert len(rows) == 8, "应给出 8 个周期"
    for r in rows:
        assert 0.0 <= r["p_up"] <= 1.0
        assert r["ci_low"] <= r["ci_high"]
        assert r["confidence"] >= 0

    print("\n=== 2. 下跌序列 ===")
    dn = np.linspace(3600, 3000, n) + np.sin(np.arange(n) / 10) * 12
    d2 = make_ohlcv(n, seed=7, start_price=3600.0)
    d2["close"] = dn
    d2["high"] = dn * 1.008
    d2["low"] = dn * 0.992
    r2 = forecast_series(d2, "合成指数2", "index")
    assert r2 and r2[0]["p_up"] < 0.6, "下跌序列的短期上涨概率应偏低"

    print("\n=== 3. 板块当日强弱因子 ===")
    fs = _board_strength_factors({"涨跌幅": 3.2, "上涨家数": 40, "下跌家数": 3,
                                  "净流入": 12.5})
    assert fs, "应生成板块强弱因子"
    print("  因子:", [f.name for f in fs])

    print("\n=== 4. 边界 ===")
    assert forecast_series(None, "x") == []
    assert forecast_series(pd.DataFrame(), "x") == []
    assert forecast_indices({}) == {}
    assert forecast_boards([]) == []
    assert analyze(with_boards=False)["detail"]
    print("  空输入安全")
    print("\nmarket_forecast self-test OK")

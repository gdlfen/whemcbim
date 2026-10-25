# -*- coding: utf-8 -*-
"""
新增开户数：确认无法自动获取，改用**可验证的新鲜代理指标**。

已尝试并失败的路径（全部记录，避免重复）
----------------------------------------
1. akshare stock_account_statistics_em            → 止于 2023-08
2. 东财 datacenter RPT_ACCOUNT_STATISTICS 等 8 个  → 报表配置不存在
3. 东财 datacenter RPT_STOCK_OPEN_DATA（正确名）   → 止于 2023-08（同源）
4. 上交所 query.sse.com.cn 各 sqlId 猜测            → 返回空 JSON
5. 上交所 投资者统计页                              → 200 但无 sqlId/数据文件线索
6. 深交所 ShowReport CATALOGID=1803_sczm           → 证券类别统计，非开户数
7. 中国结算 统计月报页                              → 200 但无文件直链（JS 渲染）
8. 中登月报目录 chinaclear.cn/zdjs/editor_file/     → 404

结论：中登已停止公开发布月度新增投资者数；交易所口径（A股新开户）
虽有发布但接口不公开/需爬 JS。自动获取不可行。

替代方案：用**散户入场意愿**的可验证代理
----------------------------------------
开户数本质上测的是"散户入场意愿"。可用同样含义、且**日频新鲜**的指标代理：
  · 两融余额 20 日变化  → 杠杆资金入场意愿（已在 market_fundamental）
  · 成交额 20/120 日比  → 市场活跃度
  · 换手率 20 日均值    → 交易热度
三者合成「散户入场热度」分（0~100），并明确标注为**代理指标**，
不冒充满开户数。同时把开户数从因子中移除（数据停更，不参与打分）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor, safe_float
from .utils import get_logger

log = get_logger()


def retail_heat(df: pd.DataFrame, margin: dict | None = None) -> dict:
    """散户入场热度（开户数的代理指标）。

    Returns
    -------
    dict(score, level, items, detail, proxies)
    """
    out = {"score": None, "level": "", "items": [], "detail": "",
           "proxies": {}, "note": "代理指标，非开户数"}
    parts: list = []
    if df is not None and len(df) and "close" in df.columns:
        d = df.dropna(subset=["close"])
        # ---- 成交额水位：20日均值 / 120日均值 ----
        amt = None
        if "amount" in d.columns and d["amount"].notna().any():
            amt = d["amount"].astype(float)
        elif "volume" in d.columns:
            amt = d["close"].astype(float) * d["volume"].astype(float)
        if amt is not None:
            a = amt.dropna()
            if len(a) >= 120:
                r = float(a.tail(20).mean() / max(a.tail(120).mean(), 1e-9))
                out["proxies"]["成交额20/120日"] = round(r, 3)
                # 1.0 为常态；1.3 以上算热，0.8 以下算冷
                v = max(-1.0, min(1.0, (r - 1.0) / 0.35))
                parts.append(("成交额水位 %.2f" % r, v, 1.0))
        # ---- 换手率热度（用振幅/波动近似，无流通股本时的替代）----
        try:
            if len(d) >= 60:
                ret = d["close"].astype(float).pct_change().abs()
                h20 = float(ret.tail(20).mean())
                h60 = float(ret.tail(60).mean())
                if h60 > 0:
                    r2 = h20 / h60
                    out["proxies"]["振幅20/60日"] = round(r2, 3)
                    v2 = max(-1.0, min(1.0, (r2 - 1.0) / 0.4))
                    parts.append(("振幅热度 %.2f" % r2, v2, 0.7))
        except Exception:  # noqa: BLE001
            pass
    # ---- 两融余额 20 日变化：杠杆资金入场意愿 ----
    mg = margin or {}
    chg = mg.get("change_20d_pct")
    if chg is not None:
        out["proxies"]["两融20日变化%"] = chg
        v3 = max(-1.0, min(1.0, float(chg) / 8.0))
        parts.append(("两融20日 %+.1f%%" % chg, v3, 1.0))

    if not parts:
        out["detail"] = "缺少成交额/两融数据，无法计算散户入场热度"
        return out
    tot = sum(w for _, _, w in parts)
    s = sum(v * w for _, v, w in parts) / max(tot, 1e-9)
    score = (s + 1) / 2 * 100
    out["score"] = round(score, 1)
    out["items"] = [(n, round(v, 2)) for n, v, _ in parts]
    out["level"] = ("过热（散户蜂拥）" if score >= 80 else
                    "偏热" if score >= 62 else
                    "中性" if score >= 38 else
                    "偏冷" if score >= 20 else "冰点（无人问津）")
    out["detail"] = ("散户入场热度 %.0f/100（%s）｜%s。"
                     "**这是代理指标，不是开户数**——开户数已停止公开发布，"
                     "但入场意愿可以用成交额水位、振幅热度、两融变化来观测，"
                     "且这三项都是日频新鲜数据，比月频开户数更及时。"
                     % (score, out["level"],
                        "；".join("%s(%+.2f)" % (n, v) for n, v in out["items"])))
    return out


def heat_factors(heat: dict) -> list:
    """散户热度因子。过热偏空（反向指标），冰点偏多。"""
    out: list = []
    sc = heat.get("score")
    if sc is None:
        return out
    if sc >= 80:
        d, s, desc = DIR_BEAR, 0.55, "散户过热，历史上多为阶段顶部区域（反向指标）"
    elif sc >= 62:
        d, s, desc = DIR_NEUTRAL, 0.3, "散户偏热，情绪积极但需防追高"
    elif sc <= 20:
        d, s, desc = DIR_BULL, 0.52, "散户冰点，历史上多为底部区域（反向指标）"
    elif sc <= 38:
        d, s, desc = DIR_NEUTRAL, 0.28, "散户偏冷，观望为主"
    else:
        d, s, desc = DIR_NEUTRAL, 0.2, "散户情绪中性"
    out.append(Factor(
        "retail_heat", "散户入场热度 %.0f/100（%s）" % (sc, heat.get("level") or ""),
        Category.SENTIMENT, d, s, 0.58,
        value="%.0f/100" % sc, detail=heat.get("detail") or desc,
        theory="散户情绪/反向指标（开户数代理）",
        horizon=("d20", "m1", "q1")))
    return out


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    from .selftest_data import make_ohlcv

    def mk(scale=1.0, seed=3):
        n = 300
        c = 100 + np.sin(np.arange(n) / 25) * 6
        df = make_ohlcv(n, seed=seed, start_price=100.0)
        df["close"] = c
        v = df["volume"].to_numpy(float)
        v[-30:] *= scale          # 近期量能放大/缩小
        df["volume"] = v
        df["amount"] = v * c
        return df

    print("=== 1. 放量（散户趋热）===")
    h1 = retail_heat(mk(2.0), {"change_20d_pct": 6.0})
    print(" ", h1["detail"][:120])
    print("  proxies:", h1["proxies"], "| score:", h1["score"], "|", h1["level"])

    print("\n=== 2. 缩量（散户转冷）===")
    h2 = retail_heat(mk(0.4, seed=7), {"change_20d_pct": -6.0})
    print(" ", h2["detail"][:120])
    print("  score:", h2["score"], "|", h2["level"])
    assert h1["score"] > h2["score"], "放量应比缩量更热"

    print("\n=== 3. 因子（过热偏空、冰点偏多）===")
    for h in (h1, h2):
        fs = heat_factors(h)
        assert fs and fs[0].detail and fs[0].theory
        print("  %-30s %s" % (fs[0].name[:28],
                              {1: "多", -1: "空", 0: "中"}[fs[0].direction]))

    print("\n=== 4. 边界 ===")
    assert retail_heat(None)["detail"]
    assert retail_heat(None)["score"] is None
    assert heat_factors({}) == []
    assert retail_heat(None)["note"] == "代理指标，非开户数"
    print("  空输入安全；已标注为代理指标")
    print("\nretail_heat self-test OK")

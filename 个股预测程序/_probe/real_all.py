# -*- coding: utf-8 -*-
"""真实数据验证脚本：对 600519（贵州茅台）跑通五个模块的取数与因子生成。

用法：$env:PYTHONIOENCODING='utf-8'; python _probe\real_all.py
"""

from __future__ import annotations

import sys
import traceback

sys.path.insert(0, r"E:\deepseek desktop\个股预测程序")

import pandas as pd  # noqa: E402

from core import capital, fundamental, macro, newsflow, sentiment  # noqa: E402

SYM = "600519"
OUT: list = []


def say(msg=""):
    OUT.append(str(msg))
    print(str(msg)[:2000])


def head(title):
    say("\n" + "=" * 78)
    say("## " + title)
    say("=" * 78)


def show_df(df, n=3):
    if isinstance(df, pd.DataFrame):
        say("  shape=%s" % (df.shape,))
        say("  列=%s" % list(df.columns))
        if not df.empty:
            say(df.head(n).to_string()[:1500])
        if df.attrs:
            say("  attrs=%s" % {k: v for k, v in list(df.attrs.items())[:6]})
    else:
        say("  非 DataFrame: %r" % type(df))


def run(title, fn, *a, **kw):
    head(title)
    try:
        r = fn(*a, **kw)
        return r
    except Exception:
        say("[异常] " + traceback.format_exc(limit=3)[-800:])
        return None


def main():
    # 清空两级缓存，确保本次为**真实联网取数**（否则会命中上一次运行的缓存、看不到降级说明）
    try:
        from core.utils import cache_clear
        say("已清空缓存文件 %d 个，本次全部走真实网络" % cache_clear())
    except Exception as exc:
        say("清空缓存失败（继续）: %r" % exc)

    # ===================================================================== #
    # 基本面
    # ===================================================================== #
    fin = run("1. fundamental.fetch_financials('600519')", fundamental.fetch_financials, SYM)
    show_df(fin, 3)
    if isinstance(fin, pd.DataFrame) and not fin.empty:
        cols = [c for c in fundamental.FIN_COLUMNS if c in fin.columns]
        nonnull = {c: int(fin[c].notna().sum()) for c in cols}
        say("  各列非空期数：%s" % {k: v for k, v in nonnull.items() if v})
        say("  全为空的列：%s" % [k for k, v in nonnull.items() if v == 0])
        say("  最新一期：%s" % fin.iloc[-1][["报告期", "营业收入", "净利润", "ROE",
                                            "毛利率", "资产负债率", "每股收益",
                                            "营业利润", "研发费用"]].to_dict())
        say("  单季/环比：%s" % fin.iloc[-1][["单季营收", "单季净利", "营收环比", "净利环比",
                                            "环比增速"]].to_dict())

    val = run("2. fundamental.fetch_valuation('600519')", fundamental.fetch_valuation, SYM)
    if isinstance(val, dict):
        for k in ("pe_ttm", "pe_static", "pb", "ps", "pcf", "market_cap", "float_market_cap",
                  "total_shares", "float_shares", "dividend_yield", "ev_ebitda",
                  "pe_percentile", "pb_percentile", "pe_history_years", "percentile_method",
                  "source", "eps_ttm", "bps", "industry_pe_median", "turnover",
                  "volume_ratio"):
            say("  %-20s = %s" % (k, val.get(k)))

    peers = run("3. fundamental.fetch_industry_peers('600519')", fundamental.fetch_industry_peers,
                SYM)
    show_df(peers, 5)

    if isinstance(fin, pd.DataFrame) and not fin.empty:
        run("4. fundamental.growth_quality_score(frame)", fundamental.growth_quality_score, fin)
        gq = fundamental.growth_quality_score(fin)
        say("  score=%s growth=%.0f quality=%.0f cashflow=%.0f safety=%.0f"
            % (gq["score"], gq["growth"], gq["quality"], gq["cashflow"], gq["safety"]))
        for line in gq["detail"]:
            say("   - " + line)
        vp = fundamental.valuation_percentile(fundamental.fetch_price_series(SYM), fin)
        say("  valuation_percentile: %s" % {k: vp.get(k) for k in
                                            ("method", "current", "percentile", "samples",
                                             "years", "median", "note")})
        ev = fundamental.earnings_catalyst(fin, None)
        say("  earnings_catalyst: %s" % [(e["type"], e["direction"]) for e in ev])

    if isinstance(fin, pd.DataFrame) and not fin.empty:
        fs = run("5. fundamental.fundamental_factors('600519')", fundamental.fundamental_factors,
                 SYM, frame=fin, valuation=val, name="贵州茅台")
        if fs is not None:
            say("  因子数量=%d" % len(fs))
            say("  因子 id=%s" % sorted({f.id for f in fs}))
            for f in fs:
                say("   %-24s %-4s 强度 %.2f 可信度 %.2f | %s"
                    % (f.id, {1: "看涨", -1: "看跌", 0: "中性"}[f.direction], f.strength,
                       f.reliability, (f.detail or "")[:70]))
    say("\n  fundamental.last_notes():")
    for n in fundamental.last_notes():
        say("   - " + n)

    # ===================================================================== #
    # 资金面
    # ===================================================================== #
    flow = run("6. capital.fetch_fund_flow('600519')", capital.fetch_fund_flow, SYM)
    show_df(flow, 3)
    if isinstance(flow, pd.DataFrame) and not flow.empty:
        say("  末行：%s" % flow.iloc[-1].to_dict())

    minf = run("7. capital.fetch_minute_flow('600519')", capital.fetch_minute_flow, SYM)
    say("  %s" % minf)

    holders = run("8. capital.fetch_holders('600519')", capital.fetch_holders, SYM)
    show_df(holders, 4)

    margin = run("9. capital.fetch_margin('600519')", capital.fetch_margin, SYM)
    show_df(margin, 3)

    lhb = run("10. capital.fetch_lhb('600519')", capital.fetch_lhb, SYM)
    show_df(lhb, 3)

    north = run("11. capital.fetch_northbound('600519')", capital.fetch_northbound, SYM)
    show_df(north, 3)

    t10 = run("12. capital.fetch_top10_float_holders('600519')",
              capital.fetch_top10_float_holders, SYM)
    show_df(t10, 3)

    cd = run("13. capital.control_degree('600519')", capital.control_degree, SYM)
    say("  %s" % cd)
    prof = run("14. capital.main_force_profile(flow)", capital.main_force_profile, flow)
    say("  %s" % prof)

    snap = capital.fetch_snapshot(SYM)
    say("  快照：%s" % {k: snap.get(k) for k in ("name", "price", "turnover", "volume_ratio",
                                                 "market_cap", "pe_ttm", "pb")})
    fs = run("15. capital.capital_factors('600519')", capital.capital_factors, SYM,
             **{"flow": flow, "holders": holders, "margin": margin, "lhb": lhb,
                "north": north, "snapshot": snap, "top10": t10})
    if fs is not None:
        say("  因子数量=%d" % len(fs))
        say("  因子 id=%s" % sorted({f.id for f in fs}))
        for f in fs:
            say("   %-26s %-4s 强度 %.2f 可信度 %.2f | %s"
                % (f.id, {1: "看涨", -1: "看跌", 0: "中性"}[f.direction], f.strength,
                   f.reliability, (f.detail or "")[:70]))
    say("\n  capital.last_notes():")
    for n in capital.last_notes():
        say("   - " + n)

    # ===================================================================== #
    # 情绪面
    # ===================================================================== #
    mk = run("16. sentiment.fetch_market_sentiment()", sentiment.fetch_market_sentiment)
    if isinstance(mk, dict) and isinstance(margin, pd.DataFrame) and not margin.empty:
        ms = sentiment.margin_sentiment(margin)
        mk = dict(mk, margin=ms)
        say("  已注入两融：current=%.2f 亿，%d 点历史，5 日 %+.2f%%"
            % (ms["current"] / 1e8, len(ms["history"]), ms["change_5d_pct"]))
    say("  %s" % {k: v for k, v in (mk or {}).items() if k not in ("boards_hot", "margin")})
    lu = run("17. sentiment.limit_up_stats()", sentiment.limit_up_stats)
    say("  %s" % lu)
    bd = run("18. sentiment.market_breadth()", sentiment.market_breadth)
    say("  %s" % {k: v for k, v in (bd or {}).items()})
    sec = run("19. sentiment.fetch_sector_heat('600519')", sentiment.fetch_sector_heat, SYM)
    if isinstance(sec, dict):
        say("  is_hot_theme=%s theme_names=%s sector_rank_pct=%s"
            % (sec.get("is_hot_theme"), sec.get("theme_names"), sec.get("sector_rank_pct")))
        say("  hottest=%s leader=%s" % (sec.get("hottest"), sec.get("leader_stock")))
        say("  boards=%s" % (sec.get("boards") or [])[:6])
        say("  detail=%s" % sec.get("detail"))
    ret = run("20. sentiment.fetch_retail_sentiment('600519')",
              sentiment.fetch_retail_sentiment, SYM)
    say("  %s" % ret)
    frame = fundamental.fetch_price_series(SYM)
    ohlcv = None
    if len(frame):
        # 用新浪日线收盘价 + 成交量（股）/ 流通股本 还原真实换手率，供情绪类因子使用
        px = fundamental.fetch_price_series(SYM)
        ohlcv = pd.DataFrame({"date": px.index, "close": px.values})
        try:
            import json as _json
            from core.utils import http_get as _hg
            raw = _hg("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                      "CN_MarketData.getKLineData?symbol=sh600519&scale=240&ma=no&datalen=1300",
                      headers={"Referer": "https://finance.sina.com.cn"})
            vols = {str(r["day"]): float(r["volume"]) for r in _json.loads(raw)}
            fshares = float((snap or {}).get("float_shares") or 1.25e9)
            ohlcv["volume"] = [vols.get(str(d.date()), float("nan")) for d in px.index]
            ohlcv["turnover"] = ohlcv["volume"] / fshares * 100.0
            say("  真实换手率序列：均值 %.3f%%（最新 %.3f%%）"
                % (ohlcv["turnover"].mean(), ohlcv["turnover"].iloc[-1]))
        except Exception as exc:
            say("  换手率构造失败，用 0.3%% 常数近似：%r" % exc)
            ohlcv["turnover"] = 0.3
    fg = run("21. sentiment.fear_greed_index('600519')", sentiment.fear_greed_index, SYM,
             frame=ohlcv, market=mk)
    say("  %s" % fg)
    fs = run("22. sentiment.sentiment_factors('600519')", sentiment.sentiment_factors, SYM,
             market=mk, sector=sec, retail=ret, frame=ohlcv, snapshot=snap)
    if fs is not None:
        say("  因子数量=%d" % len(fs))
        say("  因子 id=%s" % sorted({f.id for f in fs}))
        for f in fs:
            say("   %-30s %-4s 强度 %.2f 可信度 %.2f | %s"
                % (f.id, {1: "看涨", -1: "看跌", 0: "中性"}[f.direction], f.strength,
                   f.reliability, (f.detail or "")[:70]))
    say("\n  sentiment.last_notes():")
    for n in sentiment.last_notes():
        say("   - " + n)

    # ===================================================================== #
    # 宏观
    # ===================================================================== #
    mc = run("23. macro.fetch_macro()", macro.fetch_macro)
    if isinstance(mc, dict):
        for k in ("gdp_yoy", "cpi_yoy", "ppi_yoy", "pmi", "lpr_1y", "lpr_5y", "shibor_3m",
                  "m2_yoy", "social_financing_yoy", "usd_cny", "vix"):
            say("  %-22s = %-8s 来源=%s" % (k, mc.get(k), (mc.get("source") or {}).get(k)))
        say("  fallback_fields=%s" % mc.get("fallback_fields"))
        say("  detail=%s" % mc.get("detail"))
    com = run("24. macro.fetch_commodity_prices()", macro.fetch_commodity_prices)
    if isinstance(com, dict):
        for k, v in com.items():
            if isinstance(v, dict):
                say("  %-6s price=%-12s pct=%-8s 来源=%s"
                    % (k, v.get("price"), v.get("pct_chg"), v.get("source")))
        say("  missing=%s detail=%s" % (com.get("missing"), com.get("detail")))
    rg = run("25. macro.market_regime(None)  [真实联网取指数]", macro.market_regime, None)
    say("  %s" % rg)
    pol = run("26. macro.fetch_industry_policy('600519')", macro.fetch_industry_policy, SYM,
              "贵州茅台")
    say("  %s" % pol)
    cyc = run("27. macro.industry_cycle('600519')", macro.industry_cycle, SYM, "贵州茅台")
    say("  %s" % cyc)
    idx = macro.fetch_index_daily()
    show_df(idx, 2)
    fs = run("28. macro.macro_factors('600519')", macro.macro_factors, SYM, macro=mc,
             policy=pol, commodity=com, index_data=idx)
    if fs is not None:
        say("  因子数量=%d" % len(fs))
        say("  因子 id=%s" % sorted({f.id for f in fs}))
        for f in fs:
            say("   %-20s %-4s 强度 %.2f 可信度 %.2f | %s"
                % (f.id, {1: "看涨", -1: "看跌", 0: "中性"}[f.direction], f.strength,
                   f.reliability, (f.detail or "")[:70]))
    say("\n  macro.last_notes():")
    for n in macro.last_notes():
        say("   - " + n)

    # ===================================================================== #
    # 消息面
    # ===================================================================== #
    anns = run("29. newsflow.fetch_announcements('600519')", newsflow.fetch_announcements, SYM)
    say("  条数=%d" % len(anns or []))
    for a in (anns or [])[:5]:
        say("   %s | %s | %s | %s" % (a.get("date"), a.get("type"), a.get("source"),
                                      a.get("title")[:60]))
    if anns:
        say("  详情 URL 示例：%s" % anns[0].get("url"))
    news = run("30. newsflow.fetch_news('600519', '贵州茅台')", newsflow.fetch_news, SYM,
               "贵州茅台")
    say("  条数=%d" % len(news or []))
    for a in (news or [])[:5]:
        say("   %s | %s | %s" % (a.get("date"), a.get("source"), a.get("title")[:60]))
    items = (news or []) + (anns or [])
    cls = run("31. newsflow.classify_news(items)", newsflow.classify_news, items[:40])
    say("  条数=%d" % len(cls or []))
    for c in (cls or [])[:10]:
        say("   [%-4s] %-4s %.2f | %s" % (c["tag"],
                                          {1: "利好", -1: "利空", 0: "中性"}[c["direction"]],
                                          c["strength"], c["title"][:56]))
    bs = run("32. newsflow.blackswan_scan(items, frame)", newsflow.blackswan_scan, items[:40],
             fin)
    say("  severity=%s flags=%s" % ((bs or {}).get("severity"),
                                    [(f["type"], f["severity"]) for f in (bs or {}).get("flags", [])]))
    say("  detail=%s" % (bs or {}).get("detail"))
    poln = run("33. newsflow.policy_news_scan(items)", newsflow.policy_news_scan, items[:40])
    say("  %s" % {k: v for k, v in (poln or {}).items() if k != "items"})
    fs = run("34. newsflow.event_factors('600519')", newsflow.event_factors, SYM,
             news=news, announcements=anns, name="贵州茅台")
    if fs is not None:
        say("  因子数量=%d" % len(fs))
        say("  因子 id=%s" % sorted({f.id for f in fs}))
        for f in fs:
            say("   %-24s %-4s 强度 %.2f 可信度 %.2f 周期=%s | %s"
                % (f.id, {1: "看涨", -1: "看跌", 0: "中性"}[f.direction], f.strength,
                   f.reliability, "/".join(f.horizon) if f.horizon else "全部",
                   (f.detail or "")[:60]))
    say("\n  newsflow.last_notes():")
    for n in newsflow.last_notes():
        say("   - " + n)

    # ===================================================================== #
    head("汇总：各模块因子数量与 id 清单（供权重表核查）")
    bundles = []
    try:
        bundles.append(("fundamental", fundamental.fundamental_factors(
            SYM, frame=fin, valuation=val, name="贵州茅台")))
    except Exception as exc:
        say("fundamental 汇总失败: %r" % exc)
    try:
        bundles.append(("capital", capital.capital_factors(
            SYM, flow=flow, holders=holders, margin=margin, lhb=lhb, north=north,
            snapshot=snap, top10=t10, frame=ohlcv)))
    except Exception as exc:
        say("capital 汇总失败: %r" % exc)
    try:
        bundles.append(("sentiment", sentiment.sentiment_factors(
            SYM, market=mk, sector=sec, retail=ret, frame=ohlcv, snapshot=snap)))
    except Exception as exc:
        say("sentiment 汇总失败: %r" % exc)
    try:
        bundles.append(("macro", macro.macro_factors(
            SYM, macro=mc, policy=pol, commodity=com, index_data=idx)))
    except Exception as exc:
        say("macro 汇总失败: %r" % exc)
    try:
        bundles.append(("newsflow", newsflow.event_factors(
            SYM, news=news, announcements=anns, name="贵州茅台")))
    except Exception as exc:
        say("newsflow 汇总失败: %r" % exc)
    total = 0
    for name, fs in bundles:
        ids = sorted({f.id for f in fs})
        total += len(fs)
        say("%-12s 因子数=%-3d id(%d)=%s" % (name, len(fs), len(ids), ids))
        contrarian = sorted(set(ids) & set(__import__("core.config", fromlist=["x"])
                                           .CONTRARIAN_FACTOR_IDS))
        if contrarian:
            say("%-12s 其中反向指标：%s" % ("", contrarian))
    say("五模块合计因子数 = %d" % total)

    with open(r"E:\deepseek desktop\个股预测程序\_probe\REAL_DATA_REPORT.txt", "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print("\n>>> 已写入 _probe/REAL_DATA_REPORT.txt")


if __name__ == "__main__":
    main()

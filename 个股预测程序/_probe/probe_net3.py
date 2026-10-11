# -*- coding: utf-8 -*-
"""第三轮探测：资金流可用通道、炸板池、利润表补充、VIX、股吧人气变体。"""

from __future__ import annotations

import json
import os
import sys
import traceback
import urllib.parse

sys.path.insert(0, r"E:\deepseek desktop\个股预测程序")

import requests  # noqa: E402

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept": "*/*",
                  "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"})
OUT: list[str] = []


def say(m):
    OUT.append(str(m))
    print(str(m)[:2000])


def sec(t):
    OUT.append("\n\n## " + t)
    print("\n=== " + t + " ===")


def probe(t, fn):
    sec(t)
    try:
        fn()
        say("[OK] " + t)
    except Exception:
        say("[FAIL] " + t)
        say(traceback.format_exc(limit=3)[-900:])


def get(url, **kw):
    kw.setdefault("timeout", 12)
    return S.get(url, **kw)


FF = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"


def a_fflow_variants():
    base = ("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
            "?lmt=20&klt=101&secid=1.600519&fields1=f1,f2,f3,f7&fields2=" + FF)
    variants = [
        ("https无ut", base, {}),
        ("http无ut", base.replace("https://", "http://"), {}),
        ("加Referer", base, {"Referer": "https://quote.eastmoney.com/"}),
        ("加ut", base + "&ut=b2884a393a59ad64002292a3e90d46a5", {}),
        ("push2主机", base.replace("push2his", "push2"), {}),
        ("push2delay主机", base.replace("push2his", "push2delay"), {}),
        ("cb参数", base + "&cb=jsonp", {}),
        ("fields1全", base.replace("fields1=f1,f2,f3,f7", "fields1=f1,f2,f3,f4,f5,f6,f7,f8"), {}),
    ]
    for name, url, hdr in variants:
        try:
            r = get(url, headers=hdr or None)
            txt = r.text
            say("[%s] status=%s len=%d 前160=%s" % (name, r.status_code, len(txt), txt[:160]))
        except Exception as exc:
            say("[%s] 失败 %r" % (name, exc))


def b_ak_fund_flow():
    import akshare as ak
    try:
        df = ak.stock_individual_fund_flow(stock="600519", market="sh")
        say("ak.stock_individual_fund_flow shape=%s 列=%s" % (df.shape, list(df.columns)))
        say(df.tail(3).to_string()[:1200])
    except Exception as exc:
        say("ak.stock_individual_fund_flow 失败: %r" % exc)
    try:
        df = ak.stock_individual_fund_flow_rank(indicator="今日")
        say("rank shape=%s" % (df.shape,))
    except Exception as exc:
        say("rank 失败: %r" % exc)


def c_sina_moneyflow():
    urls = [
        ("ssl_qsfx_zjlrqs",
         "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         "MoneyFlow.ssl_qsfx_zjlrqs?page=1&num=60&sort=opendate&asc=0&daima=sh600519"),
        ("ssl_bkzj_zjlrqs",
         "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         "MoneyFlow.ssl_bkzj_zjlrqs?page=1&num=60&sort=opendate&asc=0&daima=sh600519"),
        ("ssl_qsfx_lscjfb",
         "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         "MoneyFlow.ssl_qsfx_lscjfb?page=1&num=20&sort=opendate&asc=0&daima=sh600519"),
    ]
    for name, url in urls:
        try:
            r = get(url, headers={"Referer": "https://finance.sina.com.cn"})
            r.encoding = "gbk"
            say("[%s] status=%s 前 500: %s" % (name, r.status_code, r.text[:500]))
        except Exception as exc:
            say("[%s] 失败 %r" % (name, exc))


def d_zbpool():
    import datetime as dt
    for path in ["getTopicZBPool", "getTopicFBPool"]:
        for back in range(0, 6):
            day = (dt.date.today() - dt.timedelta(days=back)).strftime("%Y%m%d")
            url = ("https://push2ex.eastmoney.com/%s?ut=7eea3edcaed734bea9cbfc24409ed989"
                   "&dpt=wz.ztzt&Pageindex=0&pagesize=50&sort=fbt%%3Aasc&date=%s" % (path, day))
            try:
                r = get(url, headers={"Referer": "https://quote.eastmoney.com/"})
                if r.status_code == 200 and r.text.strip().startswith("{"):
                    d = r.json()
                    data = d.get("data")
                    if data:
                        pool = data.get("pool") or []
                        say("[%s %s] tc=%s pool=%d" % (path, day, data.get("tc"), len(pool)))
                        if pool:
                            say("   字段: %s" % list(pool[0].keys()))
                            say("   样例: %s" % json.dumps(pool[0], ensure_ascii=False)[:500])
                        break
                    else:
                        say("[%s %s] data 空" % (path, day))
                else:
                    say("[%s %s] status=%s 前80=%s" % (path, day, r.status_code, r.text[:80]))
            except Exception as exc:
                say("[%s %s] 失败 %r" % (path, day, exc))


def e_profit_sheet():
    import akshare as ak
    import inspect
    for fn_name in ["stock_profit_sheet_by_report_em", "stock_financial_benefit_ths",
                    "stock_profit_sheet_by_yearly_em"]:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            say("%s 不存在" % fn_name)
            continue
        try:
            say("%s 签名: %s" % (fn_name, inspect.signature(fn)))
        except Exception:
            pass
    try:
        df = ak.stock_profit_sheet_by_report_em(symbol="SH600519")
        say("shape=%s" % (df.shape,))
        cols = [c for c in df.columns if any(k in c.upper() for k in
                ("REPORT_DATE", "OPERATE_PROFIT", "TOTAL_PROFIT", "RESEARCH", "OPERATE_INCOME",
                 "OPERATE_COST", "PARENT_NETPROFIT"))]
        say("关注列: %s" % cols)
        if cols:
            say(df[cols].head(4).to_string()[:1500])
    except Exception as exc:
        say("stock_profit_sheet_by_report_em 失败: %r" % exc)


def f_vix():
    os.environ.setdefault("YF_CACHE_DIR", r"E:\deepseek desktop\个股预测程序\.cache\yf")
    try:
        import yfinance as yf
        try:
            yf.set_tz_cache_location(r"E:\deepseek desktop\个股预测程序\.cache\yf")
            say("已设置 yf tz cache")
        except Exception as exc:
            say("set_tz_cache_location 失败: %r" % exc)
        for sym in ["^VIX", "VIX"]:
            try:
                t = yf.Ticker(sym)
                h = t.history(period="5d")
                say("[%s] history shape=%s" % (sym, h.shape))
                if len(h):
                    say(h.tail(3).to_string()[:400])
            except Exception as exc:
                say("[%s] 失败 %r" % (sym, exc))
    except Exception as exc:
        say("yfinance 导入失败: %r" % exc)


def g_guba_variants():
    bodies = [
        {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38", "marketType": "",
         "srcSecurityCode": "600519.SH", "pageNo": 1, "pageSize": 100},
        {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38", "marketType": "",
         "srcSecurityCode": ["600519"], "pageNo": 1, "pageSize": 100},
        {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38", "marketType": "1",
         "srcSecurityCode": "600519", "pageNo": 1, "pageSize": 100, "deviceType": "web"},
        {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38", "marketType": "",
         "srcSecurityCode": "", "pageNo": 1, "pageSize": 100},
    ]
    for b in bodies:
        try:
            r = S.post("https://emappdata.eastmoney.com/stockrank/getCurrentList", json=b,
                       headers={"Content-Type": "application/json",
                                "Referer": "https://guba.eastmoney.com/rank/"}, timeout=12)
            say("%s -> %s" % (json.dumps(b, ensure_ascii=False)[-90:], r.text[:400]))
        except Exception as exc:
            say("失败 %r" % exc)
    # guba 人气榜页面里的 JSON
    try:
        r = get("https://guba.eastmoney.com/rank/", headers={"Referer": "https://guba.eastmoney.com/"})
        txt = r.text
        say("rank 页长度=%d，含 'rank' 关键字位置: %s" % (len(txt), txt.find("getCurrentList")))
        i = txt.find("var ")
        say("片段: %s" % txt[i:i + 400].replace("\n", " "))
    except Exception as exc:
        say("rank 页失败 %r" % exc)
    # 东财搜索（简化：用 params 传参，不手工 quote）
    try:
        param = {"uid": "", "keyword": "贵州茅台", "type": ["cmsArticleWebOld"],
                 "client": "web", "clientType": "web", "clientVersion": "curr",
                 "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "time",
                                                "pageIndex": 1, "pageSize": 5,
                                                "preTag": "<em>", "postTag": "</em>"}}}
        r = get("https://search-api-web.eastmoney.com/search/jsonp",
                params={"cb": "cb", "param": json.dumps(param, ensure_ascii=False)})
        say("search(params 方式) 前 700: %s" % r.text[:700])
    except Exception as exc:
        say("search 失败 %r" % exc)


def h_quote_extra():
    """确认腾讯 f58/f64 含义与北向最新行数。"""
    r = get("https://qt.gtimg.cn/q=sh600519,sz000001,sh601318")
    r.encoding = "gbk"
    for line in r.text.strip().split("\n"):
        if "=" not in line:
            continue
        p = line.split('="', 1)[1].rsplit('"', 1)[0].split("~")
        say("%s %s 价=%s PE=%s PB=%s 流通市值=%s 总市值=%s [58]=%s [62]=%s [63]=%s [64]=%s "
            "[65]=%s [66]=%s [67]=%s [68]=%s [69]=%s [70]=%s [71]=%s [74]=%s [75]=%s [79]=%s"
            % (p[2], p[1], p[3], p[39], p[46], p[44], p[45], p[58], p[62], p[63], p[64],
               p[65], p[66], p[67], p[68], p[69], p[70], p[71], p[74], p[75], p[79]))


def i_north_rows():
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
           "reportName=RPT_MUTUAL_HOLDSTOCKNORTH_STA&columns=ALL"
           "&filter=(SECURITY_CODE%3D%22600519%22)&pageSize=60")
    d = get(url).json()
    res = (d.get("result") or {}).get("data") or []
    say("行数=%d" % len(res))
    if res:
        say("字段: %s" % list(res[0].keys()))
        say("首行: %s" % json.dumps(res[0], ensure_ascii=False)[:900])
        say("末行 TRADE_DATE=%s" % res[-1].get("TRADE_DATE"))


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [
        ("A.东财资金流通道变体", a_fflow_variants),
        ("B.akshare 个股资金流", b_ak_fund_flow),
        ("C.新浪资金流备选", c_sina_moneyflow),
        ("D.炸板池/涨停池端点", d_zbpool),
        ("E.akshare 利润表（营业利润/研发）", e_profit_sheet),
        ("F.VIX via yfinance(tz cache 修复)", f_vix),
        ("G.股吧人气/搜索变体", g_guba_variants),
        ("H.腾讯快照扩展字段确认", h_quote_extra),
        ("I.北向报表全量行", i_north_rows),
    ]
    for t, fn in tests:
        if only and only not in t:
            continue
        probe(t, fn)
    os.makedirs(r"E:\deepseek desktop\个股预测程序\_probe", exist_ok=True)
    with open(r"E:\deepseek desktop\个股预测程序\_probe\PROBE_RESULTS3.md", "w",
              encoding="utf-8") as fh:
        fh.write("# 第三轮接口探测（真实网络实测）\n" + "\n".join(OUT))
    print("\n>>> 已写入 _probe/PROBE_RESULTS3.md")

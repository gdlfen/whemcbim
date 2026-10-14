# -*- coding: utf-8 -*-
"""联网接口探测脚本（一次性，结果写入 _probe/PROBE_RESULTS.md）。

目的：在写正式模块前，确认各数据源的真实字段顺序/含义，避免臆造字段。
本脚本不属于交付物，可随时删除。
"""

from __future__ import annotations

import json
import sys
import traceback

sys.path.insert(0, r"E:\deepseek desktop\个股预测程序")

import requests  # noqa: E402

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept": "*/*",
                  "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"})

OUT: list[str] = []


def sec(title):
    OUT.append("\n\n## " + title)
    print("\n=== " + title + " ===")


def say(msg):
    OUT.append(str(msg))
    print(str(msg)[:3000])


def probe(title, fn):
    sec(title)
    try:
        fn()
        say("[OK] " + title)
    except Exception:
        say("[FAIL] " + title)
        say(traceback.format_exc(limit=4)[-1500:])


def get(url, **kw):
    kw.setdefault("timeout", 15)
    return S.get(url, **kw)


def js(url, **kw):
    r = get(url, **kw)
    return r.json()


# ---------------------------------------------------------------- 1. 腾讯实时
def t_tencent():
    r = get("https://qt.gtimg.cn/q=sh600519")
    r.encoding = "gbk"
    txt = r.text
    say("RAW(前 400 字): " + txt[:400])
    body = txt.split('="', 1)[1].rsplit('"', 1)[0]
    parts = body.split("~")
    say("字段数: %d" % len(parts))
    for i, v in enumerate(parts):
        say("  [%d] = %s" % (i, v))


# ------------------------------------------------------- 2. 东财日频资金流
def t_fflow_day():
    url = ("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
           "?secid=1.600519&fields1=f1,f2,f3,f7"
           "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
           "&klt=101&lmt=5")
    d = js(url)
    say("顶层键: %s" % list(d.keys()))
    data = d.get("data") or {}
    say("data 键: %s" % list(data.keys()))
    klines = data.get("klines") or []
    say("klines 条数: %d" % len(klines))
    for row in klines:
        cells = row.split(",")
        say("  字段数=%d" % len(cells))
        for i, c in enumerate(cells):
            say("    [%d] = %s" % (i, c))
        break
    say("最后一行: %s" % (klines[-1] if klines else "无"))


# ------------------------------------------------------- 3. 东财分时资金流
def t_fflow_min():
    url = ("https://push2his.eastmoney.com/api/qt/stock/fflow/kline/get"
           "?secid=1.600519&fields1=f1,f2,f3,f7"
           "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
           "&klt=1&lmt=0")
    d = js(url)
    data = d.get("data") or {}
    klines = data.get("klines") or []
    say("klines 条数: %d" % len(klines))
    for row in klines[:2]:
        cells = row.split(",")
        say("  字段数=%d : %s" % (len(cells), cells))
    if klines:
        say("最后一行: %s" % klines[-1])


# ------------------------------------------------- 4. 东财所属板块（slist）
def t_slist():
    url = ("https://push2.eastmoney.com/api/qt/slist/get?secid=1.600519&spt=3"
           "&fields=f12,f13,f14,f3,f62,f184&po=1&pn=1&pz=30&fid=f62")
    d = js(url)
    say("顶层: %s" % json.dumps(d, ensure_ascii=False)[:800])
    diff = ((d.get("data") or {}).get("diff")) or []
    say("板块数: %d" % len(diff))
    for it in diff[:12]:
        say("  %s" % json.dumps(it, ensure_ascii=False))


def t_slist_plain():
    url = ("https://push2.eastmoney.com/api/qt/slist/get?secid=1.600519&spt=3"
           "&fields=f12,f13,f14&pz=30")
    d = js(url)
    say(json.dumps(d, ensure_ascii=False)[:1200])


# ------------------------------------------------- 5. 东财板块成分（clist）
def t_clist():
    url = ("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10&po=1&np=1"
           "&fltt=2&invt=2&fid=f3&fs=b:BK0475&fields=f12,f14,f9,f23,f3,f20")
    d = js(url)
    diff = ((d.get("data") or {}).get("diff")) or []
    say("只数: %d" % len(diff))
    for it in diff[:10]:
        say("  %s" % json.dumps(it, ensure_ascii=False))


# --------------------------------------- 6. 东财 datacenter 财务（备选源）
def t_lixo():
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
           "reportName=RPT_LICO_FN_CPD&columns=ALL&filter=(SECUCODE%3D%22600519.SH%22)"
           "&pageSize=3&sortColumns=REPORTDATE&sortTypes=-1")
    d = js(url)
    say("success=%s message=%s" % (d.get("success"), d.get("message")))
    res = (d.get("result") or {}).get("data") or []
    say("行数: %d" % len(res))
    if res:
        say("字段: %s" % list(res[0].keys()))
        say(json.dumps(res[0], ensure_ascii=False)[:2500])


# ------------------------------------------------ 7. akshare 财务摘要
def t_ak_fin():
    import akshare as ak
    df = ak.stock_financial_abstract(symbol="600519")
    say("shape=%s" % (df.shape,))
    say("列前 12: %s" % list(df.columns)[:12])
    say("列后 6: %s" % list(df.columns)[-6:])
    say(df.head(6).to_string()[:2500])
    say("指标去重前 40: %s" % list(dict.fromkeys(df.iloc[:, 1].tolist()))[:40])


# ------------------------------------------------ 8. 股东户数（akshare）
def t_ak_gdhs():
    import akshare as ak
    df = ak.stock_zh_a_gdhs_detail_em(symbol="600519")
    say("shape=%s" % (df.shape,))
    say("列: %s" % list(df.columns))
    say(df.head(3).to_string()[:1500])


def t_em_holdernum():
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
           "reportName=RPT_HOLDERNUMLATEST&columns=ALL"
           "&filter=(SECURITY_CODE%3D%22600519%22)&pageSize=5&sortColumns=END_DATE&sortTypes=-1")
    d = js(url)
    res = (d.get("result") or {}).get("data") or []
    say("行数: %d" % len(res))
    if res:
        say("字段: %s" % list(res[0].keys()))
        say(json.dumps(res[0], ensure_ascii=False)[:1200])


# ------------------------------------------------ 9. 两融明细
def t_margin():
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
           "reportName=RPTA_WEB_RZRQ_GGMX&columns=ALL"
           "&filter=(SCODE%3D%22600519%22)&pageSize=3&sortColumns=DATE&sortTypes=-1")
    d = js(url)
    res = (d.get("result") or {}).get("data") or []
    say("行数: %d" % len(res))
    if res:
        say("字段: %s" % list(res[0].keys()))
        say(json.dumps(res[0], ensure_ascii=False)[:1500])


# ------------------------------------------------ 10. 龙虎榜
def t_lhb():
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
           "reportName=RPT_DAILYBILLBOARD_DETAILSNEW&columns=ALL"
           "&filter=(SECURITY_CODE%3D%22600519%22)&pageSize=3&sortColumns=TRADE_DATE&sortTypes=-1")
    d = js(url)
    res = (d.get("result") or {}).get("data") or []
    say("行数: %d" % len(res))
    if res:
        say("字段: %s" % list(res[0].keys()))
        say(json.dumps(res[0], ensure_ascii=False)[:1800])


def t_lhb_org():
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
           "reportName=RPT_ORGANIZATION_TRADE_DETAILS&columns=ALL"
           "&filter=(SECURITY_CODE%3D%22600519%22)&pageSize=3&sortColumns=TRADE_DATE&sortTypes=-1")
    d = js(url)
    res = (d.get("result") or {}).get("data") or []
    say("行数: %d" % len(res))
    if res:
        say("字段: %s" % list(res[0].keys()))
        say(json.dumps(res[0], ensure_ascii=False)[:1500])


# ------------------------------------------------ 11. 北向持股
def t_north():
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
           "reportName=RPT_MUTUAL_HOLDSTOCKNORTH_STA&columns=ALL"
           "&filter=(SECURITY_CODE%3D%22600519%22)&pageSize=3&sortColumns=HOLD_DATE&sortTypes=-1")
    d = js(url)
    say("success=%s msg=%s" % (d.get("success"), d.get("message")))
    res = (d.get("result") or {}).get("data") or []
    say("行数: %d" % len(res))
    if res:
        say("字段: %s" % list(res[0].keys()))
        say(json.dumps(res[0], ensure_ascii=False)[:1200])


def t_ak_hsgt():
    import akshare as ak
    df = ak.stock_hsgt_individual_em(stock="600519")
    say("shape=%s" % (df.shape,))
    say("列: %s" % list(df.columns))
    say(df.head(3).to_string()[:1200])


# ------------------------------------------------ 12. 十大流通股东
def t_top10():
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
           "reportName=RPT_F10_EH_FREEHOLDERS&columns=ALL"
           "&filter=(SECUCODE%3D%22600519.SH%22)&pageSize=12&sortColumns=END_DATE&sortTypes=-1")
    d = js(url)
    res = (d.get("result") or {}).get("data") or []
    say("行数: %d" % len(res))
    if res:
        say("字段: %s" % list(res[0].keys()))
        say(json.dumps(res[0], ensure_ascii=False)[:1200])
        say("第二行: %s" % json.dumps(res[1], ensure_ascii=False)[:800])


# ------------------------------------------------ 13. 涨停板池/炸板池
def t_ztpool():
    import datetime as dt
    # 用过去 30 天里最近一个有数据的交易日
    for back in range(0, 12):
        day = (dt.date.today() - dt.timedelta(days=back)).strftime("%Y%m%d")
        url = ("https://push2ex.eastmoney.com/getTopicZTPool?ut=7eea3edcaed734bea9cbfc24409ed989"
               "&dpt=wz.ztzt&Pageindex=0&pagesize=5&sort=fbt%3Aasc&date=" + day)
        r = get(url, headers={"Referer": "https://quote.eastmoney.com/"})
        try:
            d = r.json()
        except Exception:
            say("%s 非 JSON: %s" % (day, r.text[:200]))
            continue
        data = d.get("data")
        if not data:
            say("%s data 为空 (rc=%s)" % (day, d.get("rc")))
            continue
        pool = data.get("pool") or []
        say("日期 %s 涨停只数=%s" % (day, data.get("tc")))
        if pool:
            say("字段: %s" % list(pool[0].keys()))
            say(json.dumps(pool[0], ensure_ascii=False)[:900])
        break


def t_fbpool():
    import datetime as dt
    for back in range(0, 12):
        day = (dt.date.today() - dt.timedelta(days=back)).strftime("%Y%m%d")
        url = ("https://push2ex.eastmoney.com/getTopicFBPool?ut=7eea3edcaed734bea9cbfc24409ed989"
               "&dpt=wz.ztzt&Pageindex=0&pagesize=5&sort=fbt%3Aasc&date=" + day)
        r = get(url, headers={"Referer": "https://quote.eastmoney.com/"})
        try:
            d = r.json()
        except Exception:
            continue
        data = d.get("data")
        if not data:
            continue
        pool = data.get("pool") or []
        say("日期 %s 炸板只数=%s" % (day, data.get("tc")))
        if pool:
            say("字段: %s" % list(pool[0].keys()))
            say(json.dumps(pool[0], ensure_ascii=False)[:700])
        break


# ------------------------------------------------ 14. 指数快照 ulist
def t_ulist():
    url = ("https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2"
           "&secids=1.000001,0.399001,0.399006&fields=f2,f3,f4,f12,f14")
    d = js(url)
    say(json.dumps(d, ensure_ascii=False)[:1000])


# ------------------------------------------------ 15. 股吧人气排名
def t_guba_rank():
    url = "https://emappdata.eastmoney.com/stockrank/getCurrentList"
    r = S.post(url, json={"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38",
                          "marketType": "", "pageNo": 1, "pageSize": 100},
               headers={"Content-Type": "application/json"}, timeout=15)
    say("status=%s" % r.status_code)
    say(r.text[:800])


def t_guba_rank2():
    url = "https://emappdata.eastmoney.com/stockrank/getCurrentList"
    r = S.post(url, json={"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38",
                          "marketType": "", "pageNo": 1, "pageSize": 6000},
               headers={"Content-Type": "application/json"}, timeout=20)
    d = r.json()
    say("顶层: %s" % json.dumps(d, ensure_ascii=False)[:300])
    data = d.get("data") or []
    say("条数: %d" % len(data))
    if data:
        say("字段: %s" % list(data[0].keys()))
        hit = [x for x in data if str(x.get("sc", "")).endswith("600519") or
               str(x.get("code", "")) == "600519"]
        say("600519 条目: %s" % json.dumps(hit[:2], ensure_ascii=False))


# ------------------------------------------------ 16. 东财搜索
def t_search():
    import urllib.parse
    param = {
        "uid": "", "keyword": "600519", "type": ["cmsArticleWebOld"],
        "client": "web", "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default",
                                       "pageIndex": 1, "pageSize": 3,
                                       "preTag": "<em>", "postTag": "</em>"}},
    }
    url = ("https://search-api-web.eastmoney.com/search/jsonp?cb=cb&param="
           + urllib.parse.quote(json.dumps(param, ensure_ascii=False)))
    r = get(url)
    say("RAW 前 1200: " + r.text[:1200])


# ------------------------------------------------ 17. 新浪行情（汇率/期货/VIX）
def t_sina_hq():
    for codes in ["fx_susdcny", "hf_CL,hf_CAD,hf_GC", "gb_$vix", "sh000001"]:
        try:
            r = get("https://hq.sinajs.cn/list=" + codes,
                    headers={"Referer": "https://finance.sina.com.cn"})
            r.encoding = "gbk"
            say("[%s] -> %s" % (codes, r.text.strip()[:400]))
        except Exception as exc:
            say("[%s] 失败 %s" % (codes, exc))


def t_yf_vix():
    import yfinance as yf
    df = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=False)
    say("shape=%s" % (df.shape,))
    say(df.tail(3).to_string()[:800])


# ------------------------------------------------ 18. 公告接口
def t_ann():
    url = ("https://np-anotice-stock.eastmoney.com/api/security/ann?sr=-1&page_size=5"
           "&page_index=1&ann_type=A&client_source=web&stock_list=600519")
    d = js(url)
    say("success=%s" % d.get("success"))
    data = (d.get("data") or {}).get("list") or []
    say("条数: %d" % len(data))
    if data:
        say("字段: %s" % list(data[0].keys()))
        say(json.dumps(data[0], ensure_ascii=False)[:1200])


# ------------------------------------------------ 19. akshare 指数日线
def t_ak_index():
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol="sh000001")
    say("shape=%s 列=%s" % (df.shape, list(df.columns)))
    say(df.tail(3).to_string())


# ------------------------------------------------ 20. akshare 宏观
def t_ak_macro():
    import akshare as ak
    for name, fn in [
        ("cpi_monthly", lambda: ak.macro_china_cpi_monthly()),
        ("ppi_monthly", lambda: ak.macro_china_ppi()),
        ("pmi", lambda: ak.macro_china_pmi()),
        ("gdp", lambda: ak.macro_china_gdp()),
        ("lpr", lambda: ak.macro_china_lpr()),
        ("money_supply", lambda: ak.macro_china_money_supply()),
    ]:
        try:
            df = fn()
            say("[%s] shape=%s 列=%s" % (name, df.shape, list(df.columns)))
            say(df.tail(2).to_string()[:900])
        except Exception as exc:
            say("[%s] 失败: %r" % (name, exc))


def t_ak_shibor():
    import akshare as ak
    try:
        df = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币",
                               indicator="3月")
        say("shape=%s 列=%s" % (df.shape, list(df.columns)))
        say(df.tail(3).to_string()[:900])
    except Exception as exc:
        say("rate_interbank 失败: %r" % exc)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [
        ("1.腾讯实时快照 qt.gtimg.cn", t_tencent),
        ("2.东财日频资金流 fflow/daykline", t_fflow_day),
        ("3.东财分时资金流 fflow/kline klt=1", t_fflow_min),
        ("4.东财所属板块 slist(spt=3)", t_slist),
        ("4b.东财所属板块 slist(仅代码)", t_slist_plain),
        ("5.东财板块成分 clist(b:BK0475)", t_clist),
        ("6.东财 RPT_LICO_FN_CPD 财务", t_lixo),
        ("7.akshare stock_financial_abstract", t_ak_fin),
        ("8a.akshare stock_zh_a_gdhs_detail_em", t_ak_gdhs),
        ("8b.东财 RPT_HOLDERNUMLATEST", t_em_holdernum),
        ("9.东财 RPTA_WEB_RZRQ_GGMX 两融", t_margin),
        ("10a.东财 RPT_DAILYBILLBOARD_DETAILSNEW", t_lhb),
        ("10b.东财 RPT_ORGANIZATION_TRADE_DETAILS", t_lhb_org),
        ("11a.东财 RPT_MUTUAL_HOLDSTOCKNORTH_STA", t_north),
        ("11b.akshare stock_hsgt_individual_em", t_ak_hsgt),
        ("12.东财 RPT_F10_EH_FREEHOLDERS", t_top10),
        ("13a.getTopicZTPool", t_ztpool),
        ("13b.getTopicFBPool", t_fbpool),
        ("14.东财 ulist.np 指数", t_ulist),
        ("15a.股吧人气排名(100)", t_guba_rank),
        ("15b.股吧人气排名(全量)", t_guba_rank2),
        ("16.东财搜索 jsonp", t_search),
        ("17.新浪行情 hq.sinajs.cn", t_sina_hq),
        ("18a.东财公告 np-anotice", t_ann),
        ("19.akshare stock_zh_index_daily", t_ak_index),
        ("20.akshare 宏观", t_ak_macro),
        ("20b.akshare rate_interbank SHIBOR", t_ak_shibor),
        ("21.yfinance ^VIX", t_yf_vix),
    ]
    for title, fn in tests:
        if only and only not in title:
            continue
        probe(title, fn)

    txt = "# 接口探测结果（真实网络实测）\n" + "\n".join(OUT)
    with open(r"E:\deepseek desktop\个股预测程序\_probe\PROBE_RESULTS.md", "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    print("\n\n>>> 结果已写入 _probe/PROBE_RESULTS.md")

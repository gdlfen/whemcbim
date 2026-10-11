# -*- coding: utf-8 -*-
"""第二轮探测：针对第一轮失败/不明确的点。结果写 _probe/PROBE_RESULTS2.md。"""

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
    print(str(m)[:2500])


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
        say(traceback.format_exc(limit=3)[-1200:])


def get(url, **kw):
    kw.setdefault("timeout", 12)
    return S.get(url, **kw)


def js(url, **kw):
    return get(url, **kw).json()


# ------------------------------------------------- A. akshare 财务指标全名单
def a_ak_fin_names():
    import akshare as ak
    df = ak.stock_financial_abstract(symbol="600519")
    cur = None
    for _, row in df.iterrows():
        if row["选项"] != cur:
            cur = row["选项"]
            say("【选项】%s" % cur)
        say("   - %s" % row["指标"])
    say("列示例（报告期）: %s" % list(df.columns[2:8]))


# ------------------------------------------------- B. 东财日频资金流重试
def b_fflow(secid="1.600519", klt=101, lmt=20):
    url = ("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
           "?lmt=%d&klt=%d&secid=%s&fields1=f1,f2,f3,f7"
           "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
           "&ut=b2884a393a59ad64002292a3e90d46a5" % (lmt, klt, secid))
    last = None
    for i in range(4):
        try:
            d = js(url)
            klines = ((d.get("data") or {}).get("klines")) or []
            say("尝试 %d 成功，klines=%d 条" % (i + 1, len(klines)))
            if klines:
                cells = klines[-1].split(",")
                say("字段数=%d" % len(cells))
                for j, c in enumerate(cells):
                    say("    [%d] = %s" % (j, c))
                say("倒数第二行: %s" % klines[-2])
            return
        except Exception as exc:
            last = exc
            say("尝试 %d 失败: %r" % (i + 1, exc))
    say("全部失败: %r" % last)


def b_fflow_min():
    b_fflow(klt=1, lmt=0)


def b_fflow_sz():
    b_fflow(secid="0.000001")


# ------------------------------------------------- C. slist 全板块 + 行业板块
def c_slist_all():
    url = ("https://push2.eastmoney.com/api/qt/slist/get?secid=1.600519&spt=3"
           "&fields=f12,f13,f14,f3,f62,f184&po=1&pn=1&pz=60&fid=f62"
           "&ut=b2884a393a59ad64002292a3e90d46a5")
    d = js(url)
    diff = (d.get("data") or {}).get("diff") or {}
    items = list(diff.values()) if isinstance(diff, dict) else list(diff)
    say("板块数=%d" % len(items))
    for it in items:
        say("   %s %s f3=%s f62=%s" % (it.get("f12"), it.get("f14"),
                                       it.get("f3"), it.get("f62")))


def c_industry_boards():
    url = ("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=200&po=1&np=1"
           "&fltt=2&invt=2&fid=f3&fs=m:90+t:2&fields=f12,f14,f3,f62"
           "&ut=b2884a393a59ad64002292a3e90d46a5")
    d = js(url)
    diff = (d.get("data") or {}).get("diff") or []
    say("行业板块数=%d" % len(diff))
    codes = {it["f12"]: it["f14"] for it in diff}
    say("含 白酒/酿酒/饮料 的板块: %s" % {k: v for k, v in codes.items()
                                          if any(x in str(v) for x in ("白酒", "酿酒", "饮料", "食品"))})
    say("BK1277 = %s" % codes.get("BK1277"))
    say("前 10: %s" % json.dumps(diff[:10], ensure_ascii=False))


def c_board_members():
    url = ("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=60&po=1&np=1"
           "&fltt=2&invt=2&fid=f3&fs=b:BK1277&fields=f12,f14,f9,f23,f3,f20"
           "&ut=b2884a393a59ad64002292a3e90d46a5")
    d = js(url)
    diff = (d.get("data") or {}).get("diff") or []
    say("BK1277 成分数=%d" % len(diff))
    for it in diff[:8]:
        say("   %s" % json.dumps(it, ensure_ascii=False))


# ------------------------------------------------- D. 涨停/炸板池
def d_fbpool():
    import datetime as dt
    for back in range(0, 10):
        day = (dt.date.today() - dt.timedelta(days=back)).strftime("%Y%m%d")
        url = ("https://push2ex.eastmoney.com/getTopicFBPool?ut=7eea3edcaed734bea9cbfc24409ed989"
               "&dpt=wz.ztzt&Pageindex=0&pagesize=100&sort=fbt%3Aasc&date=" + day)
        r = get(url, headers={"Referer": "https://quote.eastmoney.com/"})
        say("%s status=%s body前200=%s" % (day, r.status_code, r.text[:200]))
        try:
            d = r.json()
        except Exception:
            continue
        data = d.get("data")
        if data:
            pool = data.get("pool") or []
            say("  -> tc=%s pool=%d" % (data.get("tc"), len(pool)))
            if pool:
                say("  字段: %s" % list(pool[0].keys()))
                say("  样例: %s" % json.dumps(pool[0], ensure_ascii=False))
            break


def d_ztpool_raw():
    import datetime as dt
    day = dt.date.today().strftime("%Y%m%d")
    url = ("https://push2ex.eastmoney.com/getTopicZTPool?ut=7eea3edcaed734bea9cbfc24409ed989"
           "&dpt=wz.ztzt&Pageindex=0&pagesize=3&sort=fbt%3Aasc&date=" + day)
    r = get(url, headers={"Referer": "https://quote.eastmoney.com/"})
    say("今天=%s 前 600 字: %s" % (day, r.text[:600]))


# ------------------------------------------------- E. 股吧人气排名
def e_guba():
    for body in [
        {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38",
         "marketType": "", "srcSecurityCode": "600519", "pageNo": 1, "pageSize": 100},
        {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38",
         "marketType": "", "srcSecurityCode": "600519", "pageNo": 1, "pageSize": 10},
    ]:
        try:
            r = S.post("https://emappdata.eastmoney.com/stockrank/getCurrentList",
                       json=body, headers={"Content-Type": "application/json"}, timeout=12)
            say("POST %s -> %s" % (json.dumps(body, ensure_ascii=False)[:120], r.text[:700]))
        except Exception as exc:
            say("失败: %r" % exc)
    # 历史人气（GubaRank 历史曲线）
    for url in [
        "https://emappdata.eastmoney.com/stockrank/getHisList?appId=appId01&globalId=786e4c21-70dc-435a-93bb-38&srcSecurityCode=600519&pageNo=1&pageSize=30",
        "https://guba.eastmoney.com/rank/",
    ]:
        try:
            r = get(url)
            say("GET %s -> status=%s 前 300: %s" % (url[:60], r.status_code, r.text[:300]))
        except Exception as exc:
            say("GET %s 失败: %r" % (url[:60], exc))


# ------------------------------------------------- F. 搜索接口
def f_search():
    for kw, types in [("贵州茅台", ["cmsArticleWebOld"]), ("600519", ["cmsArticleWebOld"]),
                      ("贵州茅台", ["cmsArticleWebOld", "cmsArticleWeb", "webSuggest"])]:
        param = {
            "uid": "", "keyword": kw, "type": types, "client": "web",
            "clientType": "web", "clientVersion": "curr",
            "param": {t: {"searchScope": "default", "sort": "time", "pageIndex": 1,
                          "pageSize": 3, "preTag": "<em>", "postTag": "</em>"} for t in types},
        }
        url = ("https://search-api-web.eastmoney.com/search/jsonp?cb=cb&param="
               + urllib.parse.quote(json.dumps(param, ensure_ascii=False)))
        try:
            r = get(url)
            say("[kw=%s types=%s] 前 900: %s" % (kw, types, r.text[:900]))
        except Exception as exc:
            say("[kw=%s] 失败 %r" % (kw, exc))


# ------------------------------------------------- G. 腾讯其它股票（确认索引）
def g_tencent_icbc():
    r = get("https://qt.gtimg.cn/q=sh601398,sz000858")
    r.encoding = "gbk"
    for line in r.text.strip().split("\n"):
        if "=" not in line:
            continue
        code = line.split("=")[0]
        body = line.split('="', 1)[1].rsplit('"', 1)[0]
        p = body.split("~")
        say("%s name=%s 字段数=%d" % (code, p[1] if len(p) > 1 else "?", len(p)))
        for i in [3, 4, 38, 39, 43, 44, 45, 46, 49, 52, 53, 57, 58, 62, 63, 64, 65, 72, 73, 74, 75]:
            if i < len(p):
                say("   [%d]=%s" % (i, p[i]))


# ------------------------------------------------- H. 股息率来源
def h_dividend():
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
           "reportName=RPT_SHAREBONUS_DET&columns=ALL"
           "&filter=(SECURITY_CODE%3D%22600519%22)&pageSize=4&sortColumns=REPORT_DATE&sortTypes=-1")
    try:
        d = js(url)
        res = (d.get("result") or {}).get("data") or []
        say("行数=%d msg=%s" % (len(res), d.get("message")))
        if res:
            say("字段: %s" % list(res[0].keys()))
            for r0 in res[:2]:
                say(json.dumps(r0, ensure_ascii=False)[:900])
    except Exception as exc:
        say("失败 %r" % exc)
    # 东财 F10 股息率
    for u in ["https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_F10_DIVIDEND_INFO&columns=ALL&filter=(SECUCODE%3D%22600519.SH%22)&pageSize=3",
              "https://push2.eastmoney.com/api/qt/stock/get?secid=1.600519&fields=f43,f57,f58,f162,f167,f168,f169,f170,f173&ut=b2884a393a59ad64002292a3e90d46a5"]:
        try:
            r = get(u)
            say("[%s] -> %s" % (u[:70], r.text[:600]))
        except Exception as exc:
            say("[%s] 失败 %r" % (u[:70], exc))


# ------------------------------------------------- I. VIX 备选
def i_vix():
    cands = [
        ("sina gb_$vix", "https://hq.sinajs.cn/list=gb_$vix", {"Referer": "https://finance.sina.com.cn"}),
        ("sina gb_vix", "https://hq.sinajs.cn/list=gb_vix", {"Referer": "https://finance.sina.com.cn"}),
        ("sina znb_vix", "https://hq.sinajs.cn/list=znb_vix", {"Referer": "https://finance.sina.com.cn"}),
        ("em 100.VIX", "https://push2.eastmoney.com/api/qt/stock/get?secid=100.VIX&fields=f43,f57,f58,f169,f170", {}),
        ("em 100.NDX", "https://push2.eastmoney.com/api/qt/stock/get?secid=100.NDX&fields=f43,f57,f58,f169,f170", {}),
        ("sina int_vix", "https://hq.sinajs.cn/list=int_vix", {"Referer": "https://finance.sina.com.cn"}),
        ("em ulist vix", "https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&secids=100.VIX,100.DJIA&fields=f2,f3,f12,f14", {}),
    ]
    for name, url, hdr in cands:
        try:
            r = get(url, headers=hdr or None)
            r.encoding = "gbk"
            say("[%s] -> %s" % (name, r.text.strip()[:300]))
        except Exception as exc:
            say("[%s] 失败 %r" % (name, exc))


# ------------------------------------------------- J. 北向 / akshare 签名
def j_north():
    for url in [
        "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_MUTUAL_HOLDSTOCKNORTH_STA&columns=ALL&filter=(SECURITY_CODE%3D%22600519%22)&pageSize=3",
        "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_MUTUAL_HOLDSTOCKNORTH_STA&columns=ALL&filter=(SECURITY_CODE%3D%22600519%22)&pageSize=3&sortColumns=TRADE_DATE&sortTypes=-1",
        "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_MUTUAL_STOCK_NORTHSTA&columns=ALL&filter=(SECURITY_CODE%3D%22600519%22)&pageSize=3",
        "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_MUTUAL_HOLD_DET&columns=ALL&filter=(SECURITY_CODE%3D%22600519%22)&pageSize=3",
    ]:
        try:
            r = get(url)
            txt = r.text
            say("[%s] -> %s" % (url.split("reportName=")[1][:40], txt[:500]))
        except Exception as exc:
            say("失败 %r" % exc)


def j_ak_hsgt():
    import akshare as ak
    import inspect
    say("stock_hsgt_individual_em 签名: %s" % inspect.signature(ak.stock_hsgt_individual_em))
    try:
        df = ak.stock_hsgt_individual_em(stock="600519")
        say("shape=%s 列=%s" % (df.shape, list(df.columns)))
    except Exception as exc:
        say("调用失败: %r" % exc)


# ------------------------------------------------- K. 宏观补充
def k_macro():
    import akshare as ak
    for name, fn in [("cpi(全国)", lambda: ak.macro_china_cpi()),
                     ("shrzgm 社融", lambda: ak.macro_china_shrzgm()),
                     ("cpi_yearly", lambda: ak.macro_china_cpi_yearly()),
                     ("pmi_baidu?", lambda: ak.macro_china_pmi_yearly())]:
        try:
            df = fn()
            say("[%s] shape=%s 列=%s" % (name, df.shape, list(df.columns)))
            say(df.head(3).to_string()[:1000])
        except Exception as exc:
            say("[%s] 失败: %r" % (name, exc))


# ------------------------------------------------- L. 写权限（缓存目录）
def l_write():
    for d in [os.path.expanduser("~"), r"E:\deepseek desktop\个股预测程序"]:
        p = os.path.join(d, "_dsh_write_test.tmp")
        try:
            with open(p, "w") as fh:
                fh.write("x")
            os.remove(p)
            say("可写: %s" % d)
        except Exception as exc:
            say("不可写: %s (%r)" % (d, exc))


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [
        ("A.akshare 财务指标全名单", a_ak_fin_names),
        ("B1.东财日频资金流(重试4次)", b_fflow),
        ("B2.东财分时资金流 klt=1", b_fflow_min),
        ("B3.东财日频资金流 深市指数", b_fflow_sz),
        ("C1.个股所属板块全列表", c_slist_all),
        ("C2.东财行业板块清单 m:90 t:2", c_industry_boards),
        ("C3.BK1277 板块成分", c_board_members),
        ("D1.炸板池 getTopicFBPool", d_fbpool),
        ("D2.涨停池原始响应", d_ztpool_raw),
        ("E.股吧人气排名(srcSecurityCode)", e_guba),
        ("F.东财搜索接口", f_search),
        ("G.腾讯快照其它股票索引", g_tencent_icbc),
        ("H.股息率来源", h_dividend),
        ("I.VIX 备选源", i_vix),
        ("J1.北向报表变体", j_north),
        ("J2.akshare hsgt 签名", j_ak_hsgt),
        ("K.宏观补充接口", k_macro),
        ("L.目录写权限", l_write),
    ]
    for t, fn in tests:
        if only and only not in t:
            continue
        probe(t, fn)
    with open(r"E:\deepseek desktop\个股预测程序\_probe\PROBE_RESULTS2.md", "w",
              encoding="utf-8") as fh:
        fh.write("# 第二轮接口探测（真实网络实测）\n" + "\n".join(OUT))
    print("\n>>> 已写入 _probe/PROBE_RESULTS2.md")

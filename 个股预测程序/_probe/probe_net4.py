# -*- coding: utf-8 -*-
"""第四轮探测：新浪 K 线/资金流、行业板块交集、涨跌家数。"""

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
OUT = []


def say(m):
    OUT.append(str(m))
    print(str(m)[:1500])


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
        say(traceback.format_exc(limit=3)[-800:])


def get(url, **kw):
    kw.setdefault("timeout", 15)
    return S.get(url, **kw)


def a_sina_kline():
    u = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         "CN_MarketData.getKLineData?symbol=sh600519&scale=240&ma=no&datalen=1300")
    r = get(u, headers={"Referer": "https://finance.sina.com.cn"})
    txt = r.text
    say("len=%d 前 300=%s" % (len(txt), txt[:300]))
    try:
        d = json.loads(txt)
        say("条数=%d 首=%s 末=%s" % (len(d), d[0], d[-1]))
    except Exception as exc:
        say("解析失败 %r" % exc)


def a2_tencent_kline():
    u = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600519,day,,,640,qfq")
    r = get(u)
    d = r.json()
    data = (d.get("data") or {}).get("sh600519") or {}
    say("键: %s" % list(data.keys()))
    for k in ("qfqday", "day"):
        if k in data:
            say("%s 条数=%d 首=%s 末=%s" % (k, len(data[k]), data[k][0], data[k][-1]))


def b_industry_boards():
    for host in ["push2.eastmoney.com", "push2delay.eastmoney.com"]:
        u = ("https://%s/api/qt/clist/get?pn=1&pz=200&po=1&np=1&fltt=2&invt=2&fid=f3"
             "&fs=m:90+t:2&fields=f12,f14,f3,f62" % host)
        for i in range(3):
            try:
                d = get(u).json()
                diff = (d.get("data") or {}).get("diff") or []
                say("[%s] 第%d次 行业板块数=%d" % (host, i + 1, len(diff)))
                codes = {it["f12"]: it["f14"] for it in diff}
                say("  BK1277=%s BK0477=%s BK0896=%s BK0438=%s" %
                    (codes.get("BK1277"), codes.get("BK0477"), codes.get("BK0896"),
                     codes.get("BK0438")))
                return
            except Exception as exc:
                say("[%s] 第%d次失败 %r" % (host, i + 1, exc))


def c_board_code():
    u = ("https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_LICO_FN_CPD"
         "&columns=SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,REPORTDATE,PUBLISHNAME,"
         "BOARD_CODE,BOARD_NAME&filter=(SECUCODE%3D%22600519.SH%22)&pageSize=1"
         "&sortColumns=REPORTDATE&sortTypes=-1")
    d = get(u).json()
    res = (d.get("result") or {}).get("data") or []
    say("行数=%d 首行=%s" % (len(res), json.dumps(res[0], ensure_ascii=False) if res else "无"))


def d_breadth():
    for secids, fields in [("1.000001", "f2,f3,f12,f14,f104,f105,f106"),
                           ("1.000001,0.399001,0.399006", "f2,f3,f12,f14,f104,f105,f106")]:
        u = ("https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&secids=%s&fields=%s"
             % (secids, fields))
        for i in range(3):
            try:
                d = get(u).json()
                say("[%s] -> %s" % (secids, json.dumps(d, ensure_ascii=False)[:700]))
                break
            except Exception as exc:
                say("[%s] 第%d次失败 %r" % (secids, i + 1, exc))


def e_sina_flow():
    u = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         "MoneyFlow.ssl_qsfx_lscjfb?page=1&num=120&sort=opendate&asc=0&daima=sh600519")
    r = get(u, headers={"Referer": "https://finance.sina.com.cn"})
    d = json.loads(r.text)
    say("条数=%d" % len(d))
    say("首行=%s" % json.dumps(d[0], ensure_ascii=False))
    say("末行=%s" % json.dumps(d[-1], ensure_ascii=False))
    say("键=%s" % list(d[0].keys()))
    r2 = get(u.replace("num=120", "num=5").replace("sh600519", "sz300750"),
             headers={"Referer": "https://finance.sina.com.cn"})
    try:
        d2 = json.loads(r2.text)
        say("创业板 300750 条数=%d 首行=%s" % (len(d2), json.dumps(d2[0], ensure_ascii=False) if d2 else "无"))
    except Exception as exc:
        say("300750 解析失败 %r -> %s" % (exc, r2.text[:200]))
    # 分钟级资金流备选
    for u3 in ["https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               "MoneyFlow.ssl_qsfx_zjlrqs?page=1&num=5&sort=opendate&asc=0&daima=sh600519",
               "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=1.600519"
               "&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"]:
        try:
            r3 = get(u3)
            say("[分钟/%s] status=%s 前 200=%s" % (u3.split("/")[-1][:30], r3.status_code, r3.text[:200]))
        except Exception as exc:
            say("[分钟] 失败 %r" % exc)


def f_f10_industry():
    """东财 F10 所属行业接口。"""
    for rn, filt in [("RPT_F10_CORETHEME_BOARDTYPE", '(SECUCODE="600519.SH")'),
                     ("RPT_F10_BASIC_ORGINFO", '(SECUCODE="600519.SH")')]:
        u = ("https://datacenter.eastmoney.com/securities/api/data/v1/get?reportName=%s"
             "&columns=ALL&filter=%s&pageSize=5" % (rn, filt.replace('"', "%22")))
        try:
            r = get(u)
            say("[%s] %s" % (rn, r.text[:400]))
        except Exception as exc:
            say("[%s] 失败 %r" % (rn, exc))


if __name__ == "__main__":
    for t, fn in [("A.新浪日线", a_sina_kline), ("A2.腾讯日线", a2_tencent_kline),
                  ("B.行业板块清单", b_industry_boards), ("C.东财行业板块代码", c_board_code),
                  ("D.涨跌家数字段", d_breadth), ("E.新浪资金流", e_sina_flow),
                  ("F.东财F10行业", f_f10_industry)]:
        probe(t, fn)
    with open(r"E:\deepseek desktop\个股预测程序\_probe\PROBE_RESULTS4.md", "w",
              encoding="utf-8") as fh:
        fh.write("# 第四轮接口探测\n" + "\n".join(OUT))
    print("\n>>> 已写入 PROBE_RESULTS4.md")

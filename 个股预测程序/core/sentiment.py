# -*- coding: utf-8 -*-
"""
市场情绪与博弈心理模块（Sentiment）。

职责
----
把**全市场赚钱效应、涨停/炸板生态、板块热度与龙头联动、散户关注度、两融杠杆情绪、
换手狂热、恐惧贪婪综合指数、波动率区间**压缩成 :class:`core.contracts.Factor`。

产出因子 id
-----------
``market_breadth``、``limit_up_frenzy``、``sector_heat``、``sector_leader_linkage``、
``margin_euphoria``、``margin_trend``、``retail_sentiment_contrarian``、``guba_bull_ratio``、
``search_heat_spike``、``turnover_euphoria``、``fear_greed``、``volatility_regime``。

**反向指标约定**：``core/config.py`` 的 ``CONTRARIAN_FACTOR_IDS`` 收录了
``retail_sentiment_contrarian / guba_bull_ratio / search_heat_spike / turnover_euphoria /
margin_euphoria / limit_up_frenzy``。这些因子在本模块中按"**原始看涨信号**"写 ``direction``
（情绪越亢奋越写 +1），由预测引擎统一翻转；``detail`` 里必须写明"这是反向指标"。
``fear_greed`` 不在反向名单里，因此**本模块内部直接做反向编码**（极度恐惧→看涨）。

数据源与实测结论
----------------
* 指数快照：``push2.eastmoney.com/api/qt/ulist.np/get``（本机**间歇性** RemoteDisconnected，
  全部请求走 :func:`core.utils.retry_call` attempts=3）。
* 涨停池：``push2ex.eastmoney.com/getTopicZTPool`` **实测可用**，字段
  ``c/m/n/p(价格×1000)/zdp/lbc(连板数)/fbt/lbt/fund(封单额)/zbc(炸板次数)/hybk(行业)/hs(换手)``，
  ``data.tc``=涨停家数、``data.qdate``=实际数据交易日（非交易日会回退）。
* 炸板池：``getTopicFBPool`` 实测 **404**；正确端点为 ``getTopicZBPool``（实测可用，
  字段含 ``ztp``(涨停价×1000)、``zf``(振幅)、``zbc``），``broken_rate = 炸板数/(涨停数+炸板数)``。
* 所属板块：``push2.eastmoney.com/api/qt/slist/get?spt=3`` 实测可用，``data.diff`` 是
  **dict**（键 "0","1",…）需兼容 list；**不加 ``fltt=2`` 时 f3/f184 是放大 100 倍的整数**。
* 板块成分（找龙头）：``api/qt/clist/get?fs=b:BKxxxx&fid=f3``。
* 股吧人气：``POST emappdata.eastmoney.com/stockrank/getCurrentList`` 实测返回
  ``data=[]``（**拿不到**，属正常），东财搜索接口返回与关键词无关的固定内容 → 一并降级。
* 散户代理：腾讯快照 ``turnover/volume_ratio`` + 个股换手率历史分位（本机实测可用）。
"""

from __future__ import annotations

import datetime as dt
import math
import re

import numpy as np
import pandas as pd

from . import config
from .contracts import (Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor,
                        clamp, now_cn, safe_float)
from .utils import (cache_key, cached, get_logger, http_get, normalize_symbol,
                    percentile_rank, retry_call)

log = get_logger("stockai.sentiment")

__all__ = [
    "fetch_market_sentiment", "fetch_sector_heat", "fetch_retail_sentiment",
    "sentiment_factors", "market_breadth", "fear_greed_index", "limit_up_stats",
    "margin_sentiment", "fetch_market_margin", "last_notes", "CONTRARIAN_HINT",
]

_EM_ULIST = "https://push2.eastmoney.com/api/qt/ulist.np/get"
_EM_SLIST = "https://push2.eastmoney.com/api/qt/slist/get"
_EM_CLIST = "https://push2.eastmoney.com/api/qt/clist/get"
_EM_ZTPOOL = ("https://push2ex.eastmoney.com/getTopicZTPool?ut=7eea3edcaed734bea9cbfc24409ed989"
              "&dpt=wz.ztzt&Pageindex=0&pagesize=100&sort=fbt%%3Aasc&date=%s")
_EM_ZBPOOL = ("https://push2ex.eastmoney.com/getTopicZBPool?ut=7eea3edcaed734bea9cbfc24409ed989"
              "&dpt=wz.ztzt&Pageindex=0&pagesize=100&sort=fbt%%3Aasc&date=%s")
_TENCENT_URL = "https://qt.gtimg.cn/q=%s"
_GUBA_RANK = "https://emappdata.eastmoney.com/stockrank/getCurrentList"

#: 腾讯快照实测索引（见 core/fundamental.py 注释）
_TX_IDX = {"name": 1, "price": 3, "prev_close": 4, "pct_chg": 32, "turnover": 38,
           "pe_ttm": 39, "amplitude": 43, "float_market_cap_yi": 44, "market_cap_yi": 45,
           "pb": 46, "volume_ratio": 49}

#: 情绪类因子适用周期（短线为主，恐惧贪婪可稍长）
SENT_HORIZON = ("d1", "d5", "d10", "d20")
SENT_HORIZON_LONG = ("d5", "d10", "d20", "m1")

#: 恐惧贪婪指数各分项权重（合计 1.0）
FEAR_GREED_WEIGHTS = {
    "new_high_ratio": 0.20,     # 个股 20 日创新高占比：越高越贪婪
    "turnover_percentile": 0.25,  # 换手率历史分位：越高越亢奋
    "margin_percentile": 0.20,  # 两融余额历史分位：越高杠杆越拥挤
    "breadth_ratio": 0.20,      # 大盘涨跌家数比：越高越贪婪
    "vol_percentile_inv": 0.15,  # 波动率分位（**反向**：高波动=恐惧）
}

#: 反向指标提示文案（写进 detail，避免用户误解方向）
CONTRARIAN_HINT = "（反向指标：情绪越亢奋越看跌，由引擎统一翻转方向）"

_LAST_NOTES: list = []


def last_notes() -> list:
    """最近一次取数的降级说明（中文）。"""
    return list(_LAST_NOTES)


def _note(msg: str) -> None:
    if msg and msg not in _LAST_NOTES:
        _LAST_NOTES.append(msg)
    log.debug(msg)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _num(x, default=float("nan")) -> float:
    return safe_float(x, default)


def _pct(v, default=float("nan")) -> float:
    """把东财 ``f3``/``f184`` 统一成百分数。

    实测：**不加 ``fltt=2``** 时这些字段是放大 100 倍的整数（f3=204 → 2.04%），
    加了 ``fltt=2`` 才是小数（2.04）。这里做双向兼容：绝对值 > 30 视为放大值。
    """
    x = _num(v, default)
    if math.isnan(x):
        return default
    return x / 100.0 if abs(x) > 30 else x


def _ensure_frame(df):
    """列名容错（大写/中文 → 小写英文），非 DataFrame 返回 None。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    ren = {"日期": "date", "时间": "date", "Date": "date", "DATE": "date",
           "收盘": "close", "开盘": "open", "最高": "high", "最低": "low",
           "成交量": "volume", "成交额": "amount", "换手率": "turnover",
           "涨跌幅": "pct_chg", "Close": "close", "Volume": "volume"}
    return df.rename(columns={k: v for k, v in ren.items() if k in df.columns}).copy()


def _code(symbol) -> str:
    try:
        return normalize_symbol(symbol)["code"]
    except Exception:  # noqa: BLE001
        return re.sub(r"\D", "", str(symbol or ""))[:6]


def _secid(symbol) -> str:
    try:
        return normalize_symbol(symbol)["secid"]
    except Exception:  # noqa: BLE001
        return "1.600519"


def _prefix(symbol) -> str:
    try:
        return normalize_symbol(symbol)["display"]
    except Exception:  # noqa: BLE001
        return "sh600519"


def _neutral(fid: str, name: str, detail: str, reliability: float = 0.2, value=None,
             theory: str = "行为金融学", horizon=SENT_HORIZON) -> Factor:
    """数据缺失时的中性因子（保证 id 集合稳定）。"""
    return Factor(fid, name, Category.SENTIMENT, DIR_NEUTRAL, 0.0,
                  clamp(reliability, 0.05, 0.5), value=value, detail=detail,
                  theory=theory, horizon=horizon)


# --------------------------------------------------------------------------- #
# 纯解析函数（可离线自测，不联网）
# --------------------------------------------------------------------------- #
def _parse_ulist(payload) -> list:
    """解析东财 ``ulist.np`` 指数快照 → ``[{code,name,price,pct,up,down,flat}]``。"""
    diff = ((payload or {}).get("data") or {}).get("diff")
    if diff is None:
        return []
    items = list(diff.values()) if isinstance(diff, dict) else list(diff)
    out = []
    for it in items:
        out.append({
            "code": str(it.get("f12") or ""), "name": it.get("f14"),
            "price": _num(it.get("f2")), "pct": _num(it.get("f3")),
            "up": _num(it.get("f104")), "down": _num(it.get("f105")),
            "flat": _num(it.get("f106")),
        })
    return out


def _parse_zt_pool(payload) -> dict:
    """解析涨停板池。

    Returns
    -------
    dict
        ``{"limit_up_count","max_continuous","continuous_count","date","items"}``；
        ``items`` 元素：``{code,name,price,pct,continuous,fund(封单额),zbc(炸板次数),
        turnover,board}``（``p`` 实测为价格×1000）。
    """
    data = (payload or {}).get("data") or {}
    pool = data.get("pool") or []
    items = []
    for it in pool:
        items.append({
            "code": str(it.get("c") or ""), "name": it.get("n"),
            "price": _num(it.get("p")) / 1000.0, "pct": _num(it.get("zdp")),
            "continuous": int(_num(it.get("lbc"), 1) or 1),
            "fund": _num(it.get("fund")), "zbc": int(_num(it.get("zbc"), 0) or 0),
            "turnover": _num(it.get("hs")), "board": it.get("hybk"),
            "first_seal": it.get("fbt"),
        })
    conts = [x["continuous"] for x in items if x["continuous"] >= 2]
    return {
        "limit_up_count": int(_num(data.get("tc"), len(items)) or len(items)),
        "max_continuous": int(max([x["continuous"] for x in items], default=0)),
        "continuous_count": len(conts),
        "date": str(data.get("qdate") or ""),
        "items": items,
    }


def _parse_zb_pool(payload) -> dict:
    """解析炸板池（``getTopicZBPool``）→ ``{"broken_count","items"}``。"""
    data = (payload or {}).get("data") or {}
    pool = data.get("pool") or []
    items = [{"code": str(it.get("c") or ""), "name": it.get("n"),
              "pct": _num(it.get("zdp")), "amplitude": _num(it.get("zf")),
              "zbc": int(_num(it.get("zbc"), 0) or 0), "board": it.get("hybk")}
             for it in pool]
    return {"broken_count": int(_num(data.get("tc"), len(items)) or len(items)),
            "items": items}


def _parse_slist(payload) -> list:
    """解析个股所属板块列表 → ``[{code,name,pct,main_net,main_pct}]``。

    实测 ``data.diff`` 可能是 dict（键为 "0","1",…）或 list；``f62``=板块主力净流入(元)，
    ``f184`` 与 ``f3`` 在不加 ``fltt=2`` 时放大 100 倍（见 :func:`_pct`）。
    """
    diff = ((payload or {}).get("data") or {}).get("diff")
    if diff is None:
        return []
    items = list(diff.values()) if isinstance(diff, dict) else list(diff)
    out = []
    for it in items:
        code = str(it.get("f12") or "")
        if not code.startswith("BK"):
            continue
        out.append({"code": code, "name": it.get("f14"), "pct": _pct(it.get("f3")),
                    "main_net": _num(it.get("f62")), "main_pct": _pct(it.get("f184"))})
    return out


# --------------------------------------------------------------------------- #
# 1. 全市场情绪
# --------------------------------------------------------------------------- #
def _producer_limit_up(days_back: int = 8) -> dict:
    """涨停/炸板统计（涨停池 + 炸板池）。"""
    today = now_cn().date()
    for i in range(days_back):
        d = (today - dt.timedelta(days=i)).strftime("%Y%m%d")
        payload = retry_call(lambda: http_get(_EM_ZTPOOL % d, json_body=True,
                                              headers={"Referer": "https://quote.eastmoney.com/"}),
                             attempts=2, delay=0.6, default=None, label="ztpool")
        zt = _parse_zt_pool(payload)
        if zt["items"] or zt["limit_up_count"]:
            zb_payload = retry_call(
                lambda: http_get(_EM_ZBPOOL % d, json_body=True,
                                 headers={"Referer": "https://quote.eastmoney.com/"}),
                attempts=2, delay=0.6, default=None, label="zbpool")
            zb = _parse_zb_pool(zb_payload)
            total = zt["limit_up_count"] + zb["broken_count"]
            out = {
                "limit_up_count": zt["limit_up_count"],
                "limit_down_count": float("nan"),   # 跌停池接口未验证，留 NaN 由 breadth 补
                "max_continuous": zt["max_continuous"],
                "continuous_count": zt["continuous_count"],
                "broken_count": zb["broken_count"],
                "broken_rate": (zb["broken_count"] / total) if total > 0 else float("nan"),
                "date": zt["date"] or d,
                "boards_hot": _hot_boards(zt["items"]),
                "source": "eastmoney_ztpool+zbpool",
            }
            _note("涨停统计（%s）：涨停 %d 家、连板 %d 家、最高 %d 板、炸板 %d 家、炸板率 %s"
                  % (out["date"], out["limit_up_count"], out["continuous_count"],
                     out["max_continuous"], out["broken_count"],
                     "%.1f%%" % (out["broken_rate"] * 100)
                     if not math.isnan(out["broken_rate"]) else "—"))
            return out
    _note("涨停池接口连续 %d 天无数据，涨停统计降级为 NaN" % days_back)
    return {"limit_up_count": float("nan"), "limit_down_count": float("nan"),
            "max_continuous": float("nan"), "continuous_count": float("nan"),
            "broken_count": float("nan"), "broken_rate": float("nan"),
            "date": "", "boards_hot": [], "source": "none"}


def _hot_boards(items: list, top: int = 5) -> list:
    """涨停股所属行业统计（赚钱效应所在板块）。"""
    cnt: dict = {}
    for it in items or []:
        b = it.get("board")
        if b:
            cnt[b] = cnt.get(b, 0) + 1
    return [{"board": k, "count": v}
            for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:top]]


def limit_up_stats(force=False) -> dict:
    """涨停/跌停/连板/炸板率统计。

    Returns
    -------
    dict
        ``{"limit_up_count","limit_down_count","max_continuous","broken_rate",
        "continuous_count"}``（取不到为 NaN）。
    """
    key = cache_key("limitups", now_cn().strftime("%Y%m%d%H"))

    def _p() -> dict:
        d = _producer_limit_up()
        return {k: d.get(k, float("nan")) for k in
                ("limit_up_count", "limit_down_count", "max_continuous", "broken_rate",
                 "continuous_count")} | {"date": d.get("date", ""), "source": d.get("source")}

    try:
        if force:
            return _p()
        out = cached(key, 600, _p)
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("limit_up_stats 异常降级：%r" % exc)
        return {}


def market_breadth(force=False) -> dict:
    """全市场涨跌家数 + 涨停生态。

    Returns
    -------
    dict
        ``{"up_count","down_count","flat_count","limit_up_count","limit_down_count",
        "max_continuous","continuous_count","broken_rate","indices":[{"name","pct","price"}],
        "breadth_ratio","breadth_source","source","detail"}``。

    说明
    ----
    涨跌家数优先取东财指数快照的 ``f104/f105/f106`` 字段（上涨/下跌/平盘家数）；
    该字段在本机因 push2 间歇性拒连可能取不到，此时用**指数涨跌幅**做粗代理，
    并把 ``breadth_source`` 标为 ``"index_proxy"``（相关因子可信度下调到 ≤0.4）。
    """
    key = cache_key("breadth", now_cn().strftime("%Y%m%d%H"))

    def _p() -> dict:
        out = {"up_count": float("nan"), "down_count": float("nan"),
               "flat_count": float("nan"), "indices": [], "breadth_ratio": float("nan"),
               "breadth_source": "none", "source": "none", "detail": ""}
        url = (_EM_ULIST + "?fltt=2&secids=1.000001,0.399001,0.399006"
               "&fields=f2,f3,f4,f12,f14,f104,f105,f106")
        payload = retry_call(lambda: http_get(url, json_body=True), attempts=3, delay=0.9,
                             default=None, label="ulist_breadth")
        idx = _parse_ulist(payload)
        if idx:
            out["indices"] = [{"code": x["code"], "name": x["name"], "price": x["price"],
                               "pct": x["pct"]} for x in idx]
            up = [x["up"] for x in idx if not math.isnan(x["up"])]
            down = [x["down"] for x in idx if not math.isnan(x["down"])]
            flat = [x["flat"] for x in idx if not math.isnan(x["flat"])]
            if up and down:
                out["up_count"] = float(sum(up)) / len(up)
                out["down_count"] = float(sum(down)) / len(down)
                out["flat_count"] = float(sum(flat)) / len(flat) if flat else float("nan")
                out["breadth_source"] = "em_f104_f105"
            else:
                out["breadth_source"] = "index_proxy"
            d = [x["pct"] for x in idx if not math.isnan(x["pct"])]
            if d:
                out["index_pct_avg"] = float(np.mean(d))
            out["source"] = "eastmoney_ulist"
        else:
            out["breadth_source"] = "none"
            _note("指数快照取数失败（push2 间歇性拒连），涨跌家数不可得")
        # 涨停生态
        lu = _producer_limit_up()
        for k in ("limit_up_count", "limit_down_count", "max_continuous",
                  "continuous_count", "broken_rate", "boards_hot"):
            out[k] = lu.get(k, float("nan"))
        if lu.get("source") != "none":
            out["source"] = (out["source"] + "+" + str(lu.get("source"))).strip("+")
            out["limit_up_date"] = lu.get("date")
        # 涨跌家数缺失 → 指数代理
        if math.isnan(_num(out.get("up_count"))) and not math.isnan(_num(out.get("index_pct_avg"))):
            pct = _num(out["index_pct_avg"])
            out["breadth_source"] = "index_proxy"
            out["detail"] = ("涨跌家数字段不可得，用指数平均涨跌幅 %+.2f%% 代理市场宽度"
                             "（可信度已下调）" % pct)
        elif not math.isnan(_num(out.get("up_count"))):
            ratio = out["up_count"] / max(1.0, out["up_count"] + out["down_count"])
            out["breadth_ratio"] = ratio
            out["detail"] = ("上涨 %d 家 / 下跌 %d 家（上涨占比 %.0f%%），涨停 %s 家、"
                             "炸板率 %s" % (out["up_count"], out["down_count"], ratio * 100,
                                          _fmt(out.get("limit_up_count")),
                                          _fmt_pct(out.get("broken_rate"))))
        return out

    try:
        if force:
            return _p()
        out = cached(key, 600, _p)
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("market_breadth 异常降级：%r" % exc)
        return {}


def _fmt(x) -> str:
    v = _num(x)
    return "—" if math.isnan(v) else ("%d" % v)


def _fmt_pct(x) -> str:
    v = _num(x)
    return "—" if math.isnan(v) else ("%.1f%%" % (v * 100))


def fetch_market_sentiment(force=False) -> dict:
    """市场情绪总览（涨跌家数 + 涨停生态 + 指数 + 全市场两融 + VIX）。

    Returns
    -------
    dict
        :func:`market_breadth` 的全部字段，另加 ``index_pct_avg``、``vix``、
        ``vix_source``、``margin``（全市场两融，供 ``margin_euphoria``/``margin_trend``
        使用；调用方也可用个股两融覆盖它）、``as_of``。
    """
    key = cache_key("mktsent", now_cn().strftime("%Y%m%d%H"))

    def _p() -> dict:
        out = market_breadth(force=True) or {}
        vix = float("nan")
        vix_src = "none"
        try:      # 新浪 gb_$vix 实测返回空串；yfinance 在本机被限流 → 兜底用默认值
            txt = http_get("https://hq.sinajs.cn/list=gb_$vix", encoding="gbk",
                           headers={"Referer": "https://finance.sina.com.cn"})
            if txt and '="' in txt and len(txt.split('="', 1)[1].rstrip('";\n')) > 3:
                vix = _num(txt.split('="', 1)[1].split(",")[0])
                vix_src = "sina"
        except Exception as exc:  # noqa: BLE001
            log.debug("VIX 取数失败: %r", exc)
        if math.isnan(vix):
            vix = float(config.MACRO_DEFAULTS["vix"])
            vix_src = "default"
            _note("VIX 无可用实时源（新浪返回空、yfinance 限流），使用配置默认值 %.1f"
                  % vix)
        out["vix"] = vix
        out["vix_source"] = vix_src
        mm = fetch_market_margin(force=True)
        if mm:
            out["margin"] = mm
        out["as_of"] = now_cn().strftime("%Y-%m-%d %H:%M")
        return out

    try:
        if force:
            return _p()
        out = cached(key, 600, _p)
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("fetch_market_sentiment 异常降级：%r" % exc)
        return {}


def fetch_market_margin(force=False) -> dict:
    """全市场两融余额（沪市，``akshare.stock_margin_sse``）→ 杠杆情绪分位数据。

    Returns
    -------
    dict
        ``{"current","history","change_5d_pct","source"}``；失败返回 ``{}``。
        这是**市场层面**的杠杆水位（个股层面的两融见 :func:`core.capital.fetch_margin`）。
    """
    key = cache_key("mktmargin", now_cn().strftime("%Y%m%d"))

    def _p() -> dict:
        try:
            import akshare as ak
            end = now_cn().date()
            start = end - dt.timedelta(days=260)
            df = retry_call(lambda: ak.stock_margin_sse(start_date=start.strftime("%Y%m%d"),
                                                        end_date=end.strftime("%Y%m%d")),
                            attempts=2, delay=0.8, default=None, label="margin_sse")
            if isinstance(df, pd.DataFrame) and not df.empty and "融资余额" in df.columns:
                d = df.copy()
                d["信用交易日期"] = pd.to_datetime(d["信用交易日期"], errors="coerce")
                d = d.dropna(subset=["信用交易日期"]).sort_values("信用交易日期")
                s = pd.to_numeric(d["融资余额"], errors="coerce").dropna()
                if len(s) >= 20:
                    cur = float(s.iloc[-1])
                    chg5 = (cur / float(s.iloc[-6]) - 1.0) * 100.0 if len(s) >= 6 \
                        else float("nan")
                    _note("全市场两融来源：akshare.stock_margin_sse（%d 个交易日，"
                          "最新融资余额 %.2f 万亿）" % (len(s), cur / 1e12))
                    return {"current": cur, "history": [float(x) for x in s.tail(120)],
                            "change_5d_pct": chg5, "n": int(len(s)),
                            "source": "akshare.stock_margin_sse"}
            _note("全市场两融数据不可得（akshare.stock_margin_sse 返回空）")
        except Exception as exc:  # noqa: BLE001
            _note("全市场两融取数失败：%r" % exc)
        return {}

    try:
        if force:
            return _p()
        out = cached(key, config.RUNTIME["cache_ttl_daily"], _p)
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("fetch_market_margin 异常降级：%r" % exc)
        return {}


# --------------------------------------------------------------------------- #
# 2. 板块热度
# --------------------------------------------------------------------------- #
def _producer_sector_heat(symbol) -> dict:
    """所属概念/行业板块热度 + 风口判断 + 龙头股联动。"""
    out = {"boards": [], "hottest": None, "is_hot_theme": False, "theme_names": [],
           "leader_stock": None, "sector_rank_pct": float("nan"), "detail": "",
           "source": "none"}
    url = (_EM_SLIST + "?secid=%s&spt=3&fields=f12,f13,f14,f3,f62,f184&po=1&pn=1&pz=60"
           "&fid=f62&fltt=2" % _secid(symbol))
    payload = retry_call(lambda: http_get(url, json_body=True), attempts=3, delay=0.9,
                         default=None, label="slist_sector")
    boards = _parse_slist(payload)
    if not boards:
        _note("所属板块接口取数失败（push2 间歇性拒连），板块热度不可得")
        out["detail"] = "板块数据不可得"
        return out
    out["source"] = "eastmoney_slist"
    out["boards"] = boards
    hottest = max(boards, key=lambda b: (_num(b.get("main_net"), -1e18),
                                         _num(b.get("pct"), -1e9)))
    out["hottest"] = hottest
    # 风口题材：板块名命中 config.HOT_THEMES
    themes = []
    for b in boards:
        name = str(b.get("name") or "")
        hit = [t for t in config.HOT_THEMES if t and (t in name or name in t)]
        if hit:
            themes.append({"board": name, "code": b.get("code"),
                           "matched": hit, "pct": b.get("pct"),
                           "main_net": b.get("main_net")})
    out["theme_names"] = [t["board"] for t in themes]
    # 热点也包含"板块资金净流入明显"但名字未命中关键词的情况
    hot_by_flow = [b for b in boards if _num(b.get("main_net"), 0) > 5e8]
    out["is_hot_theme"] = bool(themes) or bool(hot_by_flow)
    # 板块内涨幅排名 → 龙头股
    try:
        burl = (_EM_CLIST + "?pn=1&pz=50&po=1&np=1&fltt=2&invt=2&fid=f3&fs=b:%s"
                "&fields=f12,f14,f3,f20" % hottest.get("code"))
        bpayload = retry_call(lambda: http_get(burl, json_body=True), attempts=3, delay=0.9,
                              default=None, label="clist_leader")
        diff = ((bpayload or {}).get("data") or {}).get("diff") or []
        items = list(diff.values()) if isinstance(diff, dict) else list(diff)
        items = [x for x in items if str(x.get("f12")) != _code(symbol)]
        if items:
            leader = max(items, key=lambda x: _num(x.get("f3"), -999))
            out["leader_stock"] = {"code": str(leader.get("f12")), "name": leader.get("f14"),
                                   "pct": _num(leader.get("f3")),
                                   "market_cap": _num(leader.get("f20"))}
    except Exception as exc:  # noqa: BLE001
        _note("龙头股识别失败：%r" % exc)
    # 该股所属板块涨幅分位
    try:
        pcts = sorted([_num(b.get("pct")) for b in boards if not math.isnan(_num(b.get("pct")))])
        hot_pct = _num(hottest.get("pct"))
        if pcts and not math.isnan(hot_pct):
            out["sector_rank_pct"] = float(percentile_rank(pcts, hot_pct))
    except Exception:  # noqa: BLE001
        pass
    out["detail"] = ("最热板块：%s（涨幅 %s，主力净流入 %s）；命中风口题材 %s"
                     % (hottest.get("name"), _fmt_pct(_num(hottest.get("pct")) / 100.0),
                        _fmt_yi(hottest.get("main_net")),
                        "、".join(out["theme_names"]) if out["theme_names"] else "无"))
    if out.get("leader_stock"):
        out["detail"] += "；板块龙头 %s(%s) %+.2f%%" % (
            out["leader_stock"]["name"], out["leader_stock"]["code"],
            _num(out["leader_stock"]["pct"]))
    _note("板块热度：%d 个所属板块，%s" % (len(boards), out["detail"]))
    return out


def _fmt_yi(x) -> str:
    v = _num(x)
    return "—" if math.isnan(v) else ("%+.2f 亿" % (v / 1e8))


def fetch_sector_heat(symbol, force=False) -> dict:
    """板块热度与风口判断。

    Returns
    -------
    dict
        ``{"boards":[{"code","name","pct","main_net","main_pct"}...], "hottest":{...},
        "is_hot_theme":bool, "theme_names":[...], "leader_stock":{...},
        "sector_rank_pct":0~1, "detail"}``；失败返回带空值的 dict（不抛异常）。
    """
    code = _code(symbol)
    key = cache_key("sector", code)
    try:
        if force:
            return _producer_sector_heat(symbol) or {}
        out = cached(key, config.RUNTIME["cache_ttl_seconds"],
                     lambda: _producer_sector_heat(symbol))
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("fetch_sector_heat 异常降级：%r" % exc)
        return {"boards": [], "is_hot_theme": False, "theme_names": [],
                "sector_rank_pct": float("nan"), "detail": "取数异常，已降级"}


# --------------------------------------------------------------------------- #
# 3. 散户情绪（反向）
# --------------------------------------------------------------------------- #
def _producer_retail(symbol) -> dict:
    """散户关注度与交易热度（股吧人气 → 搜索热度 → 换手率/量比代理）。"""
    out = {"guba_rank": None, "guba_rank_change": None, "search_hits": None,
           "search_hits_change_pct": None, "turnover": float("nan"),
           "turnover_pct_rank": float("nan"), "volume_ratio": float("nan"),
           "retail_score": float("nan"), "source": "none", "detail": ""}
    # ---- 股吧人气排名（实测返回空列表，属正常降级） ---------------------- #
    try:
        from .utils import session
        body = {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38",
                "marketType": "", "srcSecurityCode": _code(symbol), "pageNo": 1,
                "pageSize": 100}
        r = session().post(_GUBA_RANK, json=body, timeout=config.RUNTIME["http_timeout"])
        data = (r.json() or {}).get("data") or []
        for it in data:
            if str(it.get("sc") or it.get("code") or "").endswith(_code(symbol)):
                out["guba_rank"] = _num(it.get("rk") or it.get("rank"))
        if out["guba_rank"] is None:
            _note("股吧人气排名接口返回空（实测 data=[]），散户人气字段留空")
    except Exception as exc:  # noqa: BLE001
        _note("股吧人气排名失败：%r" % exc)
    # ---- 搜索热度（实测返回与关键词无关的固定内容，视为不可靠） ---------- #
    try:
        out["search_hits"] = None
        _note("东财搜索接口返回内容与关键词无关（沙箱/网关固定响应），"
              "search_hits 置空并降级到换手率代理")
    except Exception:  # noqa: BLE001
        pass
    # ---- 代理指标：换手率 / 量比 ----------------------------------------- #
    try:
        txt = http_get(_TENCENT_URL % _prefix(symbol), encoding="gbk")
        if txt and '="' in txt:
            p = txt.split('="', 1)[1].rsplit('"', 1)[0].split("~")
            for k, i in _TX_IDX.items():
                if i < len(p) and k in ("turnover", "volume_ratio", "price", "pct_chg",
                                        "amplitude"):
                    out[k] = _num(p[i])
            out["source"] = "tencent"
    except Exception as exc:  # noqa: BLE001
        _note("腾讯快照失败：%r" % exc)
    # 换手率的历史分位由调用方用 frame 计算（见 fetch_retail_sentiment）
    if not math.isnan(_num(out.get("turnover"))):
        out["detail"] = ("换手率 %.2f%%、量比 %.2f，%s"
                         % (_num(out["turnover"]), _num(out.get("volume_ratio")),
                            "交易热度偏高" if _num(out["turnover"]) > 3 else "交易热度正常"))
    else:
        out["detail"] = "散户情绪数据不可得"
    return out


def fetch_retail_sentiment(symbol, force=False) -> dict:
    """散户情绪（**反向指标**数据源）。

    Returns
    -------
    dict
        ``{"guba_rank","guba_rank_change","search_hits","search_hits_change_pct",
        "turnover","turnover_pct_rank","volume_ratio","retail_score","source","detail"}``。
        股吧/搜索不可得时留 ``None``，但会用腾讯快照的换手率/量比补出 ``retail_score``。
        **人气排名骤升 / 换手率暴增通常对应短期见顶**。
    """
    code = _code(symbol)
    key = cache_key("retail", code)
    try:
        if force:
            return _producer_retail(symbol) or {}
        out = cached(key, config.RUNTIME["cache_ttl_seconds"],
                     lambda: _producer_retail(symbol))
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _note("fetch_retail_sentiment 异常降级：%r" % exc)
        return {"source": "none", "detail": "取数异常，已降级"}


# --------------------------------------------------------------------------- #
# 4. 恐惧贪婪指数
# --------------------------------------------------------------------------- #
def margin_sentiment(margin) -> dict:
    """把两融数据归一成情绪模块需要的结构（兼容 DataFrame 与 dict）。

    Parameters
    ----------
    margin : pd.DataFrame | dict | None
        * DataFrame：``core.capital.fetch_margin`` 的产出（需含 ``融资余额`` 列）；
        * dict：``{"current":float,"history":[float,...],"change_5d_pct":float}``
          或 ``{"margin": <上述之一>}``。

    Returns
    -------
    dict
        ``{"current","history","change_5d_pct","ratio_pct","n","source"}``；失败返回 ``{}``。
        情绪模块用它判断"两融余额历史分位（杠杆拥挤度）"与"5 日趋势"。
    """
    try:
        if isinstance(margin, dict) and "margin" in margin:
            margin = margin.get("margin")
        if margin is None:
            return {}
        if isinstance(margin, pd.DataFrame) and not margin.empty:
            df = margin
            col = "融资余额" if "融资余额" in df.columns else None
            if col is None:
                return {}
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                return {}
            hist = [float(x) for x in s.tail(120).tolist()]
            cur = float(s.iloc[-1])
            chg5 = (cur / float(s.iloc[-6]) - 1.0) * 100.0 if len(s) >= 6 else float("nan")
            ratio = float("nan")
            for c in ("融资余额占流通市值比", "占流通市值比"):
                if c in df.columns:
                    ratio = _num(df[c].dropna().iloc[-1]) if df[c].notna().any() else float("nan")
                    break
            return {"current": cur, "history": hist, "change_5d_pct": chg5,
                    "ratio_pct": ratio, "n": len(s), "source": "capital.fetch_margin"}
        if isinstance(margin, dict):
            hist = [float(x) for x in (margin.get("history") or []) if x == x]
            cur = _num(margin.get("current"))
            chg5 = _num(margin.get("change_5d_pct"))
            if hist and not math.isnan(cur) and math.isnan(chg5) and len(hist) >= 6:
                chg5 = (cur / hist[-6] - 1.0) * 100.0 if hist[-6] else float("nan")
            if math.isnan(cur) and hist:
                cur = hist[-1]
            if math.isnan(cur):
                return {}
            return {"current": cur, "history": hist, "change_5d_pct": chg5,
                    "ratio_pct": _num(margin.get("ratio_pct")), "n": len(hist),
                    "source": str(margin.get("source") or "dict")}
    except Exception as exc:  # noqa: BLE001
        _note("两融数据归一失败：%r" % exc)
    return {}


def _vol_annual_pct(close: pd.Series, window: int = 20) -> float:
    """近 ``window`` 日年化波动率（%）。"""
    try:
        r = pd.Series(close).astype(float).pct_change().dropna().tail(window)
        if len(r) < 5:
            return float("nan")
        return float(r.std(ddof=1) * math.sqrt(config.RUNTIME["trading_days_per_year"]) * 100)
    except Exception:  # noqa: BLE001
        return float("nan")


def fear_greed_index(symbol=None, frame=None, market=None) -> dict:
    """恐惧贪婪指数（0~100，越高越贪婪）。

    分项与权重（见 :data:`FEAR_GREED_WEIGHTS`）：

    ==========================  ======  ==========================================
    分项                          权重    说明
    ==========================  ======  ==========================================
    ``new_high_ratio``           0.20    个股 20 日创 20 日新高占比（越高越贪婪）
    ``turnover_percentile``      0.25    当日换手率在历史换手率中的分位
    ``margin_percentile``        0.20    两融余额在其历史中的分位（杠杆拥挤度）
    ``breadth_ratio``            0.20    大盘涨跌家数比（上涨占比）
    ``vol_percentile_inv``       0.15    20 日年化波动率的历史分位（**反向**：高波动=恐惧）
    ==========================  ======  ==========================================

    缺失子项按剩余权重归一化；等级：<20 极度恐惧 / 20~40 恐惧 / 40~60 中性 /
    60~80 贪婪 / >80 极度贪婪。

    Returns
    -------
    dict
        ``{"value","level","components","detail"}``。
    """
    comp: dict = {}
    try:
        f = _ensure_frame(frame)
        # ---- 创新高占比 --------------------------------------------------- #
        if f is not None and "close" in f.columns:
            c = pd.to_numeric(f["close"], errors="coerce").dropna()
            if len(c) >= 25:
                roll_max = c.rolling(20, min_periods=20).max()
                tail = c.tail(20)
                hits = int((tail >= roll_max.tail(20) - 1e-9).sum())
                comp["new_high_ratio"] = hits / 20.0
        # ---- 换手率分位 --------------------------------------------------- #
        if f is not None and "turnover" in f.columns:
            t = pd.to_numeric(f["turnover"], errors="coerce").dropna()
            if len(t) >= 30:
                comp["turnover_percentile"] = float(percentile_rank(t.values, float(t.iloc[-1])))
        # ---- 两融分位 ----------------------------------------------------- #
        mk = market if isinstance(market, dict) else {}
        mg = margin_sentiment(mk.get("margin"))
        if mg:
            hist = [x for x in (mg.get("history") or []) if x == x]
            cur = _num(mg.get("current"))
            if hist and not math.isnan(cur):
                comp["margin_percentile"] = float(percentile_rank(hist, cur))
        # ---- 涨跌家数比 --------------------------------------------------- #
        up, down = _num(mk.get("up_count")), _num(mk.get("down_count"))
        if not math.isnan(up) and not math.isnan(down) and (up + down) > 0:
            comp["breadth_ratio"] = float(up / (up + down))
        elif not math.isnan(_num(mk.get("breadth_ratio"))):
            comp["breadth_ratio"] = float(_num(mk["breadth_ratio"]))
        # ---- 波动率分位（反向） ------------------------------------------- #
        if f is not None and "close" in f.columns:
            c = pd.to_numeric(f["close"], errors="coerce").dropna()
            if len(c) >= 80:
                ret = c.pct_change().dropna()
                vols = ret.rolling(20).std().dropna() * math.sqrt(
                    config.RUNTIME["trading_days_per_year"]) * 100
                if len(vols) >= 30:
                    cur_v = float(vols.iloc[-1])
                    pct = float(percentile_rank(vols.values, cur_v))
                    comp["vol_percentile_inv"] = 1.0 - pct
                    comp["annual_vol_pct"] = cur_v
    except Exception as exc:  # noqa: BLE001
        _note("恐惧贪婪分项计算异常：%r" % exc)

    valid = {k: v for k, v in comp.items() if k in FEAR_GREED_WEIGHTS and v == v}
    if not valid:
        return {"value": 50.0, "level": "中性", "components": comp,
                "detail": "缺少行情/两融/涨跌家数数据，恐惧贪婪指数取中性 50"}
    wsum = sum(FEAR_GREED_WEIGHTS[k] for k in valid)
    value = sum(valid[k] * FEAR_GREED_WEIGHTS[k] for k in valid) / wsum * 100.0
    level = ("极度恐惧" if value < 20 else "恐惧" if value < 40 else "中性" if value < 60
             else "贪婪" if value < 80 else "极度贪婪")
    detail = ("恐惧贪婪指数 %.0f（%s）；分项：" % (value, level) + "、".join(
        "%s=%.2f" % (k, v) for k, v in valid.items())
        + ("；缺失分项：%s" % "、".join(k for k in FEAR_GREED_WEIGHTS if k not in valid)
           if len(valid) < len(FEAR_GREED_WEIGHTS) else ""))
    return {"value": round(float(value), 1), "level": level, "components": comp,
            "detail": detail}


# --------------------------------------------------------------------------- #
# 5. 因子合成
# --------------------------------------------------------------------------- #
def sentiment_factors(symbol, market=None, sector=None, retail=None, frame=None,
                      snapshot=None) -> list:
    """产出情绪面因子列表。

    Parameters
    ----------
    market : dict | None
        :func:`fetch_market_sentiment` 的产出（含 ``up_count/down_count/limit_up_count/
        broken_rate/max_continuous/vix`` 等）。两融数据可放在 ``market["margin"]``
        或 ``snapshot["margin"]``，既可以是 :func:`core.capital.fetch_margin` 的
        DataFrame，也可以是 :func:`margin_sentiment` 归一后的 dict。
    sector : dict | None
        :func:`fetch_sector_heat` 的产出。
    retail : dict | None
        :func:`fetch_retail_sentiment` 的产出。
    frame : pd.DataFrame | None
        个股日线（算换手率分位、波动率区间）。
    snapshot : dict | None
        个股快照（``turnover``/``volume_ratio``）。

    Returns
    -------
    list[Factor]
        全空输入时返回中性 + 低可信度因子，**不抛异常**。
        ``CONTRARIAN_FACTOR_IDS`` 中的因子按"原始看涨信号"写 direction，由引擎翻转。
    """
    fs: list = []
    try:
        mk = market if isinstance(market, dict) else {}
        snap = snapshot if isinstance(snapshot, dict) else {}
        # 两融数据可以由调用方以 DataFrame / dict 形式塞进 market 或 snapshot
        mg_in = margin_sentiment(mk.get("margin")) or margin_sentiment(snap.get("margin"))
        if mg_in:
            mk = dict(mk)
            mk["margin"] = mg_in
        sec = sector if isinstance(sector, dict) else {}
        ret = retail if isinstance(retail, dict) else {}
        f = _ensure_frame(frame)

        # ---- 1) 市场宽度 -------------------------------------------------- #
        up, down = _num(mk.get("up_count")), _num(mk.get("down_count"))
        br = _num(mk.get("breadth_ratio"))
        proxy = mk.get("breadth_source") == "index_proxy"
        if not math.isnan(up) and not math.isnan(down) and (up + down) > 0:
            br = up / (up + down)
        if math.isnan(br) and not math.isnan(_num(mk.get("index_pct_avg"))):
            br = clamp(0.5 + _num(mk["index_pct_avg"]) / 4.0, 0, 1)
        if math.isnan(br):
            fs.append(_neutral("market_breadth", "市场宽度", "涨跌家数数据不可得",
                               0.15 if not proxy else 0.3, theory="市场宽度理论",
                               horizon=SENT_HORIZON))
        else:
            score = (br - 0.5) * 2.0
            fs.append(Factor("market_breadth", "市场宽度", Category.SENTIMENT,
                             DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                             clamp(abs(score), 0, 1), 0.4 if proxy else 0.8,
                             value={"上涨占比": round(br, 3), "代理": proxy},
                             detail="全市场上涨家数占比 %.0f%%（%s）"
                                    % (br * 100, "指数代理，可信度下调" if proxy
                                       else "赚钱效应%s" % ("偏强" if br > 0.5 else "偏弱")),
                             theory="市场宽度理论", horizon=SENT_HORIZON))

        # ---- 2) 涨停狂热（反向指标） -------------------------------------- #
        lu = _num(mk.get("limit_up_count"))
        cont = _num(mk.get("continuous_count"))
        if math.isnan(lu):
            fs.append(_neutral("limit_up_frenzy", "涨停情绪",
                               "涨停池数据不可得" + CONTRARIAN_HINT, 0.2,
                               theory="情绪周期理论", horizon=SENT_HORIZON))
        else:
            # 原始看涨信号：涨停家数越多、连板越多 → 情绪越亢奋（反向指标）
            score = clamp((lu - 40.0) / 60.0, -1, 1) * 0.7 + \
                clamp((0.0 if math.isnan(cont) else cont) / 30.0, -1, 1) * 0.3
            fs.append(Factor("limit_up_frenzy", "涨停情绪", Category.SENTIMENT,
                             DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.2 else 0),
                             clamp(abs(score), 0, 1), 0.7,
                             value={"涨停家数": lu, "连板家数": cont},
                             detail="涨停 %s 家、连板 %s 家，市场情绪%s%s"
                                    % (_fmt(lu), _fmt(cont),
                                       "亢奋" if score > 0.1 else
                                       ("冰点" if score < -0.2 else "中性"),
                                       CONTRARIAN_HINT),
                             theory="情绪周期理论（涨停潮后易退潮）",
                             horizon=SENT_HORIZON))

        # ---- 3) 板块热度 -------------------------------------------------- #
        boards = sec.get("boards") if isinstance(sec.get("boards"), list) else []
        hot = sec.get("hottest") if isinstance(sec.get("hottest"), dict) else None
        if not boards or not hot:
            fs.append(_neutral("sector_heat", "板块热度", "板块数据不可得", 0.2,
                               theory="板块轮动理论", horizon=SENT_HORIZON))
        else:
            flow = _num(hot.get("main_net"))
            hpct = _num(hot.get("pct"))
            score = clamp(flow / 5e8, -1, 1) * 0.6 + clamp(hpct / 4.0, -1, 1) * 0.4
            theme = bool(sec.get("is_hot_theme"))
            if theme:
                score += 0.15
            fs.append(Factor("sector_heat", "板块热度", Category.SENTIMENT,
                             DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                             clamp(abs(score), 0, 1), 0.7,
                             value={"板块": hot.get("name"), "涨幅": hpct,
                                    "主力净额": flow,
                                    "风口题材": sec.get("theme_names")},
                             detail="所属最热板块「%s」%s，主力净流入 %s%s"
                                    % (hot.get("name"), _fmt_pct(hpct / 100.0),
                                       _fmt_yi(flow),
                                       "；处于政策/市场风口（%s）"
                                       % "、".join(sec.get("theme_names") or []) if theme
                                       else "；不在热门题材风口"),
                             theory="板块轮动理论", horizon=SENT_HORIZON))

        # ---- 4) 龙头股联动 ------------------------------------------------ #
        leader = sec.get("leader_stock") if isinstance(sec.get("leader_stock"), dict) else None
        rank_pct = _num(sec.get("sector_rank_pct"))
        if not leader:
            fs.append(_neutral("sector_leader_linkage", "龙头股联动",
                               "未识别到板块龙头" if boards else "板块数据不可得", 0.2,
                               theory="龙头效应", horizon=SENT_HORIZON))
        else:
            lp = _num(leader.get("pct"))
            score = clamp(lp / 8.0, -1, 1) * 0.6 + \
                ((rank_pct - 0.5) * 2.0 if not math.isnan(rank_pct) else 0.0) * 0.4
            fs.append(Factor("sector_leader_linkage", "龙头股联动", Category.SENTIMENT,
                             DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                             clamp(abs(score), 0, 1), 0.55,
                             value={"龙头": leader.get("name"), "龙头涨幅": lp,
                                    "板块分位": rank_pct},
                             detail="板块龙头 %s(%s) %+.2f%%，本股所属板块热度分位 %s，%s"
                                    % (leader.get("name"), leader.get("code"), lp,
                                       "%.0f%%" % (rank_pct * 100)
                                       if not math.isnan(rank_pct) else "—",
                                       "板块内有赚钱效应" if score > 0.1 else
                                       ("板块联动走弱" if score < -0.1 else "联动中性")),
                             theory="龙头效应 / 板块联动", horizon=SENT_HORIZON))

        # ---- 5) 两融情绪（反向） + 两融趋势 ------------------------------- #
        mg = margin_sentiment(mk.get("margin"))
        if not mg:
            fs.append(_neutral("margin_euphoria", "两融狂热" + CONTRARIAN_HINT,
                               "两融数据不可得", 0.2, theory="杠杆情绪",
                               horizon=SENT_HORIZON))
            fs.append(_neutral("margin_trend", "两融趋势", "两融数据不可得", 0.2,
                               theory="杠杆资金", horizon=SENT_HORIZON_LONG))
        else:
            hist = [x for x in (mg.get("history") or []) if x == x]
            cur = _num(mg.get("current"))
            pct = float(percentile_rank(hist, cur)) if hist and not math.isnan(cur) \
                else float("nan")
            if math.isnan(pct):
                fs.append(_neutral("margin_euphoria", "两融狂热" + CONTRARIAN_HINT,
                                   "两融历史样本不足", 0.25, theory="杠杆情绪",
                                   horizon=SENT_HORIZON))
            else:
                fs.append(Factor("margin_euphoria", "两融狂热", Category.SENTIMENT,
                                 DIR_BULL if pct > 0.6 else (DIR_BEAR if pct < 0.3 else 0),
                                 clamp(abs(pct - 0.5) * 2.0, 0, 1), 0.6, value=pct,
                                 detail="两融余额处于历史 %.0f%% 分位，%s%s"
                                        % (pct * 100,
                                           "杠杆拥挤、情绪过热" if pct > 0.6 else
                                           ("杠杆低位、情绪偏冷" if pct < 0.3 else "杠杆中性"),
                                           CONTRARIAN_HINT),
                                 theory="杠杆情绪（高位反转）", horizon=SENT_HORIZON))
            chg = _num(mg.get("change_5d_pct"))
            if math.isnan(chg):
                fs.append(_neutral("margin_trend", "两融趋势", "缺少两融变化数据", 0.25,
                                   theory="杠杆资金", horizon=SENT_HORIZON_LONG))
            else:
                score = clamp(chg / 5.0, -1, 1)
                fs.append(Factor("margin_trend", "两融趋势", Category.SENTIMENT,
                                 DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                                 clamp(abs(score), 0, 1), 0.6, value=chg,
                                 detail="两融余额近 5 日 %+.2f%%，杠杆资金%s"
                                        % (chg, "加仓" if chg > 0 else "撤离"),
                                 theory="杠杆资金", horizon=SENT_HORIZON_LONG))

        # ---- 6) 换手率 / 换手狂热（反向） --------------------------------- #
        turns = []
        if f is not None and "turnover" in f.columns:
            turns = [float(x) for x in pd.to_numeric(f["turnover"], errors="coerce")
                     .dropna().tolist()]
        cur_turn = _num(snap.get("turnover"))
        if math.isnan(cur_turn):
            cur_turn = _num(ret.get("turnover"))
        if math.isnan(cur_turn) and turns:
            cur_turn = turns[-1]
        turn_pct = float(percentile_rank(turns, cur_turn)) if (turns and not math.isnan(cur_turn)) \
            else _num(ret.get("turnover_pct_rank"))
        if math.isnan(cur_turn) and math.isnan(turn_pct):
            fs.append(_neutral("turnover_euphoria", "换手狂热" + CONTRARIAN_HINT,
                               "换手率数据不可得", 0.2, theory="换手率理论",
                               horizon=SENT_HORIZON))
        else:
            score = 0.0
            if not math.isnan(turn_pct):
                score += (turn_pct - 0.5) * 2.0 * 0.7
            if not math.isnan(cur_turn):
                score += clamp((cur_turn - 3.0) / 7.0, -1, 1) * 0.3
            fs.append(Factor("turnover_euphoria", "换手狂热", Category.SENTIMENT,
                             DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.2 else 0),
                             clamp(abs(score), 0, 1), 0.65,
                             value={"换手率": cur_turn, "历史分位": turn_pct},
                             detail="当前换手率 %s（历史分位 %s），%s%s"
                                    % ("%.2f%%" % cur_turn if not math.isnan(cur_turn) else "—",
                                       "%.0f%%" % (turn_pct * 100)
                                       if not math.isnan(turn_pct) else "—",
                                       "交易过热、筹码快速换手" if score > 0.1 else
                                       ("交易清淡" if score < -0.2 else "换手正常"),
                                       CONTRARIAN_HINT),
                             theory="换手率理论（高位换手见顶）", horizon=SENT_HORIZON))

        # ---- 7) 散户情绪 / 股吧 / 搜索（均为反向指标） -------------------- #
        rs = _num(ret.get("retail_score"))
        guba = _num(ret.get("guba_rank"))
        gchg = _num(ret.get("guba_rank_change"))
        hits = _num(ret.get("search_hits"))
        hchg = _num(ret.get("search_hits_change_pct"))
        # retail_score：综合人气/搜索/换手分位（拿不到股吧时用换手分位兜底）
        if math.isnan(rs):
            if not math.isnan(turn_pct):
                rs = turn_pct
        if math.isnan(rs) and math.isnan(guba) and math.isnan(hits):
            fs.append(_neutral("retail_sentiment_contrarian",
                               "散户情绪" + CONTRARIAN_HINT, "散户情绪数据不可得", 0.2,
                               theory="行为金融学（散户反向）", horizon=SENT_HORIZON))
        else:
            score = 0.0
            if not math.isnan(rs):
                score += (rs - 0.5) * 2.0
            if not math.isnan(guba):
                score += clamp((300.0 - guba) / 300.0 - 0.5, -1, 1) * 0.4
            fs.append(Factor("retail_sentiment_contrarian", "散户情绪",
                             Category.SENTIMENT,
                             DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.2 else 0),
                             clamp(abs(score), 0, 1), 0.45,
                             value={"retail_score": rs, "guba_rank": None if math.isnan(guba)
                                    else int(guba)},
                             detail="散户关注度综合分 %.2f（%s）%s"
                                    % (rs if not math.isnan(rs) else float("nan"),
                                       "关注度极高" if score > 0.1 else
                                       ("关注度低迷" if score < -0.2 else "关注度中性"),
                                       CONTRARIAN_HINT),
                             theory="行为金融学（散户反向指标）", horizon=SENT_HORIZON))
        if math.isnan(guba):
            fs.append(_neutral("guba_bull_ratio", "股吧看多比例" + CONTRARIAN_HINT,
                               "股吧人气接口无数据（实测返回空）", 0.15,
                               theory="行为金融学（散户反向）", horizon=SENT_HORIZON))
        else:
            score = clamp((300.0 - guba) / 300.0 - 0.5, -1, 1) * 2.0
            fs.append(Factor("guba_bull_ratio", "股吧看多比例", Category.SENTIMENT,
                             DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.2 else 0),
                             clamp(abs(score), 0, 1), 0.4,
                             value={"guba_rank": int(guba), "rank_change": gchg
                                    if not math.isnan(gchg) else None},
                             detail="股吧人气排名第 %d 名（%s）%s"
                                    % (int(guba), "人气骤升，短期易见顶"
                                       if score > 0.1 else "人气平稳", CONTRARIAN_HINT),
                             theory="行为金融学（散户反向指标）", horizon=SENT_HORIZON))
        if math.isnan(hits) and math.isnan(hchg):
            fs.append(_neutral("search_heat_spike", "搜索热度" + CONTRARIAN_HINT,
                               "搜索热度接口不可用（返回与关键词无关的固定响应）", 0.15,
                               theory="行为金融学（注意力效应）", horizon=SENT_HORIZON))
        else:
            score = clamp(_num(hchg, 0.0) / 100.0, -1, 1)
            fs.append(Factor("search_heat_spike", "搜索热度", Category.SENTIMENT,
                             DIR_BULL if score > 0.1 else DIR_NEUTRAL,
                             clamp(abs(score), 0, 1), 0.4,
                             value={"hits": hits, "change_pct": hchg},
                             detail="搜索热度变化 %s，%s%s"
                                    % (_fmt_pct(_num(hchg, 0) / 100.0),
                                       "关注度骤升（短期见顶风险）" if score > 0.1 else "关注度平稳",
                                       CONTRARIAN_HINT),
                             theory="注意力效应（搜索量见顶）", horizon=SENT_HORIZON))

        # ---- 8) 恐惧贪婪（模块内部已反向编码） ---------------------------- #
        fg = fear_greed_index(symbol, frame=frame, market=mk)
        val = _num(fg.get("value"), 50.0)
        score = (50.0 - val) / 50.0      # 极度恐惧(<20) → +1 看涨；极度贪婪(>80) → -1
        fs.append(Factor("fear_greed", "恐惧贪婪指数", Category.SENTIMENT,
                         DIR_BULL if score > 0.1 else (DIR_BEAR if score < -0.1 else 0),
                         clamp(abs(score), 0, 1),
                         0.6 if fg.get("components") else 0.25,
                         value="%.0f(%s)" % (val, fg.get("level")),
                         detail="%s；恐慌时贪婪、贪婪时恐慌（本因子已按反向编码）"
                                % fg.get("detail", ""),
                         theory="行为金融学（恐惧贪婪逆向）", horizon=SENT_HORIZON_LONG))

        # ---- 9) 波动率区间 ------------------------------------------------ #
        vol = _num(fg.get("components", {}).get("annual_vol_pct"))
        if math.isnan(vol) and f is not None and "close" in f.columns:
            vol = _vol_annual_pct(pd.to_numeric(f["close"], errors="coerce").dropna())
        if math.isnan(vol):
            fs.append(_neutral("volatility_regime", "波动率区间", "缺少价格数据", 0.2,
                               theory="波动率理论", horizon=SENT_HORIZON))
        else:
            high = vol > float(config.STAGE_RULES["high_vol_annual_pct"])
            low = vol < float(config.STAGE_RULES["low_vol_annual_pct"])
            fs.append(Factor("volatility_regime", "波动率区间", Category.SENTIMENT,
                             DIR_NEUTRAL, 0.0, 0.7, value=vol,
                             detail="20 日年化波动率 %.1f%%，%s（高波动区间方向判断可信度"
                                    "下降，已对情绪类因子做权重衰减）"
                                    % (vol, "属高波动区间" if high else
                                       ("属低波动区间" if low else "波动率中性")),
                             theory="波动率理论", horizon=SENT_HORIZON,
                             weight_multiplier=float(config.FORECAST["high_vol_attenuation"])
                             if high else 1.0))
    except Exception as exc:  # noqa: BLE001
        _note("sentiment_factors 内部异常（已降级）：%r" % exc)
        fs.append(_neutral("market_breadth", "市场宽度", "情绪因子生成异常，已降级", 0.1))
    return fs


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _fake_market() -> dict:
    return {
        "up_count": 3120.0, "down_count": 1750.0, "flat_count": 180.0,
        "limit_up_count": 68.0, "limit_down_count": 6.0, "max_continuous": 5.0,
        "continuous_count": 14.0, "broken_rate": 0.17, "breadth_ratio": 0.64,
        "breadth_source": "em_f104_f105", "index_pct_avg": 0.42, "vix": 16.0,
        "margin": {"current": 1.72e10, "history": list(np.linspace(1.2e10, 1.8e10, 120)),
                   "change_5d_pct": 2.1},
        "indices": [{"code": "000001", "name": "上证指数", "price": 3813.79, "pct": 0.05}],
    }


def _fake_sector() -> dict:
    return {"boards": [{"code": "BK1277", "name": "白酒Ⅱ", "pct": 0.82, "main_net": 2.02e8},
                       {"code": "BK0438", "name": "食品饮料", "pct": 1.46, "main_net": 5.7e7},
                       {"code": "BK0596", "name": "融资融券", "pct": 0.58,
                        "main_net": -1.77e10}],
            "hottest": {"code": "BK1277", "name": "白酒Ⅱ", "pct": 0.82, "main_net": 2.02e8},
            "is_hot_theme": True, "theme_names": ["白酒Ⅱ"],
            "leader_stock": {"code": "000858", "name": "五粮液", "pct": 3.2},
            "sector_rank_pct": 0.67, "detail": "selftest"}


def _fake_retail() -> dict:
    return {"guba_rank": None, "guba_rank_change": None, "search_hits": None,
            "search_hits_change_pct": None, "turnover": 0.28, "turnover_pct_rank": 0.42,
            "volume_ratio": 1.18, "retail_score": 0.42, "source": "tencent",
            "detail": "selftest"}


def _zt_payload() -> dict:
    """合成涨停池响应（字段与实测一致）。"""
    return {"rc": 0, "data": {"tc": 4, "qdate": 20261009, "pool": [
        {"c": "002058", "m": 0, "n": "紫竹高科", "p": 24440, "zdp": 9.99, "amount": 1.0e8,
         "ltsz": 3.5e9, "tshare": 3.5e9, "hs": 2.87, "lbc": 4, "fbt": 92500, "lbt": 92500,
         "fund": 8.9e7, "zbc": 0, "hybk": "电池", "zttj": {"days": 4, "ct": 4}},
        {"c": "600241", "m": 1, "n": "时代万恒", "p": 11760, "zdp": 10.0, "amount": 6.8e7,
         "ltsz": 3.4e9, "tshare": 3.4e9, "hs": 1.99, "lbc": 5, "fbt": 92500, "lbt": 92500,
         "fund": 2.29e8, "zbc": 0, "hybk": "电池", "zttj": {"days": 5, "ct": 5}},
        {"c": "605303", "m": 1, "n": "园林股份", "p": 32210, "zdp": 2.16, "amount": 8.5e8,
         "ltsz": 5.2e9, "tshare": 5.2e9, "hs": 16.3, "lbc": 1, "fbt": 92502, "lbt": 92502,
         "fund": 1.0e7, "zbc": 1, "hybk": "基础建设", "zttj": {"days": 4, "ct": 3}},
        {"c": "000001", "m": 0, "n": "平安银行", "p": 11590, "zdp": 10.01, "amount": 2.0e9,
         "ltsz": 2.2e11, "tshare": 2.2e11, "hs": 0.9, "lbc": 1, "fbt": 93000, "lbt": 93000,
         "fund": 3.0e8, "zbc": 0, "hybk": "银行", "zttj": {"days": 1, "ct": 1}}]}}


def _zb_payload() -> dict:
    return {"rc": 0, "data": {"tc": 2, "qdate": 20261009, "pool": [
        {"c": "605303", "m": 1, "n": "园林股份", "p": 32210, "ztp": 34680, "zdp": 2.16,
         "amount": 8.5e8, "ltsz": 5.2e9, "tshare": 5.2e9, "hs": 16.3, "fbt": 92502,
         "zbc": 1, "zf": 17.76, "zs": 0.0, "zttj": {"days": 4, "ct": 3}, "hybk": "基础建设"},
        {"c": "600000", "m": 1, "n": "浦发银行", "p": 12000, "ztp": 13000, "zdp": 5.0,
         "amount": 1.0e9, "ltsz": 3.0e11, "tshare": 3.0e11, "hs": 0.6, "fbt": 100000,
         "zbc": 1, "zf": 6.0, "zs": 0.0, "zttj": {"days": 1, "ct": 1}, "hybk": "银行"}]}}


def _ulist_payload() -> dict:
    return {"rc": 0, "data": {"total": 2, "diff": [
        {"f2": 3813.79, "f3": 0.05, "f4": 1.89, "f12": "000001", "f14": "上证指数",
         "f104": 3120, "f105": 1750, "f106": 180},
        {"f2": 12641.86, "f3": 0.17, "f4": 20.96, "f12": "399001", "f14": "深证成指",
         "f104": 3800, "f105": 2100, "f106": 220}]}}


def _slist_payload() -> dict:
    return {"rc": 0, "data": {"total": 3, "diff": {
        "0": {"f3": 82, "f12": "BK1277", "f13": 90, "f14": "白酒Ⅱ", "f62": 2.02e8,
              "f184": 284},
        "1": {"f3": 146, "f12": "BK0438", "f13": 90, "f14": "食品饮料", "f62": 5.7e7,
              "f184": 136},
        "2": {"f3": 58, "f12": "BK0596", "f13": 90, "f14": "融资融券", "f62": -1.77e10,
              "f184": -310}}}}


def _self_test() -> None:
    """离线自测：合成数据 + 纯解析函数，**不发网络请求**。"""
    import core.selftest_data as sd

    print("=== sentiment self-test ===")
    frame = sd.make_ohlcv(n=300, seed=11)
    market, sector, retail = _fake_market(), _fake_sector(), _fake_retail()
    snapshot = {"turnover": 0.28, "volume_ratio": 1.18, "price": 1263.0, "pct_chg": 0.57}

    # ---- 纯解析函数（模拟真实响应，离线可测） ---------------------------- #
    zt = _parse_zt_pool(_zt_payload())
    zb = _parse_zb_pool(_zb_payload())
    idx = _parse_ulist(_ulist_payload())
    boards = _parse_slist(_slist_payload())
    print("涨停池解析：涨停 %d 家、连板 %d 家、最高 %d 板、热门行业 %s"
          % (zt["limit_up_count"], zt["continuous_count"], zt["max_continuous"],
             _hot_boards(zt["items"])))
    print("炸板池解析：炸板 %d 家；炸板率 = %d/(%d+%d) = %.1f%%"
          % (zb["broken_count"], zb["broken_count"], zt["limit_up_count"],
             zb["broken_count"],
             zb["broken_count"] / (zt["limit_up_count"] + zb["broken_count"]) * 100))
    print("指数快照解析：%s" % [(x["name"], x["pct"], x["up"], x["down"]) for x in idx])
    print("板块解析（f3/f184 已从放大 100 倍还原）：%s"
          % [(b["name"], b["pct"], b["main_pct"]) for b in boards])
    assert abs(boards[0]["pct"] - 0.82) < 1e-9 and abs(boards[0]["main_pct"] - 2.84) < 1e-9

    # ---- 恐惧贪婪指数 ---------------------------------------------------- #
    fg = fear_greed_index("600519", frame=frame, market=market)
    print("恐惧贪婪指数：%.1f（%s）分项=%s"
          % (fg["value"], fg["level"],
             {k: (round(v, 3) if isinstance(v, float) else v)
              for k, v in fg["components"].items()}))
    assert 0 <= fg["value"] <= 100

    # ---- 两融归一化（DataFrame → 情绪模块结构） -------------------------- #
    margin_df = pd.DataFrame({
        "date": pd.date_range("2026-07-01", periods=60, freq="B"),
        "融资余额": np.linspace(1.55e10, 1.72e10, 60),
        "融资余额占流通市值比": 1.05,
    })
    ms = margin_sentiment(margin_df)
    print("两融归一化：current=%.2f 亿，history=%d 点，5 日变化 %+.2f%%，占流通市值 %.2f%%"
          % (ms["current"] / 1e8, len(ms["history"]), ms["change_5d_pct"], ms["ratio_pct"]))
    assert ms and len(ms["history"]) == 60
    market_with_margin = dict(market, margin=margin_df)

    # ---- 因子生成 -------------------------------------------------------- #
    fs = sentiment_factors("600519", market=market_with_margin, sector=sector, retail=retail,
                           frame=frame, snapshot=snapshot)
    ids = sorted({f.id for f in fs})
    print("因子数量：%d" % len(fs))
    print("因子 id 清单（%d）：%s" % (len(ids), ids))
    assert "margin_euphoria" in ids and "margin_trend" in ids, "两融因子应能取到数据"
    assert all(f.direction != 0 for f in fs if f.id in ("margin_euphoria", "margin_trend"))
    contrarian = {"retail_sentiment_contrarian", "guba_bull_ratio", "search_heat_spike",
                  "turnover_euphoria", "margin_euphoria", "limit_up_frenzy"}
    assert contrarian.issubset(set(ids) | set(config.CONTRARIAN_FACTOR_IDS))
    for f in fs:
        print("   %-30s %-4s 强度 %.2f 可信度 %.2f | %s"
              % (f.id, {1: "看涨", -1: "看跌", 0: "中性"}[f.direction], f.strength,
                 f.reliability, (f.detail or "")[:58]))
    assert all(f.category == Category.SENTIMENT for f in fs)

    # ---- 全空用例 -------------------------------------------------------- #
    empty = sentiment_factors("600519")
    print("空数据因子数量：%d，id：%s" % (len(empty), sorted({f.id for f in empty})))
    for f in empty:
        assert f.reliability <= 0.5
    fg0 = fear_greed_index()
    print("空数据恐惧贪婪：%.1f（%s）" % (fg0["value"], fg0["level"]))
    assert fg0["value"] == 50.0
    assert _parse_zt_pool(None)["limit_up_count"] == 0
    assert _parse_slist(None) == [] and _parse_ulist(None) == []

    print("sentiment self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()

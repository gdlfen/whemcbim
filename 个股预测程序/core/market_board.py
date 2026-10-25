# -*- coding: utf-8 -*-
"""
自上而下的市场分析：**先大盘 → 再板块/行业 → 后个股**。

为什么必须自上而下
------------------
A股个股涨跌中，很大一部分来自**系统性**（大盘）与**行业性**（板块）因素，
只有剩下那部分才是个股自身的 alpha。官方统计口径下，A股个股波动里约有一半
可以由市场因子解释，行业因子再解释一部分 —— 所以：

    个股收益 ≈ β_市场 × 大盘 + β_行业 × 行业 + 个股alpha

如果跳过前两层直接看个股，很容易把「大盘涨带起来的」当成「个股走强」，
或者把「板块整体回调」当成「个股基本面恶化」。

本模块的三层职责
----------------
1. **大盘层**（:mod:`core.market_structure`）：大周期牛熊 / 中级波段 / 短线结构
2. **板块层**（本模块）：领涨领跌板块、板块轮动阶段、热点集中度、板块动量
3. **个股层**：由 engine 原有的八大维度负责（含个股相对大盘/行业的超额）

性能设计（为什么用多线程）
--------------------------
板块数据要逐个拉历史K线（几十个板块），串行会很慢。
这里用 ``ThreadPoolExecutor`` **并发抓取**：板块列表一次拿到，K线并发拉。
并发数默认 8（可在 ``config.SPEED`` 调整），并在失败时优雅降级。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .contracts import (Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor,
                        safe_float)
from .utils import get_logger

log = get_logger()

#: 板块列表进程内缓存（东财该接口偶发返回空，且限流，缓存 180 秒足够）
_BOARD_CACHE: dict = {"ts": 0.0, "rows": [], "total": 0}
_BOARD_TTL = 180.0

#: 板块列表可用主机（按成功率排序，逐个尝试）
_BOARD_HOSTS = ("82.push2.eastmoney.com", "push2delay.eastmoney.com",
                "push2.eastmoney.com")

#: 大盘指数清单（自上而下的第一层）
MARKET_INDICES: dict[str, str] = {
    "sh": "上证指数",
    "sz": "深证成指",
    "cyb": "创业板指",
    "hs300": "沪深300",
    "zz500": "中证500",
    "kc50": "科创50",
    "bj": "北证50",          # 京股代表（可能取不到，取不到时如实标注）
}

#: 板块轮动阶段 → 该阶段通常领涨的方向（用户给的 A股轮动顺序）
ROTATION_MAP = {
    "bull_start": ("金融、地产、基建（稳增长）", "券商、银行、保险、建筑、地产"),
    "bull_mid": ("科技、新能源、高端制造（成长主线）", "半导体、AI、新能源车、光伏、军工、机械"),
    "bull_end": ("消费、医药补涨（收尾）", "白酒、食品饮料、医药、家电、旅游"),
    "bear_start": ("高股息、公用事业防御", "电力、水务、高速公路、煤炭、银行"),
    "bear_mid": ("高股息、公用事业防御", "电力、水务、高速公路、煤炭、银行"),
    "bear_end": ("超跌低估值 + 政策受益", "券商、地产链、建筑、低估值蓝筹"),
}


# --------------------------------------------------------------------------- #
# 一、大盘：多个指数共振判断（比只看一个更可靠）
# --------------------------------------------------------------------------- #
def market_overview(frames: dict, days: int = 400) -> dict:
    """汇总主要指数的涨跌与共振情况，判断「大盘到底强不强」。

    Parameters
    ----------
    frames : dict[str, DataFrame]
        ``{"sh": df, "sz": df, ...}``，由调用方并发或串行取好

    Returns
    -------
    dict：``rows / avg_ret_5d / agreement / detail``
    """
    out: dict = {"rows": [], "detail": ""}
    rows = []
    for key, nm in MARKET_INDICES.items():
        df = frames.get(key)
        if df is None or not len(df) or "close" not in df.columns:
            continue
        try:
            c = df["close"].astype(float).dropna()
            if len(c) < 25:
                continue
            r1 = float(c.iloc[-1] / c.iloc[-2] - 1) * 100
            r5 = float(c.iloc[-1] / c.iloc[-6] - 1) * 100
            n20 = min(21, len(c))
            r20 = float(c.iloc[-1] / c.iloc[-n20] - 1) * 100
            win = min(len(c), 250)
            hi, lo = float(c.tail(win).max()), float(c.tail(win).min())
            rows.append({
                "key": key, "name": nm,
                "close": round(float(c.iloc[-1]), 2),
                "ret_1d": round(r1, 2), "ret_5d": round(r5, 2),
                "ret_20d": round(r20, 2),
                "pos_pct": round((float(c.iloc[-1]) - lo) / max(hi - lo, 1e-9) * 100, 1),
                "above_ma20": bool(c.iloc[-1] >= c.tail(20).mean()),
                "above_ma60": bool(c.iloc[-1] >= c.tail(min(60, len(c))).mean()),
            })
        except Exception as exc:  # noqa: BLE001
            log.debug("指数 %s 概览失败: %s", key, exc)
    out["rows"] = rows
    if not rows:
        out["detail"] = "未取到任何大盘指数数据"
        return out
    for h in ("ret_1d", "ret_5d", "ret_20d"):
        vals = [r[h] for r in rows]
        ups = sum(1 for v in vals if v > 0)
        downs = sum(1 for v in vals if v < 0)
        out["up_%s" % h] = ups
        out["down_%s" % h] = downs
        out["avg_%s" % h] = round(float(np.mean(vals)), 2)
        out["agree_%s" % h] = round(max(ups, downs) / max(ups + downs, 1) * 100, 0)
    out["above_ma20_count"] = sum(1 for r in rows if r["above_ma20"])
    out["above_ma60_count"] = sum(1 for r in rows if r["above_ma60"])
    out["detail"] = (
        "大盘概览：%d 个指数，近5日 %d涨%d跌（一致度 %.0f%%），平均 %+.2f%%；"
        "站上 MA20 的 %d 个、站上 MA60 的 %d 个。"
        % (len(rows), out["up_ret_5d"], out["down_ret_5d"], out["agree_ret_5d"],
           out["avg_ret_5d"], out["above_ma20_count"], out["above_ma60_count"]))
    return out


# --------------------------------------------------------------------------- #
# 二、板块：并发抓取 + 轮动分析
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 同花顺行业板块（主源：列表 + 快照 + 历史同一口径）
# --------------------------------------------------------------------------- #
#: 板块主源：统一到同花顺（列表/快照/历史同一口径）
PREFER_THS = True

_THS_CODE_CACHE: dict = {}


def _ths_name_code() -> dict:
    """同花顺行业板块 名称 → 代码（881xxx）。"""
    if _THS_CODE_CACHE:
        return _THS_CODE_CACHE
    try:
        import akshare as ak

        df = ak.stock_board_industry_name_ths()
        if df is not None and len(df):
            for _, r in df.iterrows():
                nm = str(r.get("name") or "").strip()
                cd = str(r.get("code") or "").strip()
                if nm and cd:
                    _THS_CODE_CACHE[nm] = cd
    except Exception as exc:  # noqa: BLE001
        log.debug("同花顺板块名称表失败: %s", exc)
    return _THS_CODE_CACHE


def _fetch_board_list_ths() -> list:
    """同花顺行业板块当日快照（90 个，字段最全）。"""
    try:
        import akshare as ak

        df = ak.stock_board_industry_summary_ths()
        if df is None or not len(df):
            return []
        codes = _ths_name_code()
        out = []
        for _, r in df.iterrows():
            nm = str(r.get("板块") or "").strip()
            if not nm:
                continue
            try:
                pct = float(r.get("涨跌幅"))
            except Exception:  # noqa: BLE001
                continue
            out.append({
                "code": codes.get(nm) or nm,
                "name": nm, "pct": pct,
                "amount": _safe_num(r.get("总成交额")),
                "turnover": _safe_num(r.get("总成交量")),
                "net_inflow": _safe_num(r.get("净流入")),
                "up_count": _safe_int(r.get("上涨家数")),
                "down_count": _safe_int(r.get("下跌家数")),
                "leader": r.get("领涨股"),
                "price": _safe_num(r.get("均价")),
                "source": "ths_industry",
            })
        if out:
            out.sort(key=lambda x: -abs(x["pct"]))
        return out
    except Exception as exc:  # noqa: BLE001
        log.debug("同花顺板块快照失败: %s", exc)
        return []


def _safe_num(v):
    try:
        f = float(v)
        return f if f == f else None
    except Exception:  # noqa: BLE001
        return None


def _safe_int(v):
    try:
        return int(float(v))
    except Exception:  # noqa: BLE001
        return None


def fetch_ths_board_history(name: str, days: int = 400):
    """同花顺单板块历史K线（名称即参数）。失败返回 None。"""
    try:
        import akshare as ak

        # **必须显式传日期**：该函数默认参数被硬编码为
        # start_date="20200101", end_date="20240108"，
        # 不传的话历史永远只到 2024-01-08（曾被误判为"数据源停更"）。
        import datetime as _dt

        _today = _dt.date.today()
        _start = (_today - _dt.timedelta(days=max(int(days) * 2, 400)))
        df = ak.stock_board_industry_index_ths(
            symbol=name, start_date=_start.strftime("%Y%m%d"),
            end_date=_today.strftime("%Y%m%d"))
        if df is None or not len(df):
            return None
        ren = {"日期": "date", "开盘价": "open", "最高价": "high",
               "最低价": "low", "收盘价": "close", "成交量": "volume",
               "成交额": "amount"}
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
        if "date" not in df.columns or "close" not in df.columns:
            return None
        import pandas as _pd

        df["date"] = _pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df.tail(max(int(days), 120))
    except Exception as exc:  # noqa: BLE001
        log.debug("同花顺板块历史 %s 失败: %s", name, exc)
        return None


def _fetch_board_list_sina() -> list:
    """新浪行业板块列表（东财不可用时的备份源）。

    返回格式与东财一致：``code/name/pct/price/turnover/amount/leader``，
    但**没有** turnover 与领涨股（字段里没有），设为 None 由上层如实标注。
    """
    from .datasource import http_get
    import json as _json
    import re as _re

    url = "https://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
    try:
        txt = http_get(url, timeout=12, retries=1,
                       headers={"Referer": "https://finance.sina.com.cn"})
    except Exception as exc:  # noqa: BLE001
        log.debug("新浪板块列表失败: %s", exc)
        return []
    if not txt:
        return []
    m = _re.search(r"=\s*(\{.*\})\s*;?\s*$", txt.strip(), _re.S)
    if not m:
        return []
    try:
        raw = _json.loads(m.group(1))
    except Exception as exc:  # noqa: BLE001
        log.debug("新浪板块解析失败: %s", exc)
        return []
    rows = []
    for key, val in raw.items():
        try:
            p = str(val).split(",")
            if len(p) < 8:
                continue
            rows.append({
                "code": str(p[0] or key),
                "name": str(p[1] or ""),
                "pct": safe_float(p[5]),          # 涨跌幅%
                "price": safe_float(p[3]),        # 平均价
                "turnover": None,
                "amount": safe_float(p[7]),       # 总成交额
                "up_count": None, "down_count": None,
                "leader": str(p[8]).strip() if len(p) > 8 else "",
                "source": "sina_industry",
            })
        except Exception:  # noqa: BLE001
            continue
    rows.sort(key=lambda x: -(x.get("pct") if x.get("pct") == x.get("pct") else -999))
    return rows


def fetch_board_list(top: int = 40, force: bool = False,
                     fetch_all: bool = True) -> list:
    # 同花顺优先：列表、快照、历史全部同一套行业分类，
    # 避免新浪/东财/同花顺三源混用导致"同一板块不同成分"。
    if PREFER_THS:
        try:
            ths = _fetch_board_list_ths()
            if ths and len(ths) >= 20:
                log.info("板块列表：同花顺行业板块（%d 个）", len(ths))
                return ths
        except Exception as exc:  # noqa: BLE001
            log.debug("同花顺板块列表失败，降级: %s", exc)

    """取当日板块涨跌数据（东财行业板块）。

    **重要坑（实测）**：该接口每页最多返回 100 条，而全市场有 496 个行业板块。
    早期版本用 ``pz=40&po=1``（按涨幅降序）只拿到**涨幅榜前 40 个**，
    于是「板块宽度」永远算出接近 100% 上涨 —— 这是**幸存者偏差**，
    会严重误导判断（实测当天真实行情是普跌：升序那 100 个板块平均 -2.09%，
    而涨幅榜前 100 个全是上涨的）。

    所以这里改为：分页并发抓取**全部**板块，再自行排序取前/后 N。
    """

    def _one_page(pn: int, order: str = "desc"):
        from .datasource import http_get
        import json as _json

        # po=1 涨幅降序；po=0 涨幅升序（跌幅榜），用于双向采样
        _po = 0 if order == "asc" else 1
        path = ("/api/qt/clist/get?pn=%d&pz=100&po=%d&np=1"
                "&fltt=2&invt=2&fid=f3&fs=m:90+t:2+f:!50"
                "&fields=f12,f14,f2,f3,f4,f5,f6,f8,f20,f104,f105,f128,f136"
                % (pn, _po))
        txt = ""
        for host in _BOARD_HOSTS:
            try:
                txt = http_get("https://%s%s" % (host, path), timeout=12, retries=1)
                if txt and '"diff"' in txt:
                    break
                txt = ""
            except Exception:  # noqa: BLE001
                txt = ""
                continue
        try:
            if not txt:
                return [], 0
            d = (_json.loads(txt).get("data") or {})
            rows = []
            for r in (d.get("diff") or []):
                try:
                    rows.append({
                        "code": str(r.get("f12") or ""),
                        "name": str(r.get("f14") or ""),
                        "pct": safe_float(r.get("f3")),
                        "price": safe_float(r.get("f2")),
                        "turnover": safe_float(r.get("f8")),
                        "amount": safe_float(r.get("f6")),
                        "up_count": r.get("f104"), "down_count": r.get("f105"),
                        "leader": str(r.get("f128") or ""),
                    })
                except Exception:  # noqa: BLE001
                    continue
            return rows, int(d.get("total") or 0)
        except Exception as exc:  # noqa: BLE001
            log.debug("板块列表第 %d 页失败: %s", pn, exc)
            return [], 0

    import time as _time

    # 命中缓存
    if not force and _BOARD_CACHE["rows"] and \
            (_time.time() - _BOARD_CACHE["ts"]) < _BOARD_TTL:
        return list(_BOARD_CACHE["rows"])

    # 该接口偶发返回空（限流/抖动），重试几次
    first, total = _one_page(1)
    for _try in range(3):
        if first:
            break
        _time.sleep(0.6 * (_try + 1))
        first, total = _one_page(1)
    if not first:
        # 东财不可用 → 换新浪行业板块（实测稳定，覆盖全部行业）
        sina = _fetch_board_list_sina()
        if sina:
            log.info("板块列表：东财不可用，已切换到新浪行业板块（%d 个）", len(sina))
            _BOARD_CACHE["rows"] = list(sina)
            _BOARD_CACHE["ts"] = _time.time()
            _BOARD_CACHE["total"] = len(sina)
            return sina
        # 都拿不到就退回上次缓存（哪怕是旧的），总比"0 个板块"好
        if _BOARD_CACHE["rows"]:
            log.warning("板块列表取数失败，使用 %.0f 秒前的缓存",
                        _time.time() - _BOARD_CACHE["ts"])
            return list(_BOARD_CACHE["rows"])
        return []
    rows: list = []
    if not fetch_all or total <= 100:
        rows = list(first)
    else:
        pages = min(int(np.ceil(total / 100.0)), 6)
        rows = list(first)
        got_pages = 1
        with ThreadPoolExecutor(max_workers=min(pages, 5)) as pool:
            futs = {pool.submit(_one_page, pn): pn for pn in range(2, pages + 1)}
            for fu in as_completed(futs):
                try:
                    more, _ = fu.result()
                    if more:
                        rows.extend(more)
                        got_pages += 1
                except Exception:  # noqa: BLE001
                    continue
        # 分页不完整（网络抖动）→ 补一页跌幅榜，做**双向采样**，
        # 避免"只有涨幅榜"导致的宽度偏高（幸存者偏差）。
        if got_pages < pages:
            try:
                tail, _ = _one_page(1, order="asc")
                if tail:
                    rows.extend(tail)
            except Exception:  # noqa: BLE001
                pass
    seen, uniq = set(), []
    for r in rows:
        c = r.get("code")
        if c and c not in seen:
            seen.add(c)
            uniq.append(r)
    uniq.sort(key=lambda x: -(x.get("pct") if x.get("pct") == x.get("pct") else -999))
    if uniq:
        _BOARD_CACHE["rows"] = list(uniq)
        _BOARD_CACHE["total"] = int(total or len(uniq))
        _BOARD_CACHE["ts"] = __import__("time").time()
    return uniq


def _one_board_hist(board: dict, days: int = 250):
    """并发任务：抓单个板块的历史K线（板块必须用东财专用接口）。

    板块代码形如 ``BK1295``，``datasource.fetch_daily`` 不支持（那只认个股/指数
    前缀）；正确接口是 ``push2his.../stock/kline/get?secid=90.BKxxxx``。
    """
    code = str(board.get("code") or "").replace("BK", "")
    if not code:
        return board.get("code"), None
    from .datasource import http_get
    import json as _json

    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?"
           "secid=90.BK%s&fields1=f1,f2,f3,f4,f5"
           "&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=1"
           "&end=20500101&lmt=%d" % (code, max(days, 60)))
    try:
        txt = http_get(url, timeout=15, retries=1)
        if not txt:
            return board.get("code"), None
        kl = ((_json.loads(txt).get("data") or {}).get("klines")) or []
        if len(kl) < 20:
            return board.get("code"), None
        rows = []
        for line in kl:
            p = str(line).split(",")
            if len(p) < 7:
                continue
            rows.append({"date": p[0], "open": safe_float(p[1]),
                         "close": safe_float(p[2]), "high": safe_float(p[3]),
                         "low": safe_float(p[4]), "volume": safe_float(p[5])})
        df = pd.DataFrame(rows)
        if not len(df):
            return board.get("code"), None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return board.get("code"), df
    except Exception as exc:  # noqa: BLE001
        log.debug("板块 %s 历史失败: %s", board.get("name"), exc)
        return board.get("code"), None


def fetch_board_histories(boards: list, days: int = 250,
                          max_workers: int | None = None) -> dict:
    # 同花顺来源：直接用板块名称取历史（与列表同口径）
    src_hint = ""
    try:
        src_hint = str((boards[0] or {}).get("source") or "") if boards else ""
    except Exception:  # noqa: BLE001
        src_hint = ""
    if src_hint == "ths_industry":
        got: dict = {}
        for b in boards or []:
            nm = str((b or {}).get("name") or "")
            if not nm:
                continue
            df = fetch_ths_board_history(nm, days=days)
            if df is not None and len(df):
                got[str((b or {}).get("code") or nm)] = df
        # 新鲜度登记：同花顺板块历史实测最新只到 2024-01-08，
        # 登记后由通用机制统一暴露，避免"拿旧数据算概率"。
        try:
            from . import freshness as _FR

            if got:
                _FR.validate("board_history", list(got.values())[0]["date"].max())
        except Exception:  # noqa: BLE001
            pass
        log.info("板块历史：同花顺 %d/%d 个成功", len(got), len(boards or []))
        return got

    """**并发**抓取多个板块的历史K线（性能关键路径）。

    串行抓 30 个板块约需 30~60 秒；并发（默认 8 线程）通常 5~10 秒。
    """
    from . import config as _cfg

    n = max(1, min(len(boards), max_workers or int(
        (_cfg.SPEED or {}).get("board_workers", 8))))
    # 只抓前 8 与后 8（涨跌两端）的历史 —— 全量抓 40 个既慢又容易被限流，
    # 而轮动分析真正需要的是"最强/最弱"两端的动量。
    if boards and len(boards) > 16:
        boards = list(boards[:8]) + list(boards[-8:])
    out: dict = {}
    if not boards:
        return out
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(_one_board_hist, b, days): b for b in boards}
        for fu in as_completed(futs):
            b = futs[fu]
            try:
                code, df = fu.result()
            except Exception as exc:  # noqa: BLE001
                log.debug("板块K线并发任务失败 %s: %s", b.get("name"), exc)
                continue
            if df is not None:
                out[code] = df
    return out


def board_rotation(boards: list, hists: dict, phase: str = "") -> dict:
    """板块轮动分析：当日强弱、近20日动量、热点集中度、轮动阶段匹配。

    Returns
    -------
    dict：``top / bottom / momentum / concentration / rotation_hint / detail``
    """
    out: dict = {"detail": "", "items": [], "top": [], "bottom": []}
    if not boards:
        out["detail"] = "未取到板块数据"
        return out
    items = []
    for b in boards:
        code = b.get("code") or ""
        row = dict(b)
        df = hists.get(code)
        if df is not None and len(df) >= 25:
            try:
                c = df["close"].astype(float).dropna()
                if len(c) >= 21:
                    row["ret_20d"] = round(float(c.iloc[-1] / c.iloc[-21] - 1) * 100, 2)
                if len(c) >= 6:
                    row["ret_5d_calc"] = round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 2)
                win = min(len(c), 250)
                hi, lo = float(c.tail(win).max()), float(c.tail(win).min())
                row["pos_pct"] = round(
                    (float(c.iloc[-1]) - lo) / max(hi - lo, 1e-9) * 100, 1)
                row["above_ma20"] = bool(c.iloc[-1] >= c.tail(20).mean())
            except Exception:  # noqa: BLE001
                pass
        items.append(row)
    out["items"] = items
    # 当日强弱
    sorted_by_pct = sorted(items, key=lambda x: -(x.get("pct") or -999))
    out["top"] = sorted_by_pct[:8]
    out["bottom"] = sorted_by_pct[-5:]
    # 20 日动量（有历史的才参与）
    mom = [x for x in items if x.get("ret_20d") is not None]
    if mom:
        mom.sort(key=lambda x: -x["ret_20d"])
        out["momentum_top"] = [{"name": x["name"], "ret_20d": x["ret_20d"]}
                               for x in mom[:8]]
        out["momentum_bottom"] = [{"name": x["name"], "ret_20d": x["ret_20d"]}
                                  for x in mom[-5:]]
    # 热点集中度：涨幅前列板块的平均涨幅与家数
    pcts = [x.get("pct") for x in items if x.get("pct") is not None]
    ups = sum(1 for p in pcts if p > 0)
    downs = sum(1 for p in pcts if p < 0)
    # ---- 真实宽度：优先用每板块的上涨/下跌家数汇总 ----
    # 旧做法只数"涨跌幅>0 的板块个数"，本质是数板块而非数个股；
    # 若数据源只覆盖涨幅榜，会系统性高估（AI 核对指出 93.3% 极可能失真）。
    # 同花顺快照带 上涨家数/下跌家数，汇总后即全市场真实涨跌家数比。
    _up = sum(int(x.get("up_count") or 0) for x in items
              if x.get("up_count") is not None)
    _dn = sum(int(x.get("down_count") or 0) for x in items
              if x.get("down_count") is not None)
    if (_up + _dn) >= max(200, len(items) * 3):
        out["board_breadth_pct"] = round(_up / max(_up + _dn, 1) * 100, 1)
        out["breadth_basis"] = ("个股涨跌家数汇总（%d 涨 / %d 跌，覆盖 %d 个板块）"
                                 % (_up, _dn, len(items)))
        out["breadth_is_real"] = True
    else:
        out["board_breadth_pct"] = round(ups / max(len(pcts), 1) * 100, 1)
        out["breadth_basis"] = ("按板块涨跌幅计数（%d/%d 个板块上涨）——"
                                 "启发式，非真实家数比" % (ups, len(pcts)))
        out["breadth_is_real"] = False
    out["board_count"] = len(items)
    # ---- 数据完整性判断（关键：防止把"涨幅榜"当成全市场）---- #
    # 全市场板块总数：优先用接口 total；缺失时按经验值 496 估计。
    # 若退回 len(items)，"只有 100 个且全涨"会被误判为完整样本。
    total = int(_BOARD_CACHE.get("total") or 0) or max(len(items), 496)
    cover = len(items) / max(total, 1)
    out["board_total"] = total
    out["board_coverage_pct"] = round(cover * 100, 1)
    # 单向（只有涨幅榜）的判据：几乎全是上涨且样本远少于全市场
    one_sided = (ups / max(len(pcts), 1)) >= 0.9 and cover < 0.6
    two_sided = (not one_sided) and cover < 0.6 and downs > 0
    out["breadth_quality"] = (
        "不完整（仅涨幅榜，宽度偏高的风险大）" if one_sided else
        ("估计值（双向采样 %.0f%% 覆盖）" % (cover * 100) if two_sided else
         "完整"))
    out["breadth_reliable"] = bool(cover >= 0.6 and not one_sided)
    if pcts:
        out["board_avg_pct"] = round(float(np.mean(pcts)), 2)
        out["board_median_pct"] = round(float(np.median(pcts)), 2)
    # 轮动阶段提示
    if phase and phase in ROTATION_MAP:
        themes, examples = ROTATION_MAP[phase]
        hit = [x["name"] for x in out["top"]
               if any(k in str(x.get("name")) for k in examples.split("、"))]
        # AI 核对指出：matched 为空时仍断言"高股息防御风格占优"属过度解读
        # （提示举例电力/水务/煤炭/银行，但清单中一个都没命中）。
        # 未匹配时必须明确说明，避免把方向性提示当成已验证结论。
        _unmatched = not hit
        out["rotation_hint"] = {
            "phase_themes": themes, "phase_examples": examples,
            "matched": hit, "unmatched": _unmatched,
            "detail": ("当前阶段偏好的方向：%s（例：%s）" % (themes, examples)
                        + ("\n\n⚠️ **本轮未匹配到具体板块**（清单中无板块命中上述方向）——"
                           "这只是基于阶段的方向性提示，**不构成风格断言**；"
                           "需等板块覆盖完整、且确有板块命中后再确认。"
                           if _unmatched else
                           "\n\n已命中板块：%s" % "、".join(hit[:6]))),
        }
    _q = out.get("breadth_quality") or "完整"
    out["detail"] = (
        "板块情况：%d/%d 个板块中 %d 个上涨（板块宽度 %.0f%%），平均 %+.2f%%。"
        "数据完整性：%s。当日领涨：%s。"
        % (out["board_count"], total, ups, out.get("board_breadth_pct") or 0,
           out.get("board_avg_pct") or 0, _q,
           "、".join("%s(%.1f%%)" % (x["name"], x.get("pct") or 0)
                    for x in out["top"][:5])))
    return out


# --------------------------------------------------------------------------- #
# 三、因子
# --------------------------------------------------------------------------- #
def market_board_factors(overview: dict, rotation: dict) -> list[Factor]:
    """把大盘 + 板块情况转成因子。"""
    out: list[Factor] = []
    # 大盘共振
    ag = safe_float(overview.get("agree_ret_5d"))
    up = overview.get("up_ret_5d") or 0
    dn = overview.get("down_ret_5d") or 0
    if ag == ag and (up + dn) >= 3:
        direction = DIR_BULL if up > dn else (DIR_BEAR if dn > up else DIR_NEUTRAL)
        out.append(Factor(
            id="market_breadth", name="大盘%s（%d涨%d跌）"
                 % ("共振向上" if direction > 0 else
                    ("共振向下" if direction < 0 else "分歧"), up, dn),
            category=Category.MACRO, direction=direction,
            strength=min(0.3 + (ag - 50) / 100.0 * 1.6, 0.85),
            reliability=0.74 if ag >= 70 else 0.55,
            value="近5日一致度 %.0f%%，平均 %+.2f%%" % (ag, overview.get("avg_ret_5d") or 0),
            detail="%s 一致度越高说明是**系统性行情**（个股跟随大盘概率高）；"
                   "分歧则说明是结构性行情（选错方向代价大）。"
                   % (overview.get("detail") or ""),
            theory="自上而下/市场宽度", horizon=("d1", "d5", "d10", "d20")))
    # 站上均线的指数个数
    n20 = overview.get("above_ma20_count") or 0
    tot = len(overview.get("rows") or []) or 1
    ratio = n20 / tot
    out.append(Factor(
        id="market_ma_health", name="大盘均线健康度 %.0f%%" % (ratio * 100),
        category=Category.MACRO,
        direction=DIR_BULL if ratio >= 0.6 else (DIR_BEAR if ratio <= 0.3 else DIR_NEUTRAL),
        strength=min(abs(ratio - 0.5) * 1.6 + 0.2, 0.8), reliability=0.7,
        value="%d/%d 个指数站上MA20" % (n20, tot),
        detail="多数指数站上 MA20 说明中期趋势向上；"
               "只有少数站上说明是局部反弹，不宜当趋势。",
        theory="均线系统/市场宽度", horizon=("d5", "d10", "d20")))
    # 板块宽度
    bw = safe_float(rotation.get("board_breadth_pct"))
    if bw == bw:
        out.append(Factor(
            id="board_breadth", name="板块宽度 %.0f%%" % bw,
            category=Category.SENTIMENT,
            direction=DIR_BULL if bw >= 60 else (DIR_BEAR if bw <= 30 else DIR_NEUTRAL),
            strength=min(abs(bw - 50) / 100.0 * 1.8 + 0.15, 0.8),
            # 样本不完整（只拿到部分板块）时可信度必须下调，
            # 否则"涨幅榜 100% 上涨"会被误当成"普涨"
            reliability=(0.66 if rotation.get("breadth_reliable") else 0.4),
            value="%d 个板块中 %d 个上涨（%s）" % (
                rotation.get("board_count") or 0,
                int(round((bw / 100.0) * (rotation.get("board_count") or 0))),
                rotation.get("breadth_quality") or "完整"),
            detail="板块宽度衡量「赚钱效应」是否扩散：宽度高说明普涨（资金充沛），"
                   "宽度低说明只有少数板块在涨（存量博弈，追高风险大）。",
            theory="市场宽度/赚钱效应", horizon=("d5", "d10", "d20")))
    # 领涨板块（龙头效应）
    top = rotation.get("top") or []
    if top:
        t0 = top[0]
        out.append(Factor(
            id="board_leader", name="当日领涨板块：%s" % t0.get("name"),
            category=Category.SENTIMENT,
            direction=DIR_BULL if (t0.get("pct") or 0) > 2 else DIR_NEUTRAL,
            strength=min(abs(t0.get("pct") or 0) / 8.0 + 0.2, 0.7),
            reliability=0.55,
            value="%+.2f%%" % (t0.get("pct") or 0),
            detail="领涨板块：%s。热点集中度：前5板块平均 %+.2f%%。"
                   "追热点要注意「一日游」风险——连续多日领涨才算真主线。"
                   % ("、".join("%s(%+.1f%%)" % (x.get("name"), x.get("pct") or 0)
                               for x in top[:5]),
                      float(np.mean([x.get("pct") or 0 for x in top[:5]]))),
            theory="板块轮动/热点效应", horizon=("d1", "d5")))
    return out


# --------------------------------------------------------------------------- #
# 四、一站式（带并发）
# --------------------------------------------------------------------------- #
def analyze_topdown(index_frames: dict | None = None, *,
                    days: int = 400, phase: str = "",
                    with_boards: bool = True,
                    max_workers: int | None = None,
                    verbose: bool = False) -> dict:
    """自上而下分析：大盘概览 → 板块轮动 → 因子。

    尽量并发：指数与板块列表同时取，板块历史再并发。
    """
    from . import config as _cfg
    from . import global_market as GM

    out: dict = {"overview": {}, "rotation": {}, "factors": [],
                 "detail": "", "boards_raw": []}
    frames = dict(index_frames or {})
    boards: list = []

    # ---- 阶段一：并发取「指数历史」与「板块列表" ---- #
    jobs = []
    with ThreadPoolExecutor(max_workers=max(2, int(
            (_cfg.SPEED or {}).get("market_workers", 6)))) as pool:
        idx_futs = {}
        for key in MARKET_INDICES:
            if key in frames and frames[key] is not None and len(frames[key]):
                continue
            idx_futs[pool.submit(GM.fetch_index_history, key, days)] = key
        board_fut = pool.submit(fetch_board_list, 40) if with_boards else None
        for fu in as_completed(list(idx_futs) + ([board_fut] if board_fut else [])):
            if board_fut is not None and fu is board_fut:
                try:
                    boards = fu.result() or []
                except Exception as exc:  # noqa: BLE001
                    log.debug("板块列表失败: %s", exc)
                continue
            key = idx_futs.get(fu)
            try:
                df = fu.result()
                if df is not None and len(df):
                    frames[key] = df
            except Exception as exc:  # noqa: BLE001
                log.debug("指数 %s 取数失败: %s", key, exc)

    out["overview"] = market_overview(frames, days)

    # ---- 阶段二：并发抓板块历史（性能关键）---- #
    if boards and with_boards:
        # 宽度/强弱必须用**全量**板块计算；只有历史K线为控制请求量才截取前 N 个。
        # 早期版本把截断后的 sub 传给 board_rotation，导致覆盖度按 24/496 算、
        # 被误判为"数据不完整"（明明拿的是全量）。
        sub = boards[:int((_cfg.SPEED or {}).get("board_max", 24))]
        hists = fetch_board_histories(sub, days=min(days, 250),
                                      max_workers=max_workers)
        out["rotation"] = board_rotation(boards, hists, phase=phase)
        out["boards_raw"] = sub
        out["boards_hist_count"] = len(hists)
    else:
        out["rotation"] = board_rotation(boards, {}, phase=phase)

    out["factors"] = market_board_factors(out["overview"], out["rotation"])
    bits = [out["overview"].get("detail") or "", out["rotation"].get("detail") or ""]
    out["detail"] = "；".join(b for b in bits if b)
    out["index_frames"] = frames

    # ---- 口径一致性修正 ----
    # 宽度改用「个股涨跌家数汇总」后，已覆盖全部 90 个板块（含下跌方向），
    # 不再存在"只统计涨幅榜"的偏差；若仍沿用旧的完整性折扣，
    # 会出现"83.3% 却标注仅涨幅榜不完整"的自相矛盾（AI 核对图4 第4条）。
    if out.get("breadth_is_real"):
        out["breadth_quality"] = ("完整（个股涨跌家数汇总，无需完整性折扣）")
        out["breadth_reliable"] = True
    return out


if __name__ == "__main__":  # pragma: no cover
    import time
    import warnings

    warnings.filterwarnings("ignore")

    print("=== 1. 大盘概览（并发取 7 个指数）===")
    t0 = time.time()
    got = analyze_topdown(with_boards=True)
    dt = time.time() - t0
    ov = got["overview"]
    print("  耗时 %.1f 秒（含板块并发抓取）" % dt)
    print("  ", ov.get("detail"))
    for r in ov.get("rows") or []:
        print("     %-8s %8.2f  %+6.2f%%  %+6.2f%%  %+6.2f%%  位置%5.1f%%  %s"
              % (r["name"], r["close"], r["ret_1d"], r["ret_5d"], r["ret_20d"],
                 r["pos_pct"], "站上MA20" if r["above_ma20"] else "破MA20"))

    rot = got["rotation"]
    print("\n=== 2. 板块轮动 ===")
    print("  ", rot.get("detail"))
    print("  领涨前5:")
    for x in (rot.get("top") or [])[:5]:
        print("     %-14s %+6.2f%%  换手%.2f%%  20日 %s" % (
            x.get("name"), x.get("pct") or 0, x.get("turnover") or 0,
            ("%+.2f%%" % x["ret_20d"]) if x.get("ret_20d") is not None else "—"))
    if rot.get("momentum_top"):
        print("  20日动量前5:", "、".join(
            "%s(%+.1f%%)" % (x["name"], x["ret_20d"]) for x in rot["momentum_top"][:5]))

    print("\n=== 3. 因子 ===")
    for f in got["factors"]:
        print("   %-30s %s 强度%.2f 可信度%.2f" % (
            f.name[:28], {1: "多", -1: "空", 0: "中"}[f.direction],
            f.strength, f.reliability))
    assert len(got["factors"]) >= 3, "因子偏少"
    for f in got["factors"]:
        assert f.detail and f.theory and "None" not in str(f.value)

    print("\n=== 4. 边界情况 ===")
    assert analyze_topdown(with_boards=False)["overview"] is not None
    assert board_rotation([], {})["detail"]
    assert market_overview({})["detail"]
    print("  空输入安全")
    print("\nmarket_board self-test OK")

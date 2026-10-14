# -*- coding: utf-8 -*-
"""
大盘与外围股市影响 —— A股大盘 / 港股 / 外围股市对个股的传导。

覆盖范围
--------
* **A 股大盘**：上证指数、深证成指、创业板指、沪深300、中证500、科创50
  （同市场联动最强，是"系统性水位"）
* **港股**：恒生指数、恒生科技、国企指数
  （与 A 股共享中国基本面，且**同日交易时段重叠**，联动非常直接）
* **美股**：道琼斯、纳斯达克、标普500（**隔夜效应**：美股收盘早于 A 股开盘，
  其涨跌在 A 股开盘前就已确定，是最干净的"领先信号"）
* **风险偏好**：VIX 恐慌指数（全球风险偏好，高位压制风险资产）
* **A50 期货**：富时中国 A50（海外资金对 A 股的定价，常领先现货）

传导逻辑
--------
1. **同向联动**：外围大涨 → A 股开盘情绪偏暖；外围大跌 → 承压。
2. **领先滞后**：这里用**真实的时序关系**检验——计算"外围指数前一日涨跌"
   与"个股次日涨跌"的相关性，而不是想当然地假设联动方向。
3. **相对强度**：个股相对大盘的强弱（跑赢/跑输），是资金关注度的直接体现。
4. **风险偏好**：VIX 高位时，所有风险资产的方向判断都应降级。

数据源（均已实测）
------------------
* 历史K线：A股用 akshare（8000+ 根），港股用腾讯（300 根）
* 实时行情：腾讯 ``qt.gtimg.cn``（A股/港股/美股/VIX 均可）
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config
from .contracts import Category, DIR_BEAR, DIR_BULL, DIR_NEUTRAL, Factor, safe_float
from .utils import cache_key, cached, get_logger, http_get, retry_call

log = get_logger()

#: 指数清单：key -> (中文名, 腾讯行情代码, 腾讯K线代码, akshare代码, 分组)
INDICES: dict[str, tuple] = {
    # ---- A 股大盘 ----
    "sh": ("上证指数", "sh000001", "sh000001", "sh000001", "a_share"),
    "sz": ("深证成指", "sz399001", "sz399001", "sz399001", "a_share"),
    "cyb": ("创业板指", "sz399006", "sz399006", "sz399006", "a_share"),
    "hs300": ("沪深300", "sh000300", "sh000300", "sh000300", "a_share"),
    "zz500": ("中证500", "sh000905", "sh000905", "sh000905", "a_share"),
    "kc50": ("科创50", "sh000688", "sh000688", "sh000688", "a_share"),
    # ---- 港股 ----
    "hsi": ("恒生指数", "hkHSI", "hkHSI", None, "hk"),
    "hstech": ("恒生科技", "hkHSTECH", "hkHSTECH", None, "hk"),
    "hscei": ("国企指数", "hkHSCEI", "hkHSCEI", None, "hk"),
    # ---- 美股 ----
    "dji": ("道琼斯", "usDJI", "usDJI", None, "us"),
    "ixic": ("纳斯达克", "usIXIC", "usIXIC", None, "us"),
    "spx": ("标普500", "usINX", "usINX", None, "us"),
    # ---- 亚太 / 欧洲（隔夜，行情代码以 sina: 前缀表示走新浪 int_ 接口）----
    # 注：日经、英国富时走新浪 int_ 国际指数接口（腾讯无对应行情）。
    #     韩国 KOSPI 在实测可用的免费源（新浪 int_/gb_、腾讯 us/hk/jp 前缀、
    #     akshare index_global_hist_em）中**均取不到**，因此未纳入；
    #     报告与界面会如实说明"韩国股市数据源不可用"，而不是编造数据。
    "n225": ("日经225", "sina:int_nikkei", None, None, "asia"),
    "ftse": ("英国富时100", "sina:int_ftse", None, None, "eu"),
    # ---- 风险偏好 ----
    "vix": ("VIX恐慌指数", "usVIX", None, None, "risk"),
    # ---- A50 期货 ----
    "a50": ("富时中国A50", "hf_CHA50CFD", None, None, "a50"),
}

GROUP_LABEL = {"a_share": "A股大盘", "hk": "港股", "us": "美股（隔夜）",
               "asia": "亚太股市（隔夜）", "eu": "欧洲股市（隔夜）",
               "risk": "全球风险偏好", "a50": "A50期货"}


# --------------------------------------------------------------------------- #
# 取数
# --------------------------------------------------------------------------- #
def _quote_sina_int(code: str) -> dict:
    """新浪国际指数行情（``int_`` 前缀，如 ``int_nikkei``）。

    格式与腾讯完全不同：``名称,现价,涨跌额,涨跌幅%``（GBK）。
    实测可用：int_nikkei（日经）、int_ftse（英国富时）、int_dji/nasdaq/sp500、
    int_bovespa；**KOSPI 等代码返回空**（该市场暂无可用免费源）。
    """
    try:
        raw = http_get("https://hq.sinajs.cn/list=%s" % code, timeout=8, retries=1,
                       encoding="gbk",
                       headers={"Referer": "https://finance.sina.com.cn"})
        if not raw or "=" not in raw:
            return {}
        body = raw.strip().split("=", 1)[-1].strip().strip('";')
        if not body:
            return {}
        p = body.split(",")
        if len(p) < 4:
            return {}
        price = safe_float(p[1])
        chg = safe_float(p[2])
        pct = safe_float(p[3])
        if price != price or price <= 0:
            return {}
        prev = price - chg if chg == chg else float("nan")
        return {"name": str(p[0]).strip(), "price": price, "prev_close": prev,
                "pct_chg": pct, "source": "sina.int"}
    except Exception as exc:  # noqa: BLE001
        log.debug("新浪国际指数 %s 失败: %s", code, exc)
        return {}


def _quote_any(code: str) -> dict:
    """按行情代码前缀分派到对应数据源。"""
    if not code:
        return {}
    if code.startswith("sina:"):
        return _quote_sina_int(code.split(":", 1)[1])
    return _quote_tencent(code)


def _quote_tencent(code: str) -> dict:
    """腾讯实时行情（支持 A股/港股/美股/VIX/A50）。

    字段布局因市场而异（实测确认，不能一套索引走天下）：

    * **A股 / 港股**：``[3]现价 [4]昨收 [5]涨跌幅% [33]最高 [34]最低``
    * **美股指数**：``[3]现价 [5]昨收 [4]涨跌额`` —— 顺手用 [5] 反算涨跌幅
    * **VIX**：``[3]现价 [4]昨收 [5]涨跌幅``（多数时候涨跌幅为 0，需自行计算）
    * **A50（hf_ 前缀）**：完全不同的字段，单独解析
    """
    try:
        raw = http_get("https://qt.gtimg.cn/q=%s" % code, timeout=8, retries=1,
                       encoding="gbk")
        if not raw or "none_match" in raw:
            return {}
        body = raw.strip().split("=", 1)[-1].strip().strip('";')
        p = body.split("~")
        if len(p) < 6:
            return {}

        if code.startswith("hf_"):        # A50 期货这类 hf_ 行情
            return _quote_hf(code, p)

        price = safe_float(p[3])
        if price != price or price <= 0:
            return {}
        is_us = code.startswith("us")
        if is_us:
            prev = safe_float(p[5])       # 美股：索引5 是昨收
            chg = safe_float(p[4])        # 索引4 是涨跌额
            pct = (chg / prev * 100) if (prev == prev and prev) else float("nan")
        else:
            prev = safe_float(p[4])
            pct = safe_float(p[5])
        if pct != pct and prev == prev and prev:
            pct = (price / prev - 1) * 100
        return {"name": str(p[1] or "").strip(), "price": price, "prev_close": prev,
                "pct_chg": pct,
                "high": safe_float(p[33]) if len(p) > 33 else float("nan"),
                "low": safe_float(p[34]) if len(p) > 34 else float("nan"),
                "source": "tencent"}
    except Exception as exc:  # noqa: BLE001
        log.debug("指数行情 %s 失败: %s", code, exc)
        return {}


def _quote_hf(code: str, parts: list) -> dict:
    """解析 ``hf_`` 前缀行情（外盘期货，字段与股票完全不同）。"""
    try:
        # hf_ 行情是单字段逗号分隔：现价,昨收?,...,名称,...
        fields = parts[0].split(",") if parts else []
        if len(fields) < 2:
            return {}
        price = safe_float(fields[0])
        prev = safe_float(fields[7]) if len(fields) > 7 else float("nan")
        if prev != prev or not prev:
            prev = safe_float(fields[1])
        pct = (price / prev - 1) * 100 if (prev == prev and prev) else float("nan")
        high = safe_float(fields[4]) if len(fields) > 4 else float("nan")
        low = safe_float(fields[5]) if len(fields) > 5 else float("nan")
        if price != price or price <= 0:
            return {}
        return {"name": str(fields[-2] if len(fields) > 2 else code), "price": price,
                "prev_close": prev, "pct_chg": pct, "high": high, "low": low,
                "source": "tencent.hf"}
    except Exception as exc:  # noqa: BLE001
        log.debug("hf 行情解析失败 %s: %s", code, exc)
        return {}


def enrich_snapshot_with_history(snapshot: dict, index_frames: dict,
                                 stocks: dict | None = None) -> dict:
    """用**日线收盘序列**校准实时行情的涨跌幅。

    为什么必须这么做：腾讯行情接口里"涨跌幅"所在的字段位置因市场而异
    （A股/港股/美股/指数各不相同），靠猜索引会把"现价"当成"涨跌幅"
    （实测出现过 +3804% 这种荒谬值）。日线收盘价是**可靠且已验证**的，
    因此统一用 ``今日收盘 / 昨日收盘 - 1`` 重算，覆盖掉接口给的不可靠字段。

    ``stocks`` 可传入 ``{code: 日线DataFrame}`` 用来处理个股本身。
    """
    out = dict(snapshot or {})
    frames = dict(index_frames or {})
    if stocks:
        frames = {**frames, **stocks}
    for key, snap in out.items():
        if not isinstance(snap, dict):
            continue
        df = frames.get(key)
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2:
            # 没有日线就退化为"接口给的涨跌幅"，但做合理性校验
            pct = safe_float(snap.get("pct_chg"))
            if pct == pct and abs(pct) > 30:      # 明显是把价格当成了百分比
                snap["pct_chg"] = float("nan")
                snap["pct_note"] = "接口涨跌幅字段不可靠，已置空"
            continue
        try:
            c = df["close"].astype(float).dropna()
            if len(c) < 2:
                continue
            last, prev = float(c.iloc[-1]), float(c.iloc[-2])
            if prev > 0:
                snap["pct_chg"] = (last / prev - 1) * 100
                snap["prev_close"] = prev
                snap["price"] = last
                snap["pct_from"] = "日线重算"
                snap["last_date"] = str(pd.to_datetime(df["date"].iloc[-1]))[:10]
        except Exception as exc:  # noqa: BLE001
            log.debug("校准 %s 涨跌幅失败: %s", key, exc)
    return out


def _kline_tencent(code: str, count: int = 300, period: str = "day") -> pd.DataFrame:
    """腾讯指数K线。

    **美股指数必须走 usfqkline 接口**——用 fqkline 只会返回当天 1 行
    （这是实测踩到的坑，会导致美股相关性分析完全无法计算）。
    """
    import json as _json

    paths = (["appstock/app/usfqkline/get", "appstock/app/fqkline/get"]
             if code.startswith("us") else ["appstock/app/fqkline/get"])
    for path in paths:
        try:
            u = ("https://web.ifzq.gtimg.cn/%s?param=%s,%s,,,%d,qfq"
                 % (path, code, period, count))
            j = _json.loads(http_get(u, timeout=12, retries=2))
            d = (j.get("data") or {}).get(code) or {}
            k = d.get("qfqday") or d.get("day") or []
            if len(k) < 5:            # 少于 5 根视为失败，换下一个接口
                continue
            rows = []
            for r in k:
                if len(r) < 6:
                    continue
                rows.append({"date": pd.Timestamp(str(r[0])), "open": safe_float(r[1]),
                             "close": safe_float(r[2]), "high": safe_float(r[3]),
                             "low": safe_float(r[4]), "volume": safe_float(r[5])})
            df = pd.DataFrame(rows).dropna(subset=["close"])
            df = df[df["close"] > 0].reset_index(drop=True)
            if len(df) >= 5:
                return df
        except Exception as exc:  # noqa: BLE001
            log.debug("指数K线 %s (%s) 失败: %s", code, path, exc)
    return pd.DataFrame()


def _kline_akshare(ak_code: str, days: int = 500) -> pd.DataFrame:
    """akshare 指数日线（A股最可靠，历史最长）。"""
    try:
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol=ak_code)
        if df is None or not len(df):
            return pd.DataFrame()
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        return df.tail(days).reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("akshare 指数 %s 失败: %s", ak_code, exc)
        return pd.DataFrame()


def fetch_index_history(key: str, days: int = 400, force: bool = False) -> pd.DataFrame:
    """取单个指数的日线历史（A股优先 akshare，其余用腾讯）。"""
    spec = INDICES.get(key)
    if not spec:
        return pd.DataFrame()
    name, qcode, kcode, ak_code, group = spec

    def _p() -> pd.DataFrame:
        df = pd.DataFrame()
        if ak_code:
            df = _kline_akshare(ak_code, max(days, 260))
        if (df is None or df.empty) and kcode:
            df = _kline_tencent(kcode, min(max(days, 120), 800))
        if df is not None and not df.empty:
            df.attrs["name"] = name
            df.attrs["group"] = group
        return df if df is not None else pd.DataFrame()

    try:
        if force:
            return _p()
        key_c = cache_key("gidx", key, days)
        got = cached(key_c, config.RUNTIME["cache_ttl_seconds"], _p)
        return got if isinstance(got, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        log.debug("指数历史 %s 降级: %s", key, exc)
        return pd.DataFrame()


def fetch_global_snapshot(force: bool = False) -> dict:
    """并发取全部指数的实时行情（用于当日/隔夜影响分析）。"""
    key = cache_key("gsnap")

    def _p() -> dict:
        out: dict = {}
        for k, spec in INDICES.items():
            name, qcode = spec[0], spec[1]
            got = retry_call(lambda q=qcode: _quote_any(q), attempts=2, delay=0.4,
                             default={}, label="quote_%s" % k)
            if got:
                out[k] = {**got, "name": name, "group": spec[4]}
        return out

    try:
        if force:
            return _p()
        got = cached(key, 300, _p)       # 5 分钟，指数盘中会变
        return got if isinstance(got, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.debug("全球快照降级: %s", exc)
        return {}


# --------------------------------------------------------------------------- #
# 联动分析
# --------------------------------------------------------------------------- #
def _pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return float("nan")
    a, b = a[ok], b[ok]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def linkage_analysis(stock_df: pd.DataFrame, index_frames: dict,
                     horizons=(1, 5)) -> dict:
    """检验"各外围指数 → 该股"的真实联动关系（不是想当然的假设）。

    对每个指数分别算两类相关性：

    * **同向（contemporaneous）**：指数当日涨跌 ↔ 个股当日涨跌
    * **领先（lead）**：指数**前一日**涨跌 ↔ 个股当日涨跌
      —— 这才是可用于预测的关系（美股隔夜效应就属于这一类）

    Returns
    -------
    dict：``{key: {"name","group","same_day_corr","lead_corr","best_horizon",...}}``
    """
    out: dict = {}
    if stock_df is None or len(stock_df) < 60 or "close" not in stock_df.columns:
        return out
    sd = stock_df.copy()
    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd = sd.dropna(subset=["date", "close"]).sort_values("date")
    sr = sd["close"].astype(float).pct_change().to_numpy() * 100

    for key, frame in (index_frames or {}).items():
        if frame is None or not isinstance(frame, pd.DataFrame) or len(frame) < 40:
            continue
        spec = INDICES.get(key) or ("", "", "", "", "")
        try:
            f = frame.copy()
            f["date"] = pd.to_datetime(f["date"], errors="coerce")
            f = f.dropna(subset=["date", "close"]).sort_values("date")
            f["ir"] = f["close"].astype(float).pct_change() * 100
            m = pd.merge(sd[["date", "close"]], f[["date", "ir"]], on="date", how="left")
            ir = m["ir"].to_numpy(dtype=float)
            n = len(ir)
            same = _pearson(ir, sr)
            # 领先一期：指数 t-1 的涨跌 vs 个股 t 的涨跌
            lead = _pearson(np.concatenate([[np.nan], ir[:-1]]), sr) if n > 2 else float("nan")
            # 多日累计领先（指数近 h 日累计涨跌 vs 个股当日）
            lead_h = {}
            for h in horizons:
                if n > h + 20:
                    cum = pd.Series(ir).rolling(h).sum().to_numpy()
                    lead_h[h] = _pearson(cum, sr)
            best_h = max(lead_h.items(), key=lambda kv: abs(kv[1]))[0] if lead_h else None
            n_valid = int(np.isfinite(ir).sum())
            # 最近的累计涨跌（近5日/近20日）
            last5 = float(np.nansum(ir[-5:])) if n >= 5 else float("nan")
            last20 = float(np.nansum(ir[-20:])) if n >= 20 else float("nan")
            out[key] = {
                "name": spec[0], "group": spec[4],
                "same_day_corr": same, "lead_corr": lead,
                "lead_corr_by_h": lead_h, "best_horizon": best_h,
                "recent_5d_pct": last5, "recent_20d_pct": last20,
                "n": n_valid, "last_date": str(f["date"].iloc[-1])[:10],
            }
        except Exception as exc:  # noqa: BLE001
            log.debug("联动分析 %s 失败: %s", key, exc)
    return out


def relative_strength(stock_df: pd.DataFrame, bench_df: pd.DataFrame,
                      bench_name: str = "上证指数") -> dict:
    """个股相对大盘的强度（近 5/20/60 日超额收益 + Beta + 相关）。"""
    out: dict = {}
    try:
        s = stock_df.copy()
        b = bench_df.copy()
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        b["date"] = pd.to_datetime(b["date"], errors="coerce")
        m = pd.merge(s[["date", "close"]].rename(columns={"close": "sc"}),
                     b[["date", "close"]].rename(columns={"close": "bc"}),
                     on="date", how="inner").dropna().sort_values("date")
        if len(m) < 30:
            return out
        sr = m["sc"].astype(float).pct_change()
        br = m["bc"].astype(float).pct_change()
        for h in (5, 20, 60):
            if len(m) > h:
                sret = float(m["sc"].iloc[-1] / m["sc"].iloc[-1 - h] - 1) * 100
                bret = float(m["bc"].iloc[-1] / m["bc"].iloc[-1 - h] - 1) * 100
                out["excess_%dd" % h] = round(sret - bret, 2)
                out["stock_%dd" % h] = round(sret, 2)
                out["bench_%dd" % h] = round(bret, 2)
        a, bb = sr.to_numpy()[-250:], br.to_numpy()[-250:]
        ok = np.isfinite(a) & np.isfinite(bb)
        if ok.sum() > 40 and np.var(bb[ok]) > 1e-14:
            out["beta_250d"] = round(float(np.cov(a[ok], bb[ok])[0, 1] / np.var(bb[ok])), 3)
            corr = round(_pearson(a, bb), 3)
            out["corr_250d"] = corr
            # 相关性很低时明确提示"该股近期与大盘脱钩"，避免用户误以为大盘能带动它
            if corr == corr and abs(corr) < 0.15:
                out["decoupled"] = True
                out["corr_note"] = ("该股近 250 日与%s相关性仅 %.2f，已明显脱钩，"
                                    "大盘涨跌对其解释力很弱" % (bench_name, corr))
        out["bench_name"] = bench_name
    except Exception as exc:  # noqa: BLE001
        log.debug("相对强度计算失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 因子
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# A 股指数之间的相互印证 / 顶底背离
# --------------------------------------------------------------------------- #
#: 相互印证时使用的核心指数（同属 A 股，理应高度同向）
CROSS_KEYS = ("sh", "sz", "cyb", "hs300", "zz500", "kc50")
#: 大盘股 vs 小盘股的代表（用于判断"二八分化"）
LARGE_CAP_KEYS = ("hs300", "sh")
SMALL_CAP_KEYS = ("zz500", "cyb", "kc50")


def cross_index_analysis(index_frames: dict, snapshot: dict | None = None,
                         lookback: int = 120) -> dict:
    """分析**沪深各大指数之间的相互印证与背离**。

    三件事
    ------
    1. **同向性（印证）**：近 1/5/20 日各指数涨跌方向是否一致。
       全部同向 = 系统性行情，个股跟随概率高；
       上证涨而创业板跌 = 结构性行情，选错方向的代价很大。
    2. **顶/底背离**：比较"创新高/新低的指数个数"与指数整体涨幅。
       例如多数指数未创新高而上证创新高 → 顶背离（上涨不牢）；
       多数指数未创新低而某指数创新低 → 底背离（下跌过度）。
    3. **二八分化**：大盘股指数与小盘股指数的强弱差（近 20 日），
       判断当前是"权重搭台"还是"成长唱戏"。

    Returns
    -------
    dict：``{"agreement","direction","divergence","style","detail","rows"}``
    """
    out: dict = {"agreement": {}, "divergence": "none", "style": "均衡",
                 "detail": "", "rows": []}
    frames = {k: v for k, v in (index_frames or {}).items()
              if k in CROSS_KEYS and isinstance(v, pd.DataFrame) and len(v) >= 30}
    if len(frames) < 2:
        out["detail"] = "A股指数数据不足，无法做相互印证"
        return out
    snap = snapshot or {}

    # ---- 1) 各周期涨跌幅与方向 ---- #
    rows = []
    for k in CROSS_KEYS:
        df = frames.get(k)
        if df is None or df.empty:
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
            new_high = bool(c.iloc[-1] >= hi - 1e-9)
            new_low = bool(c.iloc[-1] <= lo + 1e-9)
            snap_pct = safe_float((snap.get(k) or {}).get("pct_chg"))
            rows.append({"key": k, "name": INDICES.get(k, ("",))[0],
                         "pct_1d": round(r1, 2), "pct_5d": round(r5, 2),
                         "pct_20d": round(r20, 2),
                         "new_high": new_high, "new_low": new_low,
                         "snap_pct": snap_pct if snap_pct == snap_pct else None})
        except Exception as exc:  # noqa: BLE001
            log.debug("指数 %s 交叉分析失败: %s", k, exc)
    out["rows"] = rows
    if len(rows) < 2:
        out["detail"] = "有效指数不足，无法做相互印证"
        return out

    # ---- 2) 同向性（印证）---- #
    for horizon, fld in (("1日", "pct_1d"), ("5日", "pct_5d"), ("20日", "pct_20d")):
        vals = [r[fld] for r in rows if r.get(fld) is not None]
        if not vals:
            continue
        ups = sum(1 for v in vals if v > 0)
        downs = sum(1 for v in vals if v < 0)
        total = ups + downs
        if total == 0:
            continue
        agree = max(ups, downs) / total
        out["agreement"][horizon] = {
            "up": ups, "down": downs, "total": total,
            "ratio": round(agree, 3),
            "aligned": bool(agree >= 0.8),
            "direction": 1 if ups > downs else (-1 if downs > ups else 0),
        }

    # ---- 3) 顶/底背离 ---- #
    n_high = sum(1 for r in rows if r.get("new_high"))
    n_low = sum(1 for r in rows if r.get("new_low"))
    avg20 = sum(r["pct_20d"] for r in rows) / len(rows)
    divergence = "none"
    div_detail = ""
    if n_high == 1 and len(rows) >= 3 and avg20 > 0:
        # 只有一个指数创新高、整体涨幅温和 → 上涨不具广泛性
        divergence = "top"
        div_detail = ("仅「%s」创出一年新高，其余 %d 个指数未跟上，"
                      "上涨缺乏广泛性，存在顶背离风险"
                      % (next(r["name"] for r in rows if r["new_high"]),
                         len(rows) - 1))
    elif n_low == 1 and len(rows) >= 3 and avg20 < 0:
        divergence = "bottom"
        div_detail = ("仅「%s」创出一年新低，其余 %d 个指数未同步走弱，"
                      "属于超跌背离，往往先于市场企稳"
                      % (next(r["name"] for r in rows if r["new_low"]),
                         len(rows) - 1))
    elif n_high >= len(rows) - 1 and avg20 > 0:
        div_detail = "绝大多数指数同步创出一年新高，属全面强势（非背离）"
    elif n_low >= len(rows) - 1 and avg20 < 0:
        div_detail = "绝大多数指数同步创出一年新低，属全面弱势（非背离）"
    out["divergence"] = divergence
    out["new_high_count"] = n_high
    out["new_low_count"] = n_low
    out["divergence_detail"] = div_detail

    # ---- 4) 二八分化 ---- #
    def _avg(keys):
        vals = [r["pct_20d"] for r in rows if r["key"] in keys]
        return sum(vals) / len(vals) if vals else float("nan")

    large, small = _avg(LARGE_CAP_KEYS), _avg(SMALL_CAP_KEYS)
    style = "均衡"
    spread = float("nan")
    if large == large and small == small:
        spread = large - small
        if spread > 2.0:
            style = "大盘权重占优（二八分化，资金偏防御）"
        elif spread < -2.0:
            style = "中小盘成长占优（资金偏进攻）"
        else:
            style = "大小盘基本同步"
    out["style"] = style
    out["style_spread"] = round(spread, 2) if spread == spread else None
    out["large_cap_pct"] = round(large, 2) if large == large else None
    out["small_cap_pct"] = round(small, 2) if small == small else None

    a5 = out["agreement"].get("5日") or {}
    out["detail"] = ("A股主要指数近5日 %s（%d涨%d跌，一致度 %.0f%%）；%s%s"
                     % ("同向" if a5.get("aligned") else "分歧",
                        a5.get("up", 0), a5.get("down", 0),
                        (a5.get("ratio") or 0) * 100,
                        (div_detail + "；") if div_detail else "",
                        "风格上" + style))
    return out


def cross_index_factors(cross: dict, symbol_names=None) -> list:
    """把"沪深指数相互印证/背离"转成因子。"""
    out: list[Factor] = []
    if not isinstance(cross, dict) or not cross.get("agreement"):
        return out
    a5 = cross["agreement"].get("5日") or {}
    a1 = cross["agreement"].get("1日") or {}
    ratio = float(a5.get("ratio") or 0)
    direction = int(a5.get("direction") or 0)
    if direction:
        strength = 0.3 + max(0.0, ratio - 0.5) * 1.2
        out.append(Factor(
            id="index_cross_agreement",
            name="沪深指数%s" % ("同向印证" if a5.get("aligned") else "方向分歧"),
            category=Category.MACRO,
            direction=DIR_BULL if direction > 0 else DIR_BEAR,
            strength=min(strength, 0.85),
            reliability=0.72 if a5.get("aligned") else 0.5,
            value="近5日 %d涨%d跌（一致度 %.0f%%）；近1日 %d涨%d跌"
                  % (a5.get("up", 0), a5.get("down", 0), ratio * 100,
                     a1.get("up", 0), a1.get("down", 0)),
            detail=("沪深主要指数近5日%s，%s"
                    % ("高度同向，属系统性行情，个股跟随大盘的概率高"
                       if a5.get("aligned") else
                       "方向分歧，属结构性行情，选错风格的代价很大",
                       "整体" + ("向上" if direction > 0 else "向下"))),
            theory="指数相互印证/道氏理论", horizon=("d1", "d5", "d10", "d20")))
    # 顶/底背离
    div = cross.get("divergence")
    if div == "top":
        out.append(Factor(
            id="index_divergence_cross", name="A股指数顶背离",
            category=Category.MACRO, direction=DIR_BEAR, strength=0.6, reliability=0.62,
            value="仅 %d 个指数创新高" % cross.get("new_high_count", 0),
            detail=str(cross.get("divergence_detail") or "") +
                   "。历史上「独强」行情往往难以持续，需警惕高位补跌",
            theory="道氏理论（指数须相互印证）", horizon=("d5", "d10", "d20", "m1")))
    elif div == "bottom":
        out.append(Factor(
            id="index_divergence_cross", name="A股指数底背离",
            category=Category.MACRO, direction=DIR_BULL, strength=0.55, reliability=0.6,
            value="仅 %d 个指数创新低" % cross.get("new_low_count", 0),
            detail=str(cross.get("divergence_detail") or "") +
                   "。道氏理论认为指数应相互印证，单独走弱常属超跌",
            theory="道氏理论（指数须相互印证）", horizon=("d5", "d10", "d20")))
    # 二八分化
    sp = safe_float(cross.get("style_spread"))
    if sp == sp and abs(sp) > 2.0:
        big = sp > 0
        out.append(Factor(
            id="index_style_rotation", name="市场风格：%s" % cross.get("style"),
            category=Category.MACRO,
            direction=DIR_NEUTRAL, strength=min(abs(sp) / 8.0, 0.6), reliability=0.6,
            value="近20日 大盘权重 %+.2f%% vs 中小盘 %+.2f%%（差 %.2f pp）"
                  % (cross.get("large_cap_pct") or 0, cross.get("small_cap_pct") or 0, sp),
            detail=("当前资金%s，%s"
                    % ("偏好大盘权重股（防御特征）" if big else "偏好中小盘成长股（进攻特征）",
                       "若你的股票与当前风格不符，即使大盘上涨也可能不涨")),
            theory="风格轮动/二八分化", horizon=("d10", "d20", "m1")))
    return out


def global_market_factors(stock_df: pd.DataFrame, snapshot: dict,
                          index_frames: dict | None = None,
                          linkage: dict | None = None,
                          relative: dict | None = None) -> list:
    """把大盘/港美股的外围影响转成因子。

    Parameters
    ----------
    snapshot : :func:`fetch_global_snapshot` 的结果（实时涨跌）
    linkage : :func:`linkage_analysis` 的结果（相关性，用于给可信度定级）
    relative : :func:`relative_strength` 的结果
    """
    out: list[Factor] = []
    snap = snapshot or {}
    lk = linkage or {}

    # ---- 1) A 股大盘水位 ----
    a_keys = [k for k, v in INDICES.items() if v[4] == "a_share"]
    a_pcts = [(snap[k]["name"], safe_float(snap[k].get("pct_chg")))
              for k in a_keys if k in snap]
    a_pcts = [(n, p) for n, p in a_pcts if p == p]
    if a_pcts:
        avg = sum(p for _, p in a_pcts) / len(a_pcts)
        up = sum(1 for _, p in a_pcts if p > 0)
        # 用个股与沪深300的相关性决定可信度：越相关，大盘影响越大
        corr = abs(safe_float((lk.get("hs300") or {}).get("same_day_corr")))
        rel = 0.5 + (0.35 if corr == corr else 0.15)
        strength = min(abs(avg) / 2.0, 0.85)
        if abs(avg) < 0.15:
            direction, strength = DIR_NEUTRAL, 0.15
        else:
            direction = DIR_BULL if avg > 0 else DIR_BEAR
        out.append(Factor(
            id="market_a_share", name="A股大盘环境", category=Category.MACRO,
            direction=direction, strength=strength, reliability=min(rel, 0.9),
            value="；".join("%s %+.2f%%" % (n, p) for n, p in a_pcts[:6]),
            detail="A股主要指数当日平均%s %.2f%%（%d涨%d跌）；该股与沪深300相关性 %.2f，"
                   "大盘环境对个股构成系统性%s" % (
                       "上涨" if avg > 0 else ("下跌" if avg < 0 else "走平"), abs(avg),
                       up, len(a_pcts) - up, corr if corr == corr else 0.0,
                       "支撑" if avg > 0 else "压力"),
            theory="市场系统性风险/同市场联动", horizon=None))

    # ---- 2) 港股影响（同时段交易，联动直接） ----
    hk = [(snap[k]["name"], safe_float(snap[k].get("pct_chg")))
          for k in ("hsi", "hstech", "hscei") if k in snap]
    hk = [(n, p) for n, p in hk if p == p]
    if hk:
        avg = sum(p for _, p in hk) / len(hk)
        hk_corr = max((abs(safe_float((lk.get(k) or {}).get("same_day_corr")))
                       for k in ("hsi", "hstech", "hscei")), default=float("nan"))
        direction = DIR_NEUTRAL if abs(avg) < 0.2 else (DIR_BULL if avg > 0 else DIR_BEAR)
        out.append(Factor(
            id="market_hk", name="港股市场影响", category=Category.MACRO,
            direction=direction, strength=min(abs(avg) / 2.2, 0.75),
            reliability=0.6 if not (hk_corr == hk_corr) else min(0.55 + hk_corr * 0.4, 0.88),
            value="；".join("%s %+.2f%%" % (n, p) for n, p in hk),
            detail="港股（%s）当日平均%s %.2f%%；港股与A股共享中国基本面且交易时段重叠，"
                   "是同日情绪的重要参照%s" % (
                       "、".join(n for n, _ in hk), "上涨" if avg > 0 else "下跌", abs(avg),
                       "（该股与港股相关性 %.2f）" % hk_corr if hk_corr == hk_corr else ""),
            theory="跨市场联动", horizon=None))

    # ---- 3) 美股隔夜效应（领先信号：美股收盘早于A股开盘） ----
    us = [(snap[k]["name"], safe_float(snap[k].get("pct_chg")))
          for k in ("dji", "ixic", "spx") if k in snap]
    us = [(n, p) for n, p in us if p == p]
    if us:
        avg = sum(p for _, p in us) / len(us)
        lead = max((abs(safe_float((lk.get(k) or {}).get("lead_corr")))
                    for k in ("dji", "ixic", "spx")), default=float("nan"))
        direction = DIR_NEUTRAL if abs(avg) < 0.25 else (DIR_BULL if avg > 0 else DIR_BEAR)
        out.append(Factor(
            id="market_us_overnight", name="美股隔夜影响", category=Category.MACRO,
            direction=direction, strength=min(abs(avg) / 2.5, 0.7),
            reliability=0.55 if not (lead == lead) else min(0.5 + lead * 0.5, 0.85),
            value="；".join("%s %+.2f%%" % (n, p) for n, p in us),
            detail="美股上一交易日（%s）平均%s %.2f%%。美股收盘早于A股开盘，"
                   "其涨跌在A股开盘前已确定，属**领先信号**%s" % (
                       "、".join(n for n, _ in us), "上涨" if avg > 0 else "下跌", abs(avg),
                       "（领先相关性 %.2f）" % lead if lead == lead else ""),
            theory="隔夜效应/领先滞后关系", horizon=("d1", "d5", "d10")))

    # ---- 4) 全球风险偏好（VIX） ----
    vix = snap.get("vix") or {}
    vp = safe_float(vix.get("price"))
    if vp == vp:
        if vp >= 30:
            d, s, lv = DIR_BEAR, 0.75, "极度恐慌"
        elif vp >= 22:
            d, s, lv = DIR_BEAR, 0.5, "偏恐慌"
        elif vp <= 13:
            d, s, lv = DIR_BULL, 0.4, "极度乐观"
        elif vp <= 17:
            d, s, lv = DIR_BULL, 0.3, "偏乐观"
        else:
            d, s, lv = DIR_NEUTRAL, 0.2, "中性"
        out.append(Factor(
            id="global_risk_appetite", name="全球风险偏好（VIX）",
            category=Category.MACRO, direction=d, strength=s, reliability=0.6,
            value="VIX=%.2f（%s）" % (vp, lv),
            detail="VIX=%.2f 属「%s」区间。VIX 高位时全球资金避险，"
                   "对A股（尤其高估值成长股）构成压制；低位则风险偏好回暖" % (vp, lv),
            theory="风险偏好/波动率", horizon=("d5", "d10", "d20", "m1")))

    # ---- 5) A50 期货（海外对A股的定价） ----
    a50 = snap.get("a50") or {}
    ap = safe_float(a50.get("price"))
    aprev = safe_float(a50.get("prev_close"))
    if ap == ap and aprev == aprev and aprev:
        pct = (ap / aprev - 1) * 100
        d = DIR_NEUTRAL if abs(pct) < 0.2 else (DIR_BULL if pct > 0 else DIR_BEAR)
        out.append(Factor(
            id="market_a50", name="富时中国A50期货", category=Category.MACRO,
            direction=d, strength=min(abs(pct) / 2.0, 0.65), reliability=0.55,
            value="%.1f（%+.2f%%）" % (ap, pct),
            detail="A50 期货是海外资金对A股大盘的定价，常领先沪深300现货；"
                   "当前 %+.2f%%，%s" % (pct, "指向偏暖" if pct > 0 else "指向偏弱"),
            theory="领先指标/跨市场定价", horizon=("d1", "d5")))

    # ---- 6) 相对强度（个股 vs 大盘） ----
    rel = relative or {}
    ex20 = safe_float(rel.get("excess_20d"))
    if ex20 == ex20:
        d = DIR_NEUTRAL
        if ex20 > 5:
            d = DIR_BULL
        elif ex20 < -5:
            d = DIR_BEAR
        out.append(Factor(
            id="relative_strength", name="相对大盘强度", category=Category.MACRO,
            direction=d, strength=min(abs(ex20) / 20.0, 0.8), reliability=0.75,
            value="近20日超额 %+.2f%%（个股 %+.2f%% vs %s %+.2f%%）" % (
                ex20, safe_float(rel.get("stock_20d")), rel.get("bench_name", "大盘"),
                safe_float(rel.get("bench_20d"))),
            detail="个股近20日相对%s%s %.2f 个百分点%s；Beta=%.2f，相关性=%.2f" % (
                rel.get("bench_name", "大盘"), "跑赢" if ex20 > 0 else "跑输", abs(ex20),
                "，说明资金关注度高、相对强势" if ex20 > 5 else
                ("，弱势特征明显，反弹易受大盘拖累" if ex20 < -5 else ""),
                safe_float(rel.get("beta_250d")), safe_float(rel.get("corr_250d"))),
            theory="相对强度理论", horizon=("d5", "d10", "d20", "m1")))

    # ---- 7) 联动最紧密的单一指数（给用户一个"看哪个指数"的结论）----
    if lk:
        try:
            best = max(lk.items(),
                       key=lambda kv: abs(safe_float((kv[1] or {}).get("lead_corr")) or 0))
            bd = best[1]
            lead = safe_float(bd.get("lead_corr"))
            if lead == lead and abs(lead) > 0.05:
                d = DIR_NEUTRAL
                rec = safe_float(bd.get("recent_5d_pct"))
                if rec == rec and abs(rec) > 0.5:
                    d = DIR_BULL if (rec > 0) == (lead > 0) else DIR_BEAR
                out.append(Factor(
                    id="key_linkage_index", name="最强联动指数", category=Category.MACRO,
                    direction=d, strength=min(abs(lead) * 1.6, 0.6), reliability=0.6,
                    value="%s（领先相关 %.2f，近5日 %+.2f%%）" % (
                        bd.get("name"), lead, rec if rec == rec else 0.0),
                    detail="历史上与该股联动最强的是「%s」：它前一日涨跌与个股当日涨跌的"
                           "相关性为 %.2f（样本 %d 天）%s" % (
                               bd.get("name"), lead, bd.get("n", 0),
                               "，可作为开盘前的重点参考" if abs(lead) > 0.15 else ""),
                    theory="领先滞后关系", horizon=("d1", "d5")))
        except Exception as exc:  # noqa: BLE001
            log.debug("最强联动指数判定失败: %s", exc)
    return out


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def analyze_market_influence(stock_df: pd.DataFrame, days: int = 400,
                             force: bool = False, verbose: bool = False) -> dict:
    """一站式：取数 → 联动检验 → 相对强度 → 因子。

    Returns
    -------
    dict：``{"snapshot","indices","linkage","relative","factors","detail"}``
    """
    result: dict = {"snapshot": {}, "indices": {}, "linkage": {}, "relative": {},
                    "factors": [], "detail": ""}
    try:
        snap = fetch_global_snapshot(force=force)
        result["snapshot"] = snap
        frames: dict = {}
        for k in INDICES:
            df = fetch_index_history(k, days=days, force=force)
            if df is not None and not df.empty:
                frames[k] = df
        result["indices"] = frames
        # 用日线收盘校准实时涨跌幅（接口字段位置不可靠）
        snap = enrich_snapshot_with_history(snap, frames)
        result["snapshot"] = snap
        lk = linkage_analysis(stock_df, frames)
        result["linkage"] = lk
        # 相对强度用沪深300（其次上证指数）
        bench = frames.get("hs300")
        bench_name = "沪深300"
        if bench is None or bench.empty:
            bench, bench_name = frames.get("sh"), "上证指数"
        if bench is not None and not bench.empty:
            result["relative"] = relative_strength(stock_df, bench, bench_name)
        # A股指数之间的相互印证 / 顶底背离（与个股基准无关，只要指数够就算）
        cross: dict = {}
        try:
            cross = cross_index_analysis(frames, snap)
            result["cross"] = cross
        except Exception as exc:  # noqa: BLE001
            log.debug("指数交叉印证失败: %s", exc)
        result["factors"] = global_market_factors(stock_df, snap, frames, lk,
                                                 result["relative"])
        try:
            result["factors"] = list(result["factors"]) + cross_index_factors(cross)
        except Exception as exc:  # noqa: BLE001
            log.debug("交叉印证因子生成失败: %s", exc)
        got = [v[0] for v in INDICES.values() if v[0]]
        result["detail"] = ("已覆盖 %d 个指数（%s），成功取到历史 %d 个、实时 %d 个"
                           % (len(INDICES), "、".join(GROUP_LABEL.values()),
                              len(frames), len(snap)))
    except Exception as exc:  # noqa: BLE001
        log.warning("大盘/外围分析失败: %s", exc)
        result["detail"] = "大盘与外围分析失败：%s" % exc
    return result


if __name__ == "__main__":  # pragma: no cover
    import time

    from .selftest_data import make_ohlcv

    print("=== 指数清单 ===")
    for grp in ("a_share", "hk", "us", "risk", "a50"):
        names = [v[0] for v in INDICES.values() if v[4] == grp]
        print("  %-14s %s" % (GROUP_LABEL[grp], "、".join(names)))

    print("\n=== 离线自测（合成数据）===")
    df = make_ohlcv(400, seed=7)
    fake_snap = {
        "sh": {"name": "上证指数", "pct_chg": 1.2, "price": 3800},
        "sz": {"name": "深证成指", "pct_chg": 1.5, "price": 12600},
        "cyb": {"name": "创业板指", "pct_chg": 2.0, "price": 3040},
        "hs300": {"name": "沪深300", "pct_chg": 1.1, "price": 4300},
        "hsi": {"name": "恒生指数", "pct_chg": 0.8, "price": 24000},
        "hstech": {"name": "恒生科技", "pct_chg": 1.6, "price": 4100},
        "dji": {"name": "道琼斯", "pct_chg": -0.4, "price": 51000},
        "ixic": {"name": "纳斯达克", "pct_chg": -0.9, "price": 27000},
        "vix": {"name": "VIX", "pct_chg": 3.0, "price": 24.5},
        "a50": {"name": "A50", "pct_chg": 0.6, "price": 13700, "prev_close": 13620},
    }
    idx = make_ohlcv(400, seed=11)
    idx = idx.rename(columns={"close": "close"})[["date", "open", "high", "low",
                                                  "close", "volume"]]
    fake_linkage = {"hs300": {"name": "沪深300", "group": "a_share",
                              "same_day_corr": 0.62, "lead_corr": 0.08,
                              "recent_5d_pct": 2.1, "n": 380},
                    "ixic": {"name": "纳斯达克", "group": "us",
                             "same_day_corr": 0.21, "lead_corr": 0.19,
                             "recent_5d_pct": -1.4, "n": 380}}
    fs = global_market_factors(df, fake_snap, {"hs300": idx}, fake_linkage,
                               {"excess_20d": 7.5, "stock_20d": 9.0, "bench_20d": 1.5,
                                "bench_name": "沪深300", "beta_250d": 0.93,
                                "corr_250d": 0.62})
    print("因子 %d 条：" % len(fs))
    for f in fs:
        print("  [%s] %-18s %-4s 强度%.2f 可信度%.2f | %s" % (
            f.category, f.name, {1: "多", -1: "空", 0: "中"}[f.direction],
            f.strength, f.reliability, f.detail[:72]))
    ids = {f.id for f in fs}
    need = {"market_a_share", "market_hk", "market_us_overnight",
            "global_risk_appetite", "market_a50", "relative_strength",
            "key_linkage_index"}
    assert need <= ids, "缺少因子：%s" % (need - ids)
    assert all(0 <= f.strength <= 1 and 0 <= f.reliability <= 1 for f in fs)
    assert all(f.theory for f in fs), "因子缺少理论依据"
    print("\n离线自测通过（%d 条因子全部具备方向/强度/可信度/理论依据）" % len(fs))

    print("\n=== 联网自测 ===")
    t0 = time.time()
    got = analyze_market_influence(df, days=300)
    print("取到历史指数 %d 个：" % len(got["indices"]))
    for k, v in got["indices"].items():
        print("    %-8s %-10s rows=%-5d %s" % (k, INDICES[k][0], len(v),
                                               str(v["date"].iloc[-1])[:10]))
    print("实时行情 %d 个：%s" % (len(got["snapshot"]),
                                "、".join(v.get("name", k)
                                          for k, v in got["snapshot"].items())))
    print("联动分析 %d 个：" % len(got["linkage"]))
    for k, v in sorted(got["linkage"].items(),
                       key=lambda kv: -abs(kv[1].get("lead_corr") or 0))[:6]:
        print("    %-10s 同日相关 %+.3f 领先相关 %+.3f (n=%d)" % (
            v["name"], v.get("same_day_corr") or 0, v.get("lead_corr") or 0, v.get("n", 0)))
    print("相对强度：", {k: v for k, v in (got["relative"] or {}).items()
                        if k.startswith("excess") or k in ("beta_250d", "corr_250d")})
    print("因子 %d 条" % len(got["factors"]))
    print("耗时 %.1fs" % (time.time() - t0))
    print("\nglobal_market self-test OK")

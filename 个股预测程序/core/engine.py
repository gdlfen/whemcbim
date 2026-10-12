# -*- coding: utf-8 -*-
"""
分析编排引擎 —— 把"取数 → 八大维度分析 → 因子汇总 → 概率预测 → 报告"串成一条流水线。

设计原则
--------
1. **任何单点失败都不影响最终结果**：每个分析模块都单独 try/except，
   失败时把异常记进 ``bundle.errors`` 并在报告中标注"该项数据缺失"，
   其余维度继续工作。预测引擎只依赖"有哪些因子"，不依赖"因子从哪来"。
2. **数据源可降级**：联网失败 → 用缓存 → 仍失败 → 明确报错（或演示用合成数据）。
3. **单一入口**：无论 CLI、Streamlit 还是批处理，都调用 :class:`Predictor`，
   保证本地与线上行为一致。
"""

from __future__ import annotations

import importlib
import traceback

import pandas as pd

from . import config
from .contracts import HORIZONS, AnalysisBundle, FactorSet, PredictionResult
from .utils import get_logger, http_get, normalize_symbol, retry_call

log = get_logger()


def _mod(name: str):
    """容错导入子模块；不存在或导入失败返回 None。"""
    try:
        return importlib.import_module("." + name, __package__)
    except Exception as exc:  # noqa: BLE001
        log.debug("模块 %s 不可用: %s", name, exc)
        return None


def _call(fn, *args, **kwargs):
    """调用分析函数，自动裁掉它不接受的 kwargs（接口演进时的兼容层）。"""
    if fn is None:
        return None
    try:
        import inspect

        sig = inspect.signature(fn)
        accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        log.debug("调用 %s 失败: %s", getattr(fn, "__name__", fn), exc)
        raise


def lookup_code_by_name(keyword: str, limit: int = 8) -> list:
    """用股票名称/拼音模糊查询代码。

    为什么单独做这件事：``datasource.resolve_symbol`` 依赖一份 5573 行的全市场
    代码表，冷启动要拉 18 页、在部分网络环境下会失败。而用户在手机上手输的
    往往是"贵州茅台"这样的名称，所以这里用两个**单次请求**的联想接口兜底：

    * 新浪 suggest（支持中文名，返回 ``名称,类型,代码,sh代码,名称,...``）
    * 腾讯 smartbox（支持中文名与拼音首字母，返回 ``市场~代码~名称~拼音~类型``）

    Returns
    -------
    list[dict]，每项 ``{"code","name","prefix","display"}``，按相关度排序；失败返回 ``[]``。
    """
    kw = str(keyword or "").strip()
    if not kw:
        return []
    out: list[dict] = []
    seen: set = set()

    def _add(code: str, name: str, prefix: str = "") -> None:
        code = "".join(ch for ch in str(code) if ch.isalnum())
        if not code or code in seen:
            return
        # 只保留 A 股 6 位数字代码（港股/美股走另一条路）
        if not (len(code) == 6 and code.isdigit()):
            return
        info = normalize_symbol((prefix + code) if prefix else code)
        if not info.get("code"):
            return
        seen.add(code)
        out.append({"code": info["code"], "name": str(name or "").strip(),
                    "prefix": info["prefix"], "display": info["display"]})

    # ---- 新浪 suggest ----
    try:
        from urllib.parse import quote

        u = ("https://suggest3.sinajs.cn/suggest/type=11,12,13,14,15&key="
             + quote(kw, encoding="utf-8"))
        txt = http_get(u, timeout=6.0, retries=1,
                       headers={"Referer": "https://finance.sina.com.cn"}) or ""
        if "=" in txt:
            payload = txt.split("=", 1)[1].strip().strip('";')
            for item in payload.split(";"):
                parts = item.split(",")
                if len(parts) >= 4 and parts[2].strip().isdigit():
                    _add(parts[2], parts[0], "sh" if parts[3].startswith("sh") else "sz")
    except Exception as exc:  # noqa: BLE001
        log.debug("新浪 suggest 查询失败: %s", exc)

    # ---- 腾讯 smartbox（中文名与拼音首字母都支持）----
    if len(out) < limit:
        try:
            from urllib.parse import quote

            u = "https://smartbox.gtimg.cn/s3/?v=2&q=%s&t=all" % quote(kw, encoding="utf-8")
            raw = http_get(u, timeout=6.0, retries=1) or ""
            body = raw
            if "\\u" in body:                     # 解 unicode 转义
                body = body.encode("latin-1", "ignore").decode("unicode_escape", "ignore")
            if '"' in body:
                body = body.split('"', 1)[1].rsplit('"', 1)[0]
            for item in body.split("^"):
                parts = item.split("~")
                if len(parts) >= 4 and parts[1].strip().isdigit():
                    _add(parts[1], parts[2], parts[0].strip().lower())
        except Exception as exc:  # noqa: BLE001
            log.debug("腾讯 smartbox 查询失败: %s", exc)

    return out[:limit]


def fetch_stock_news(symbol_info: dict, name: str = "", limit: int = 20) -> list:
    """抓取**只与这只股票相关**的新闻（东财个股新闻列表接口，按股票代码过滤）。

    为什么要单独做：把"全市场搜索"的结果当个股新闻用，会出现分析海天味业却
    推送"半导体板块""某公司收购"这种完全无关的内容。这里改用按代码过滤的
    列表接口，并对标题做一次相关性复核。

    Returns
    -------
    list[dict]，每项 ``{"title","date","source","url","tag","direction","detail"}``；
    失败或全部不相关时返回 ``[]``。
    """
    info = symbol_info or {}
    code = str(info.get("code") or "").strip()
    if not code or not info.get("is_cn", True):
        return []
    mtype = "1" if info.get("prefix") == "sh" else "0"
    out: list[dict] = []
    try:
        import json as _json

        u = ("https://np-listapi.eastmoney.com/comm/web/getListInfo?client=web&biz=web_news"
             "&mTypeAndCode=%s.%s&pageSize=%d&pageIndex=1&type=1" % (mtype, code, max(limit, 10)))
        raw = http_get(u, timeout=10, retries=2,
                       headers={"Referer": "https://so.eastmoney.com/"})
        j = _json.loads(raw)
        items = ((j.get("data") or {}).get("list")) or []
    except Exception as exc:  # noqa: BLE001
        log.debug("个股新闻列表取数失败: %s", exc)
        return []

    kw = str(name or "").strip()
    for it in items:
        title = str(it.get("Art_Title") or "").strip()
        if not title:
            continue
        # 相关性复核：标题须含股票名称或代码；否则丢弃（宁可少，不可错）
        if kw and kw not in title and code not in title:
            continue
        art_code = str(it.get("Art_Code") or "")
        out.append({
            "title": title,
            "date": str(it.get("Art_ShowTime") or "")[:19],
            "source": str(it.get("Art_MediaName") or it.get("Np_dst") or "东方财富"),
            "url": ("https://finance.eastmoney.com/a/%s.html" % art_code) if art_code else "",
            "summary": str(it.get("Art_Summary") or "")[:200],
            "tag": "", "direction": 0, "detail": "",
        })
        if len(out) >= limit:
            break
    if out:
        log.info("个股新闻：从东财个股列表取得 %d 条与「%s」直接相关的新闻",
                 len(out), kw or code)
    return out


def _is_relevant(item, code: str, name: str) -> bool:
    """判断一条资讯是否与目标个股直接相关（标题含名称/代码）。"""
    if not isinstance(item, dict):
        return False
    title = str(item.get("title") or "")
    if not title:
        return False
    if name and name in title:
        return True
    if code and code in title:
        return True
    # 名称的简称匹配（如"海天味业"→"海天"），但要求长度≥2 且不是常见泛词
    short = str(name or "").strip()[:2]
    generic = {"中国", "华夏", "东方", "南方", "北方", "新能", "科技", "股份", "集团",
               "控股", "国际", "银行", "证券", "医药", "地产", "能源", "电力", "汽车"}
    return bool(len(short) >= 2 and short not in generic and short in title)


class Predictor:
    """一只股票的完整预测流程。

    Examples
    --------
    >>> p = Predictor("600519", use_ml=True)
    >>> result = p.run()               # PredictionResult
    >>> print(result.headline())

    离线演示（无网络时）：``Predictor("600519", demo=True)``
    """

    def __init__(self, symbol: str, name: str = "", *, days: int | None = None,
                 use_ml: bool = True, demo: bool = False, offline: bool = False,
                 horizons=HORIZONS, verbose: bool = True):
        self.query = str(symbol or "").strip()
        self.symbol_info = normalize_symbol(self.query)
        self.code = self.symbol_info["code"]
        self.name = name
        # 用户可能直接输入中文名/拼音（手机上手输很常见）。此时
        # normalize_symbol 得不到代码，用联想接口解析一次（单次请求，很快）。
        if not self.code and self.query:
            if not self.name and not self.query.isascii():
                self.name = self.query          # 至少把'海天味业'显示出来
            try:
                hits = lookup_code_by_name(self.query)
            except Exception as exc:  # noqa: BLE001
                log.debug("名称解析失败: %s", exc)
                hits = []
            if hits:
                best = hits[0]
                self.symbol_info = normalize_symbol(best["prefix"] + best["code"])
                self.code = self.symbol_info["code"]
                if not self.name and best.get("name"):
                    self.name = best["name"]
                self.name_hits = hits
                log.info("按名称「%s」解析到 %s（%s）", self.query, self.code,
                         self.name or best.get("name", ""))
        if not self.code and self.query:
            self.symbol_info["display"] = self.query
            if not self.symbol_info.get("name"):
                self.symbol_info["name"] = self.query
        self.days = days or config.RUNTIME["history_days"]
        self.use_ml = use_ml
        self.demo = demo
        self.offline = offline or config.RUNTIME.get("offline", False)
        self.horizons = tuple(horizons)
        self.verbose = verbose
        self.sources: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # 日志
    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        if self.verbose:
            log.info(msg)

    def _pick(self, *mod_names, fn_name: str):
        """在多个候选模块里找第一个提供指定函数的模块。"""
        for mn in mod_names:
            m = _mod(mn)
            if m is not None and hasattr(m, fn_name):
                return getattr(m, fn_name), mn
        return None, ""

    # ------------------------------------------------------------------ #
    # 1. 取数
    # ------------------------------------------------------------------ #
    def fetch_data(self, bundle: AnalysisBundle) -> None:
        ds = _mod("datasource")
        daily = weekly = monthly = None
        if self.demo or ds is None:
            from .selftest_data import make_index, make_ohlcv

            daily = make_ohlcv(n=min(self.days, 900), seed=abs(hash(self.code)) % 9999,
                               start_price=30.0, symbol=self.code)
            bundle.note("演示模式：使用合成历史数据，仅用于验证流程与界面")
            self.sources["daily"] = "synthetic"
        else:
            self._log("拉取 %s 日线数据…" % self.display)
            daily = retry_call(lambda: ds.fetch_kline(self.code, period="daily",
                                                      days=self.days),
                               attempts=2, delay=0.8, default=None, label="fetch_daily")
            if daily is None or not len(daily):
                self._log("日线数据获取失败，尝试不带周期参数…")
                daily = retry_call(lambda: ds.fetch_daily(self.code, days=self.days),
                                   attempts=2, delay=0.8, default=None, label="fetch_daily2")
            if daily is None or not len(daily):
                from .selftest_data import make_ohlcv
                daily = make_ohlcv(n=min(self.days, 900), seed=7,
                                   start_price=30.0, symbol=self.code)
                bundle.note("⚠ 实时数据源不可用，已降级为合成数据演示（结果不可用于实盘决策）")
                self.sources["daily"] = "synthetic(fallback)"
            else:
                self.sources["daily"] = str(getattr(daily, "attrs", {}).get("source", "network"))
            for period, setter in (("weekly", "kline_weekly"), ("monthly", "kline_monthly")):
                try:
                    k = ds.fetch_kline(self.code, period=period, days=300)
                    if k is not None and len(k):
                        setattr(bundle, setter, k)
                except Exception as exc:  # noqa: BLE001
                    log.debug("%s 线获取失败: %s", period, exc)

        bundle.ohlcv = daily
        if self.name:
            bundle.name = self.name
        # 名称解析：优先用 K 线接口附带的名称（腾讯/新浪都会带），
        # 其次查代码表；两者都拿不到时先留空，等实时快照拿到后再定（见下）。
        if not bundle.name:
            attrs_name = (getattr(daily, "attrs", {}) or {}).get("name")
            if attrs_name:
                bundle.name = str(attrs_name).strip()
        if not bundle.name and ds is not None:
            try:
                info = ds.resolve_symbol(self.code) or {}
                nm = info.get("name")
                if nm and str(nm).strip() != self.code:
                    bundle.name = str(nm).strip()
            except Exception as exc:  # noqa: BLE001
                log.debug("resolve_symbol 失败: %s", exc)

        if bundle.kline_weekly is None and daily is not None and len(daily) > 30:
            try:
                w = _mod("indicators")
                if w and hasattr(w, "resample_kline"):
                    bundle.kline_weekly = w.resample_kline(daily, "W-FRI")
            except Exception:  # noqa: BLE001
                pass
        if daily is not None and len(daily):
            bundle.as_of = str(pd.to_datetime(daily["date"].iloc[-1]).date())

        # 实时快照（拿不到就用日线最后一行兜底）
        snap: dict = {}
        if ds is not None and not self.demo:
            snap = retry_call(lambda: ds.fetch_realtime(self.code), attempts=2,
                              delay=0.6, default={}, label="realtime") or {}
        if not snap and daily is not None and len(daily):
            last = daily.iloc[-1]
            snap = {"price": float(last["close"]), "source": "kline_fallback",
                    "prev_close": float(daily["close"].iloc[-2]) if len(daily) > 1 else None}
        # 股票名称按可靠度依次回退：用户传入 > 实时快照 > 日线接口附带 > 代码表 > 代码本身。
        # 注意：代码表查询（resolve_symbol）在未缓存时可能返回 None，
        # 绝不能让它覆盖已经拿到的真实名称。
        name_candidates = [
            self.name,
            snap.get("name"),
            (getattr(daily, "attrs", {}) or {}).get("name") if daily is not None else None,
            bundle.name,
        ]
        resolved = next((str(x).strip() for x in name_candidates
                         if x and str(x).strip() and str(x).strip() != self.code), "")
        bundle.name = resolved or bundle.name or self.code
        snap["name"] = bundle.name
        bundle.snapshot = {**snap, "symbol": self.code, "name": bundle.name}

        # 大盘指数
        idx = None
        if ds is not None and not self.demo:
            idx = retry_call(lambda: ds.fetch_index(config.RUNTIME["index_symbol"],
                                                    days=min(self.days, 1500)),
                             attempts=2, delay=0.6, default=None, label="index")
        if idx is None or not len(idx):
            from .selftest_data import make_index
            idx = make_index(n=min(self.days, 900), seed=7)
            if not self.demo:
                bundle.note("大盘指数数据获取失败，已用合成指数替代（相对强度类因子可信度已下调）")
        bundle.index_data = idx

        # 所属板块（同时用于后续"按行业筛选上下游商品"）
        if ds is not None and not self.demo and hasattr(ds, "fetch_sector_list"):
            boards = retry_call(lambda: ds.fetch_sector_list(self.code), attempts=2,
                                delay=0.6, default=[], label="sector") or []
            if boards:
                bundle.sentiment["boards_raw"] = boards
                self.sources["sector"] = "eastmoney"
        # 行业/概念名称（供商品、政策、题材等模块做相关性判断）
        board_names = [b.get("name") for b in (bundle.sentiment.get("boards_raw") or [])
                       if isinstance(b, dict) and b.get("name")]
        bundle.snapshot["industry_names"] = board_names
        if board_names:
            self._log("所属板块：%s" % "、".join(board_names[:8]))

    @property
    def display(self) -> str:
        return self.symbol_info.get("display") or self.code

    # ------------------------------------------------------------------ #
    # 2. 八大维度分析
    # ------------------------------------------------------------------ #
    def analyze(self, bundle: AnalysisBundle) -> None:
        df = bundle.ohlcv
        idx = bundle.index_data
        snap = bundle.snapshot or {}
        price = snap.get("price")

        steps = [
            ("个股特性与阶段", self._stage),
            ("技术面", self._technical),
            ("大盘与外围市场", self._global_market),
            ("基本面", self._fundamental),
            ("资金面与主力动向", self._capital),
            ("主力操盘意图", self._main_intent),
            ("市场情绪与博弈", self._sentiment),
            ("宏观与行业", self._macro),
            ("消息面与突发事件", self._news),
            ("周期与季节性", self._seasonality),
            ("金融理论模型", self._theory),
        ]
        for label, fn in steps:
            try:
                if self.verbose:
                    self._log("分析 %s…" % label)
                fn(bundle)
            except Exception as exc:  # noqa: BLE001
                bundle.error(label, exc)
                log.warning("%s 分析失败：%s", label, exc)
                log.debug("%s", traceback.format_exc(limit=3))
        # 收集各模块登记的数据源降级说明（last_notes()），让报告能如实反映数据完整性
        self._collect_notes(bundle)

    def _collect_notes(self, bundle: AnalysisBundle) -> None:
        """把各分析模块的降级说明汇总进 bundle.notes()。"""
        for modname in ("datasource", "fundamental", "capital", "sentiment",
                        "macro", "newsflow"):
            m = _mod(modname)
            fn = getattr(m, "last_notes", None) if m is not None else None
            if not callable(fn):
                continue
            try:
                notes = fn()
            except Exception as exc:  # noqa: BLE001
                log.debug("%s.last_notes() 失败: %s", modname, exc)
                continue
            if isinstance(notes, str):
                notes = [notes]
            for n in notes or []:
                if n:
                    bundle.note(str(n))

    # -- 个股特性/阶段 ------------------------------------------------- #
    def _stage(self, bundle: AnalysisBundle) -> None:
        """个股特性、所处阶段、定价权归属、量化活跃度。

        优先使用独立的 ``core.stage`` 模块（若存在），否则用 ``core.technical``
        里的 ``describe_stage`` / ``stage_factors``（内置实现）。
        """
        kw = dict(code=self.code, name=bundle.name, snapshot=bundle.snapshot)
        fn, mn = self._pick("stage", "technical", fn_name="stage_factors")
        if fn is not None:
            self._absorb(bundle, _call(fn, bundle.ohlcv, **kw), "stage", mn)
        fn2, _ = self._pick("stage", "technical", fn_name="describe_stage")
        if fn2 is not None:
            out2 = _call(fn2, bundle.ohlcv, **kw)
            if isinstance(out2, dict):
                bundle.stage.update(out2)
                factor = out2.get("factor")
                if factor is not None:
                    bundle.add(factor)

    # -- 技术面 --------------------------------------------------------- #
    def _technical(self, bundle: AnalysisBundle) -> None:
        fn, mn = self._pick("technical", fn_name="technical_factors")
        if fn is None:
            return
        out = _call(fn, bundle.ohlcv, index_df=bundle.index_data,
                    weekly=bundle.kline_weekly, monthly=bundle.kline_monthly,
                    snapshot=bundle.snapshot, code=self.code)
        self._absorb(bundle, out, "technical", mn)

    # -- 大盘与外围市场 ----------------------------------------------- #
    def _global_market(self, bundle: AnalysisBundle) -> None:
        """A股大盘、港股、外围股市的影响（同市场联动 / 港股同步 / 美股隔夜）。"""
        m = _mod("global_market")
        if m is None or bundle.ohlcv is None:
            return
        try:
            got = _call(m.analyze_market_influence, bundle.ohlcv,
                        days=min(self.days, 800), force=False) or {}
        except Exception as exc:  # noqa: BLE001
            bundle.error("大盘与外围", exc)
            return
        bundle.market = {
            "snapshot": got.get("snapshot") or {},
            "linkage": got.get("linkage") or {},
            "relative": got.get("relative") or {},
            "detail": got.get("detail") or "",
        }
        idx_frames = got.get("indices") or {}
        if idx_frames:
            bundle.market["index_history"] = {
                k: v.tail(120) for k, v in idx_frames.items()
                if hasattr(v, "tail")
            }
        self._absorb(bundle, got.get("factors") or [], "market", "global_market")
        if got.get("detail"):
            bundle.note(str(got["detail"]))

    # -- 主力操盘意图 --------------------------------------------------- #
    def _main_intent(self, bundle: AnalysisBundle) -> None:
        """主力当前操盘意图（建仓/拉升/洗盘/出货/打压吸筹…）。"""
        m = _mod("main_force_intent")
        if m is None or bundle.ohlcv is None:
            return
        flow = (bundle.capital or {}).get("flow")
        try:
            got = _call(m.analyze_main_force, bundle.ohlcv, flow,
                        frame=bundle.ohlcv, snapshot=bundle.snapshot) or {}
        except Exception as exc:  # noqa: BLE001
            bundle.error("主力意图", exc)
            return
        if not got:
            return
        bundle.capital["main_intent"] = got
        try:
            ser = _call(m.intent_series, bundle.ohlcv, flow, window=30) or []
            bundle.capital["intent_series"] = ser
        except Exception as exc:  # noqa: BLE001
            log.debug("意图序列计算失败: %s", exc)
        self._absorb(bundle, _call(m.intent_factors, bundle.ohlcv, flow,
                                   frame=bundle.ohlcv, snapshot=bundle.snapshot) or [],
                     "capital", "main_force_intent")
        self._log("主力意图：%s（意图分 %+.2f）"
                  % (got.get("intent_cn"), got.get("score", 0)))

    # -- 基本面 --------------------------------------------------------- #
    def _fundamental(self, bundle: AnalysisBundle) -> None:
        m = _mod("fundamental")
        if m is None:
            return
        frame = None
        if hasattr(m, "fetch_financials") and not self.demo:
            frame = retry_call(lambda: m.fetch_financials(self.code), attempts=2,
                               delay=0.6, default=None, label="financials")
            if frame is not None and len(frame):
                self.sources["financials"] = "network"
        bundle.financials = frame
        val = {}
        if hasattr(m, "fetch_valuation") and not self.demo:
            val = retry_call(lambda: m.fetch_valuation(self.code), attempts=2,
                             delay=0.6, default={}, label="valuation") or {}
        bundle.valuation = val
        if hasattr(m, "fundamental_factors"):
            out = _call(m.fundamental_factors, self.code, frame=frame, valuation=val,
                        price=bundle.snapshot.get("price"), name=bundle.name,
                        news=bundle.news)
            self._absorb(bundle, out, "fundamental", "fundamental")

    # -- 资金面 --------------------------------------------------------- #
    def _capital(self, bundle: AnalysisBundle) -> None:
        m = _mod("capital")
        if m is None:
            return
        flow = holders = margin = lhb = north = top10 = None
        if not self.demo:
            for attr, fname in (("flow", "fetch_fund_flow"), ("holders", "fetch_holders"),
                                ("margin", "fetch_margin"), ("lhb", "fetch_lhb"),
                                ("north", "fetch_northbound"),
                                ("top10", "fetch_top10_float_holders")):
                f = getattr(m, fname, None)
                if f is None:
                    continue
                val = retry_call(lambda f=f: f(self.code), attempts=2, delay=0.5,
                                 default=None, label=fname)
                if attr == "flow":
                    flow = val
                elif attr == "holders":
                    holders = val
                elif attr == "margin":
                    margin = val
                elif attr == "lhb":
                    lhb = val
                elif attr == "top10":
                    top10 = val
                else:
                    north = val
        bundle.capital = {"flow": flow, "holders": holders, "margin": margin,
                          "lhb": lhb, "north": north, "top10": top10,
                          "minute_flow": (retry_call(lambda: m.fetch_minute_flow(self.code),
                                                     attempts=1, default={}, label="minute")
                                          if hasattr(m, "fetch_minute_flow") and not self.demo else {})}
        if hasattr(m, "capital_factors"):
            out = _call(m.capital_factors, self.code, flow=flow, holders=holders,
                        margin=margin, lhb=lhb, north=north, top10=top10,
                        snapshot=bundle.snapshot, frame=bundle.ohlcv)
            self._absorb(bundle, out, "capital", "capital")
        if hasattr(m, "control_degree"):
            cd = _call(m.control_degree, self.code, frame=bundle.ohlcv, holders=holders)
            if isinstance(cd, dict):
                bundle.stage["control_degree"] = cd

    # -- 情绪 ----------------------------------------------------------- #
    def _sentiment(self, bundle: AnalysisBundle) -> None:
        m = _mod("sentiment")
        if m is None:
            return
        market = sector = retail = {}
        if not self.demo:
            for key, fname in (("market", "fetch_market_sentiment"),
                               ("sector", "fetch_sector_heat"),
                               ("retail", "fetch_retail_sentiment")):
                f = getattr(m, fname, None)
                if f is None:
                    continue
                val = retry_call(lambda f=f: f(self.code), attempts=2, delay=0.5,
                                 default={}, label=fname) or {}
                if key == "market":
                    market = val
                elif key == "sector":
                    sector = val
                else:
                    retail = val
        if not sector and bundle.sentiment.get("boards_raw"):
            sector = {"boards": bundle.sentiment["boards_raw"]}
        bundle.sentiment.update({"market": market, "sector": sector, "retail": retail})
        if hasattr(m, "sentiment_factors"):
            out = _call(m.sentiment_factors, self.code, market=market, sector=sector,
                        retail=retail, frame=bundle.ohlcv, snapshot=bundle.snapshot)
            self._absorb(bundle, out, "sentiment", "sentiment")
        if hasattr(m, "fear_greed_index"):
            fg = _call(m.fear_greed_index, self.code, frame=bundle.ohlcv, market=market)
            if isinstance(fg, dict):
                bundle.sentiment["fear_greed"] = fg

    # -- 宏观 ----------------------------------------------------------- #
    def _macro(self, bundle: AnalysisBundle) -> None:
        m = _mod("macro")
        if m is None:
            return
        macro = policy = commodity = {}
        if not self.demo:
            for key, fname in (("macro", "fetch_macro"), ("commodity", "fetch_commodity_prices")):
                f = getattr(m, fname, None)
                if f is None:
                    continue
                val = retry_call(lambda f=f: f(), attempts=2, delay=0.6, default={},
                                 label=fname) or {}
                if key == "macro":
                    macro = val
                else:
                    commodity = val
            f = getattr(m, "fetch_industry_policy", None)
            if f is not None:
                policy = retry_call(lambda: f(self.code, bundle.name), attempts=2,
                                    delay=0.5, default={}, label="policy") or {}
        # 上下游商品：**按个股所属行业筛选**，只保留与该公司真正相关的品种
        industry_names = (bundle.snapshot or {}).get("industry_names") or []
        commodity_rel: dict = {}
        try:
            from . import commodity_industry as CI

            commodity_rel = CI.relevant_commodities(commodity, industry_names)
            if commodity_rel.get("items"):
                commodity = {**commodity, **commodity_rel["items"]}
        except Exception as exc:  # noqa: BLE001
            bundle.error("上下游商品筛选", exc)
        bundle.macro.update({"macro": macro, "policy": policy, "commodity": commodity,
                             "commodity_relevant": commodity_rel,
                             "industry_names": industry_names})
        if hasattr(m, "market_regime"):
            reg = _call(m.market_regime, bundle.index_data)
            if isinstance(reg, dict):
                bundle.macro["market_regime"] = reg
        if hasattr(m, "macro_factors"):
            out = _call(m.macro_factors, self.code, macro=macro, policy=policy,
                        commodity=commodity, index_data=bundle.index_data,
                        frame=bundle.ohlcv)
            self._absorb(bundle, out, "macro", "macro")
        # 补一条"按行业筛选后的商品价格"因子（比通用商品因子更贴合该股）
        try:
            from . import commodity_industry as CI

            self._absorb(bundle, CI.commodity_factors(commodity_rel.get("items") or {},
                                                     industry_names), "macro", "commodity")
        except Exception as exc:  # noqa: BLE001
            log.debug("商品因子生成失败: %s", exc)

    # -- 消息面 --------------------------------------------------------- #
    def _news(self, bundle: AnalysisBundle) -> None:
        m = _mod("newsflow")
        news, anns = [], []
        # ① 个股专属新闻（按代码过滤，相关性最高）——放在最优先
        if not self.demo:
            news = retry_call(lambda: fetch_stock_news(self.symbol_info, bundle.name, 20),
                              attempts=2, delay=0.5, default=[], label="stock_news") or []
            if news and m is not None and hasattr(m, "classify_news"):
                try:
                    news = m.classify_news(news)
                except Exception as exc:  # noqa: BLE001
                    log.debug("个股新闻分类失败: %s", exc)
        if m is None:
            bundle.news = list(news) + list(anns)
            return
        # ② 公告（按代码查询，官方来源，同样高度相关）
        if not self.demo:
            f = getattr(m, "fetch_announcements", None)
            if f is not None:
                anns = retry_call(lambda: f(self.code), attempts=2, delay=0.5,
                                  default=[], label="anns") or []
        # ③ 补充搜索类新闻：只有在"标题与个股相关"时才采纳
        extra = []
        if not self.demo and len(news) < 5:
            f = getattr(m, "fetch_news", None)
            if f is not None:
                cand = retry_call(lambda: f(self.code, bundle.name), attempts=1,
                                  delay=0.4, default=[], label="news") or []
                extra = [x for x in cand if _is_relevant(x, self.code, bundle.name)]
                if len(cand) and not extra:
                    bundle.note("已过滤掉 %d 条与「%s」无直接关系的搜索类资讯"
                                % (len(cand), bundle.name or self.code))
                if extra and hasattr(m, "classify_news"):
                    try:
                        extra = m.classify_news(extra)
                    except Exception:  # noqa: BLE001
                        pass
        bundle.news = list(news) + list(anns) + list(extra)
        if hasattr(m, "event_factors"):
            out = _call(m.event_factors, self.code, news=bundle.news, announcements=anns,
                        name=bundle.name)
            self._absorb(bundle, out, "news", "newsflow")

    # -- 季节性 --------------------------------------------------------- #
    def _seasonality(self, bundle: AnalysisBundle) -> None:
        m = _mod("seasonality")
        if m is None:
            return
        if hasattr(m, "seasonality_factors"):
            out = _call(m.seasonality_factors, self.code, bundle.ohlcv,
                        frame=bundle.financials)
            self._absorb(bundle, out, "seasonality", "seasonality")
        if hasattr(m, "seasonality_report"):
            rep = _call(m.seasonality_report, self.code, bundle.ohlcv)
            if isinstance(rep, dict):
                bundle.seasonality.update(rep)

    # -- 理论模型 ------------------------------------------------------- #
    def _theory(self, bundle: AnalysisBundle) -> None:
        m = _mod("theory")
        if m is None:
            return
        if hasattr(m, "theory_factors"):
            out = _call(m.theory_factors, bundle.ohlcv, weekly=bundle.kline_weekly,
                        monthly=bundle.kline_monthly)
            self._absorb(bundle, out, "theory", "theory")
        if hasattr(m, "theory_summary"):
            rep = _call(m.theory_summary, bundle.ohlcv, weekly=bundle.kline_weekly,
                        monthly=bundle.kline_monthly)
            if isinstance(rep, dict):
                bundle.theory.update(rep)

    # ------------------------------------------------------------------ #
    # 工具：吸收模块返回值
    # ------------------------------------------------------------------ #
    def _absorb(self, bundle: AnalysisBundle, out, category: str, modname: str) -> None:
        """模块可能返回 list[Factor]、单个 Factor、或 (factors, rich_dict)。

        本方法统一处理，并把富数据塞进 bundle 的对应字段。
        """
        if out is None:
            return
        rich = None
        if isinstance(out, tuple) and len(out) == 2:
            out, rich = out
        if isinstance(out, dict):
            rich = out
            out = out.get("factors") or []
        if isinstance(out, list):
            added = 0
            for f in out:
                if f is None:
                    continue
                if hasattr(f, "direction") and hasattr(f, "strength"):
                    bundle.factors.append(f)
                    added += 1
            if added:
                self.sources[category] = "%s×%d" % (modname, added)
        if isinstance(rich, dict):
            target = {
                "stage": bundle.stage, "valuation": bundle.valuation,
                "capital": bundle.capital, "sentiment": bundle.sentiment,
                "macro": bundle.macro, "seasonality": bundle.seasonality,
                "theory": bundle.theory, "indicators": bundle.indicators,
            }.get(category)
            if target is not None:
                for k, v in rich.items():
                    if k not in ("factors",):
                        target.setdefault(k, v)

    # ------------------------------------------------------------------ #
    # 3. 主流程
    # ------------------------------------------------------------------ #
    def run(self, progress=None) -> PredictionResult:
        """执行完整流程。

        Parameters
        ----------
        progress : callable | None
            形如 ``progress(stage_name, pct)`` 的回调，供 Streamlit 显示进度。
        """
        if not self.code:
            raise ValueError(
                "无法识别股票代码「%s」。请输入 6 位 A 股代码（如 600519、000001、300750）"
                "或带市场前缀的写法（sh600519、000001.SZ、00700.HK、AAPL）。"
                % (self.query or "（空）"))

        def _p(stage: str, pct: float):
            if progress:
                try:
                    progress(stage, pct)
                except Exception:  # noqa: BLE001
                    pass

        bundle = AnalysisBundle(symbol=self.code, name=self.name, as_of="",
                                factors=FactorSet())
        _p("获取行情数据", 0.05)
        self.fetch_data(bundle)
        _p("八大维度分析", 0.20)
        self.analyze(bundle)
        _p("因子汇总与概率推导", 0.72)
        result = build_prediction(bundle, use_ml=self.use_ml, horizons=self.horizons,
                                  progress=lambda m, p: _p(m, 0.72 + 0.25 * p))
        result.meta["sources"] = dict(self.sources)
        result.meta["demo"] = bool(self.demo or self.sources.get("daily", "").startswith("synthetic"))
        result.meta["errors"] = list(bundle.errors)
        result.meta["notes"] = list(bundle.notes)
        _p("生成报告", 0.98)
        return result


def build_prediction(bundle: AnalysisBundle, *, use_ml: bool = True,
                     horizons=HORIZONS, progress=None) -> PredictionResult:
    """对已填充的 bundle 生成预测（供外部直接复用）。"""
    from . import mlmodel
    from .forecast import build_forecasts

    ml_models = None
    if use_ml and config.FORECAST.get("ml_enabled", True):
        df = bundle.ohlcv
        if df is not None and len(df) >= config.RUNTIME["min_history_for_ml"]:
            try:
                if progress:
                    progress("训练AI模型", 0.05)
                models = train_all(df, bundle, horizons)
                ml_models = models
            except Exception as exc:  # noqa: BLE001
                bundle.error("AI模型训练", exc)
                log.warning("AI 模型训练失败，使用纯规则模型：%s", exc)
    if progress:
        progress("推导概率", 0.6)
    result = build_forecasts(bundle, ml_models=ml_models, use_ml=use_ml, horizons=horizons)
    try:
        from .forecast import ensemble_view

        result.ensemble = ensemble_view(result)
    except Exception as exc:  # noqa: BLE001
        bundle.error("综合评分", exc)
    return result


def train_all(df: pd.DataFrame, bundle: AnalysisBundle, horizons=HORIZONS) -> dict:
    """训练 AI 模型：技术特征 + 逐根 K 线的规则分特征（在 :mod:`core.forecast` 中实现）。"""
    from .forecast import train_models_with_rule_feature

    return train_models_with_rule_feature(df, bundle, horizons)


def quick_predict(symbol: str, *, use_ml: bool = False, demo: bool = False,
                  days: int | None = None) -> PredictionResult:
    """一行代码出结果（默认关闭 AI 训练以加快速度）。"""
    return Predictor(symbol, use_ml=use_ml, demo=demo, days=days).run()


if __name__ == "__main__":  # pragma: no cover
    import time

    t0 = time.time()
    r = quick_predict("600519", use_ml=False, demo=True)
    print(r.headline())
    print()
    print(r.to_frame().to_string(index=False))
    print()
    print("综合评分:", r.ensemble.get("score_pct"), r.ensemble.get("stance"))
    print("数据源:", r.meta.get("sources"))
    if r.meta.get("errors"):
        print("模块异常:", r.meta["errors"])
    print("备注:", r.meta.get("notes"))
    print("耗时 %.1fs" % (time.time() - t0))
    print("engine self-test OK")

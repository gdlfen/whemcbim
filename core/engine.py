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
from .utils import get_logger, normalize_symbol, retry_call

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
        self.query = symbol
        self.symbol_info = normalize_symbol(symbol)
        self.code = self.symbol_info["code"]
        self.name = name
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
        # 缓存交互时，代码名可能来自数据源
        try:
            if ds is not None and not self.name:
                info = ds.resolve_symbol(self.code)
                bundle.name = info.get("name") or bundle.name or self.code
        except Exception:  # noqa: BLE001
            bundle.name = bundle.name or self.code

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

        # 所属板块
        if ds is not None and not self.demo and hasattr(ds, "fetch_sector_list"):
            boards = retry_call(lambda: ds.fetch_sector_list(self.code), attempts=2,
                                delay=0.6, default=[], label="sector") or []
            if boards:
                bundle.sentiment["boards_raw"] = boards
                self.sources["sector"] = "eastmoney"

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
            ("基本面", self._fundamental),
            ("资金面与主力动向", self._capital),
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
        flow = holders = margin = lhb = north = None
        if not self.demo:
            for attr, fname in (("flow", "fetch_fund_flow"), ("holders", "fetch_holders"),
                                ("margin", "fetch_margin"), ("lhb", "fetch_lhb"),
                                ("north", "fetch_northbound")):
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
                else:
                    north = val
        bundle.capital = {"flow": flow, "holders": holders, "margin": margin,
                          "lhb": lhb, "north": north,
                          "minute_flow": (retry_call(lambda: m.fetch_minute_flow(self.code),
                                                     attempts=1, default={}, label="minute")
                                          if hasattr(m, "fetch_minute_flow") and not self.demo else {})}
        if hasattr(m, "capital_factors"):
            out = _call(m.capital_factors, self.code, flow=flow, holders=holders,
                        margin=margin, lhb=lhb, north=north,
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
        bundle.macro.update({"macro": macro, "policy": policy, "commodity": commodity})
        if hasattr(m, "market_regime"):
            reg = _call(m.market_regime, bundle.index_data)
            if isinstance(reg, dict):
                bundle.macro["market_regime"] = reg
        if hasattr(m, "macro_factors"):
            out = _call(m.macro_factors, self.code, macro=macro, policy=policy,
                        commodity=commodity, index_data=bundle.index_data,
                        frame=bundle.ohlcv)
            self._absorb(bundle, out, "macro", "macro")

    # -- 消息面 --------------------------------------------------------- #
    def _news(self, bundle: AnalysisBundle) -> None:
        m = _mod("newsflow")
        if m is None:
            return
        news, anns = [], []
        if not self.demo:
            f = getattr(m, "fetch_news", None)
            if f is not None:
                news = retry_call(lambda: f(self.code, bundle.name), attempts=2,
                                  delay=0.5, default=[], label="news") or []
            f = getattr(m, "fetch_announcements", None)
            if f is not None:
                anns = retry_call(lambda: f(self.code), attempts=2, delay=0.5,
                                  default=[], label="anns") or []
        if news and hasattr(m, "classify_news"):
            try:
                news = m.classify_news(news)
            except Exception as exc:  # noqa: BLE001
                log.debug("新闻分类失败: %s", exc)
        bundle.news = list(news) + list(anns)
        if hasattr(m, "event_factors"):
            out = _call(m.event_factors, self.code, news=news, announcements=anns,
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

# -*- coding: utf-8 -*-
"""
一键自检脚本 —— 验证项目完整性、各模块可用性、以及端到端流程能否跑通。

用法
----
::

    python verify_all.py            # 离线自检（不联网，用合成数据）
    python verify_all.py --online   # 额外做真实数据端到端测试（需要网络）

退出码 0 表示全部通过；非 0 表示有项目失败（输出里会指出）。
适合放进 CI，或在部署前手动跑一遍确认环境没问题。
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def check(name: str, fn, *, critical: bool = True):
    """执行一项检查并记录结果。"""
    t0 = time.time()
    try:
        msg = fn()
        results.append((name, PASS, str(msg or "")[:150]))
        print("  [%s] %-42s %5.1fs  %s" % (PASS, name, time.time() - t0, str(msg or "")[:80]))
        return True
    except Exception as exc:  # noqa: BLE001
        detail = "%s: %s" % (type(exc).__name__, exc)
        results.append((name, FAIL if critical else SKIP, detail))
        print("  [%s] %-42s %5.1fs  %s" % (FAIL if critical else SKIP, name,
                                           time.time() - t0, detail[:110]))
        if os.environ.get("VERIFY_TRACE"):
            traceback.print_exc()
        return False


# --------------------------------------------------------------------------- #
# 1. 模块可用性
# --------------------------------------------------------------------------- #
CORE_MODULES = [
    "contracts", "config", "utils", "selftest_data",
    "datasource", "indicators", "patterns", "technical",
    "fundamental", "capital", "sentiment", "macro", "newsflow",
    "seasonality", "theory", "mlmodel", "forecast", "backtest",
    "charts", "report", "engine",
    # 新增维度模块
    "global_market", "main_force_intent", "commodity_industry",
]
# 允许缺失的模块（缺失时引擎会自动跳过该维度，只是功能减少）
OPTIONAL_MODULES = {"datasource", "technical", "fundamental", "capital",
                    "sentiment", "macro", "newsflow",
                    "global_market", "main_force_intent", "commodity_industry"}


def check_files() -> str:
    required = ["app.py", "run_local.py", "requirements.txt", "core/__init__.py",
                "core/contracts.py", "core/engine.py", "core/forecast.py"]
    missing = [f for f in required if not os.path.exists(os.path.join(_HERE, f))]
    if missing:
        raise FileNotFoundError("缺少文件：%s" % ", ".join(missing))
    n_py = sum(1 for r, _, fs in os.walk(_HERE) for f in fs
               if f.endswith(".py") and "__pycache__" not in r)
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(os.path.join(_HERE, "core")) for f in fs
                if f.endswith(".py"))
    return "必需文件齐全；Python 文件 %d 个；core 代码 %.0f KB" % (n_py, total / 1024)


def check_imports() -> str:
    ok, missing, failed = [], [], []
    for m in CORE_MODULES:
        try:
            importlib.import_module("core." + m)
            ok.append(m)
        except ModuleNotFoundError as exc:
            missing.append("%s(%s)" % (m, exc.name))
        except Exception as exc:  # noqa: BLE001
            failed.append("%s(%s)" % (m, type(exc).__name__))
    if failed:
        raise RuntimeError("导入报错：%s" % ", ".join(failed))
    hard_missing = [m for m in missing if m.split("(")[0] not in OPTIONAL_MODULES]
    if hard_missing:
        raise RuntimeError("缺少核心模块：%s" % ", ".join(hard_missing))
    return "可用 %d 个%s" % (len(ok), ("；可选模块缺失：%s" % ", ".join(missing)) if missing else "")


def check_deps() -> str:
    info = []
    for pkg in ("numpy", "pandas", "requests", "matplotlib", "streamlit", "akshare",
                "yfinance"):
        try:
            mod = importlib.import_module(pkg)
            info.append("%s=%s" % (pkg, getattr(mod, "__version__", "?")))
        except Exception:  # noqa: BLE001
            info.append("%s=缺失" % pkg)
    return " ".join(info)


# --------------------------------------------------------------------------- #
# 2. 数值内核
# --------------------------------------------------------------------------- #
def check_causality() -> str:
    """特征与规则分必须严格因果（这是最容易出错、后果最严重的地方）。"""
    import numpy as np

    from core.forecast import fast_rule_score_history
    from core.mlmodel import build_features
    from core.selftest_data import make_index, make_ohlcv

    df = make_ohlcv(1000, seed=42)
    idx = make_index(1000, seed=7)
    x_full = build_features(df, idx)
    x_trunc = build_features(df.iloc[:700], idx.iloc[:700])
    a = x_full.iloc[:700].fillna(-999).to_numpy(dtype=float)
    b = x_trunc.fillna(-999).to_numpy(dtype=float)
    bad = [c for j, c in enumerate(x_full.columns)
           if not np.allclose(a[:, j], b[:, j], atol=1e-9)]
    if bad:
        raise AssertionError("特征存在未来函数：%s" % bad)
    for key in ("d1", "d20", "y1"):
        s1 = fast_rule_score_history(df, key)
        s2 = fast_rule_score_history(df.iloc[:700], key)
        if not np.allclose(s1[:700], s2, atol=1e-9):
            raise AssertionError("规则分(%s)存在未来函数" % key)
    return "%d 个特征 + 3 个周期的规则分均通过因果性检查" % x_full.shape[1]


def check_no_inf() -> str:
    import numpy as np

    from core.mlmodel import build_features
    from core.selftest_data import make_index, make_ohlcv

    df = make_ohlcv(800, seed=11)
    x = build_features(df, make_index(800, seed=3))
    arr = x.to_numpy(dtype=float)
    if np.isinf(arr).any():
        raise AssertionError("特征含 ±inf")
    ratio = float(np.isfinite(arr).mean())
    if ratio < 0.85:
        raise AssertionError("有效特征比例过低 %.3f" % ratio)
    return "无 ±inf，有效值占比 %.1f%%" % (ratio * 100)


def check_probability_consistency() -> str:
    from core.contracts import AnalysisBundle, Category, Factor
    from core.forecast import build_forecasts
    from core.selftest_data import make_index, make_ohlcv

    df, idx = make_ohlcv(800, seed=5), make_index(800, seed=5)
    b = AnalysisBundle(symbol="TEST", name="自检", as_of="2025-01-01", ohlcv=df,
                       index_data=idx)
    for i, (cat, d, s) in enumerate([(Category.TECH, 1, 0.8), (Category.CAPITAL, 1, 0.7),
                                     (Category.FUND, -1, 0.5), (Category.SENTIMENT, -1, 0.4),
                                     (Category.THEORY, 1, 0.6), (Category.MACRO, 1, 0.4)]):
        b.add(Factor("chk_%d" % i, "自检因子%d" % i, cat, d, s, 0.8))
    r = build_forecasts(b, use_ml=False)
    assert len(r.forecasts) == 8, "应输出 8 个周期"
    sigs = [f.sigma for f in r.forecasts]
    assert sigs == sorted(sigs), "σ 必须随周期单调递增：%s" % sigs
    for f in r.forecasts:
        assert 0.02 <= f.p_up <= 0.98, "概率越界"
        assert abs(f.p_up + f.p_down - 1) < 1e-9
        assert f.ci_low < f.exp_return < f.ci_high, "区间未包住期望值"
        assert f.p_big_up + f.p_big_down <= 1 + 1e-9
        assert 0 <= f.confidence <= 1
    # 短线资金/技术看多、长线基本面看空 → 短线分数应高于长线
    assert r.by_key("d1").strength_score > r.by_key("y1").strength_score, \
        "权重矩阵未体现周期差异"
    return "8 周期概率/区间/σ 全部自洽（σ %.2f%%→%.2f%%）" % (sigs[0], sigs[-1])


def check_models() -> str:
    import numpy as np

    from core import mlmodel
    from core.selftest_data import make_ohlcv

    df = make_ohlcv(900, seed=21)
    models = mlmodel.train_models(df, {"index_df": None}, horizons=mlmodel.HORIZONS[:4]
                                  if hasattr(mlmodel, "HORIZONS") else None)
    assert models, "训练未返回任何模型"
    usable = [k for k, m in models.items() if m.usable]
    assert usable, "没有可用的模型"
    preds = mlmodel.predict(models, df, {})
    assert preds, "预测未返回结果"
    for k, p in preds.items():
        assert 0 <= p.p_up <= 1, "概率越界 %s" % k
    fs = mlmodel.ml_factors(preds)
    for f in fs:
        assert 0 <= f.strength <= 1 and 0 <= f.reliability <= 1
    return "训练 %d 个周期（可用 %d），预测 %d 个，ML因子 %d 条" % (
        len(models), len(usable), len(preds), len(fs))


def check_backtest() -> str:
    import numpy as np

    from core.backtest import backtest_factor_score, backtest_prediction_history
    from core.selftest_data import make_ohlcv

    df = make_ohlcv(1300, seed=42)
    bt = backtest_prediction_history(df)
    assert len(bt) == 8, "应回测 8 个周期：%d" % len(bt)
    # 随机信号不应表现出有效性（防止回测实现有系统性偏差）
    rng = np.random.default_rng(1)
    rand = rng.normal(0, 0.3, len(df))
    ic = backtest_factor_score(df, rand, horizons=["d20"])["d20"].ic
    assert abs(ic) < 0.15, "随机信号出现异常 IC=%.3f，回测实现可疑" % ic
    n_valid = sum(1 for v in bt.values() if v.n_samples >= 30)
    return "8 周期回测完成（有效样本周期 %d 个）；随机信号对照 IC=%.3f" % (n_valid, ic)


# --------------------------------------------------------------------------- #
# 3. 端到端
# --------------------------------------------------------------------------- #
def check_engine_demo() -> str:
    from core.engine import quick_predict

    r = quick_predict("600519", use_ml=False, demo=True)
    assert len(r.forecasts) == 8
    from core.report import build_html, build_text

    txt = build_text(r)
    html = build_html(r)
    assert len(txt) > 1500 and len(html) > 5000, "报告过短"
    for kw in ("八大周期", "八维", "理论", "免责"):
        assert kw in html, "报告缺少区块：%s" % kw
    return "端到端跑通；文本 %d 字 / HTML %d 字" % (len(txt), len(html))


def check_charts() -> str:
    from core import charts as ch
    from core.contracts import ProbabilityForecast, PredictionResult
    from core.selftest_data import make_ohlcv

    df = make_ohlcv(300, seed=5)
    res = PredictionResult(symbol="T", name="T", price=30.0, forecasts=[
        ProbabilityForecast(horizon=h, label=l, p_up=p, exp_return=e, ci_low=e - 3,
                            ci_high=e + 3, sigma=3.0)
        for h, l, p, e in [("d1", "1天", 0.55, 0.2), ("d20", "20天", 0.6, 1.5),
                           ("y1", "1年", 0.52, 5.0)]])
    figs = [ch.plot_price(df, bars=100), ch.plot_indicators(df, bars=100),
            ch.plot_multi_horizon(res), ch.plot_confidence_cone(res),
            ch.plot_probability_gauge(res)]
    ok = sum(1 for f in figs if f is not None)
    assert ok >= 4, "图表生成失败过多（%d/5）" % ok
    for f in figs:
        if f is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(f)
            except Exception:  # noqa: BLE001
                pass
    return "5 类图表生成 %d 张；中文字体=%s" % (ok, ch.setup_chinese_font())


def check_app_syntax() -> str:
    """检查 Streamlit 应用能否被编译（不启动服务器）。"""
    import py_compile

    p = os.path.join(_HERE, "app.py")
    py_compile.compile(p, doraise=True)
    src = open(p, encoding="utf-8").read()
    for kw in ("st.set_page_config", "cached_analysis", "render_forecast_table",
               "render_dimensions", "DISCLAIMER"):
        assert kw in src, "app.py 缺少关键部分：%s" % kw
    return "app.py 编译通过，关键组件齐全（%d 行）" % len(src.splitlines())


def check_market_and_intent() -> str:
    """新增维度：大盘/外围联动、主力操盘意图（离线合成数据）。"""
    from core import global_market as G
    from core import main_force_intent as MFI
    from core.selftest_data import make_ohlcv

    df = make_ohlcv(300, seed=13)
    # 主力意图：构造"建仓"场景，应能识别
    import numpy as np
    import pandas as pd
    price = np.array(list(np.linspace(30, 20, 200)) + list(np.linspace(20, 20.4, 100)))
    d = make_ohlcv(300, seed=13, start_price=20.0)
    d["close"] = price
    d["high"] = price * 1.01
    d["low"] = price * 0.99
    flow = pd.DataFrame({"date": d["date"],
                         "main_net": [0.0] * 200 + [3e6, 4e6, 2e6, 5e6, 3e6] * 20})
    r = MFI.analyze_main_force(d, flow)
    assert r["intent"] != "watch", "建仓场景应识别出非观望意图：%s" % r["intent"]
    fs = MFI.intent_factors(d, flow)
    assert fs, "主力意图应生成因子"
    # 大盘/外围：离线用构造快照验证因子生成
    snap = {"sh": {"name": "上证指数", "pct_chg": 1.0},
            "hs300": {"name": "沪深300", "pct_chg": 1.1},
            "hsi": {"name": "恒生指数", "pct_chg": 0.9},
            "ixic": {"name": "纳斯达克", "pct_chg": -1.2},
            "vix": {"name": "VIX", "price": 26.0}}
    gfs = G.global_market_factors(df, snap, {}, {"hs300": {"same_day_corr": 0.5,
                                                           "lead_corr": 0.1, "n": 200}})
    assert len(gfs) >= 3, "大盘/外围应至少生成 3 条因子"
    ids = {f.id for f in gfs}
    assert "market_a_share" in ids and "market_us_overnight" in ids
    return "大盘/外围因子 %d 条；主力意图=%s（意图分 %+.2f）" % (
        len(gfs), r["intent_cn"], r["score"])


def check_online(symbol: str = "600519") -> str:
    """真实数据端到端测试。"""
    from core import datasource as ds
    from core.engine import quick_predict

    df = ds.fetch_kline(symbol, period="daily", days=1600)
    if df is None or df.empty:
        raise RuntimeError("未能取到日线数据（网络或数据源不可用）")
    r = quick_predict(symbol, use_ml=False)
    src = (r.meta.get("sources") or {}).get("daily")
    factors = r.meta.get("n_factors", 0)
    if str(src).startswith("synthetic"):
        raise AssertionError("实际用了合成数据（真实数据源不可用）")
    if factors < 15:
        raise AssertionError("因子数偏少（%d），可能多个数据源不可用" % factors)
    return "真实数据 %d 行（源=%s），因子 %d 条，20日涨概率 %.1f%%" % (
        len(df), src, factors, (r.by_key("d20") or r.forecasts[0]).p_up * 100)


def check_health(symbol: str = "600519") -> str:
    from core import datasource as ds

    rep = ds.data_health_check(symbol)
    overall = rep.get("overall")
    items = rep.get("items") or {}
    ok = sum(1 for v in items.values() if isinstance(v, dict) and v.get("ok"))
    return "整体=%s，%d/%d 个数据源可用" % (overall, ok, len(items))


def check_realtime(symbol: str = "600519") -> str:
    from core import datasource as ds

    rt = ds.fetch_realtime(symbol)
    if not rt or rt.get("price") in (None, 0):
        raise RuntimeError("未取到实时快照")
    return "现价 %s（源=%s）" % (rt.get("price"), rt.get("source"))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="项目自检")
    ap.add_argument("--online", action="store_true", help="额外做真实数据端到端测试")
    ap.add_argument("--symbol", default="600519", help="在线测试用的股票代码")
    ap.add_argument("--skip-heavy", action="store_true", help="跳过耗时较长的检查")
    args = ap.parse_args()

    print("=" * 88)
    print("个股涨跌概率预测程序 —— 自检")
    print("工作目录：%s" % _HERE)
    print("=" * 88)

    print("\n【1】项目结构与依赖")
    check("项目文件完整性", check_files)
    check("核心模块导入", check_imports)
    check("第三方库版本", check_deps, critical=False)
    check("Streamlit 应用可编译", check_app_syntax)

    print("\n【2】数值内核正确性")
    check("特征/规则分因果性（无未来函数）", check_causality)
    check("特征数值健康（无 ±inf）", check_no_inf)
    check("概率/区间/波动率自洽性", check_probability_consistency)
    check("大盘/外围联动 + 主力意图", check_market_and_intent)
    if not args.skip_heavy:
        check("AI 模型训练与预测", check_models)
        check("历史回测与随机对照", check_backtest)
    else:
        print("  [SKIP] AI 模型训练与预测（--skip-heavy）")
        print("  [SKIP] 历史回测与随机对照（--skip-heavy）")

    print("\n【3】端到端流程（离线合成数据）")
    check("引擎全流程 + 报告生成", check_engine_demo)
    check("图表生成", check_charts)

    if args.online:
        print("\n【4】真实数据端到端（联网）")
        check("数据源健康检查", lambda: check_health(args.symbol), critical=False)
        check("实时快照", lambda: check_realtime(args.symbol), critical=False)
        check("真实数据完整分析", lambda: check_online(args.symbol), critical=False)

    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print("\n" + "=" * 88)
    print("结果：通过 %d ｜ 失败 %d ｜ 跳过 %d" % (n_pass, n_fail, n_skip))
    if n_fail:
        print("\n失败项：")
        for name, st, msg in results:
            if st == FAIL:
                print("  ✗ %s\n      %s" % (name, msg))
        print("\n提示：设置环境变量 VERIFY_TRACE=1 可打印完整异常栈。")
    else:
        print("✓ 全部通过，可以运行：")
        print("    本地网页版： streamlit run app.py")
        print("    本地命令行： python run_local.py 600519")
    print("=" * 88)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

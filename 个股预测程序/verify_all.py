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
    "global_market", "main_force_intent", "commodity_industry", "fund",
    "market_structure", "market_board", "progress", "v8_guard", "keypoint", "market_forecast", "market_fundamental", "market_phase", "market_context", "freshness", "macro_source", "retail_heat", "ai_analyst",
]
# 允许缺失的模块（缺失时引擎会自动跳过该维度，只是功能减少）
OPTIONAL_MODULES = {"datasource", "technical", "fundamental", "capital",
                    "sentiment", "macro", "newsflow",
                    "global_market", "main_force_intent", "commodity_industry", "fund",
                    "market_structure", "market_board", "progress", "v8_guard", "keypoint", "market_forecast", "market_fundamental"}


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


def check_self_learning() -> str:
    """自我学习闭环：历史回填 → 核账 → 校准 → 参数可被预测引擎读取。"""
    import os

    import numpy as np
    import pandas as pd

    # 用独立目录，避免污染真实档案（放在工作区内，兼容受限环境）
    _root = os.path.dirname(os.path.abspath(__file__))
    os.environ["STOCKAI_DATA_DIR"] = os.path.join(_root, ".verify_learn")
    from core import learning as L

    L.reset("all")
    rng = np.random.default_rng(3)
    n = 420
    dates = pd.bdate_range("2024-06-03", periods=n)
    rets = rng.normal(0.0008, 0.02, n)
    closes = 20 * np.cumprod(1 + rets)
    daily = pd.DataFrame({"date": dates, "open": closes * 0.995, "high": closes * 1.01,
                          "low": closes * 0.99, "close": closes,
                          "volume": rng.integers(1e6, 5e6, n).astype(float)})
    bf = L.backfill_history("VERIFY", daily, step=8, max_points=30)
    assert bf.get("added", 0) > 0, "历史回填应产生样本：%s" % bf
    lr = L.learn("VERIFY")
    assert lr.get("samples", 0) > 0, "学习应有样本"
    adj = L.get_adjustments("VERIFY", "d20")
    assert adj and "prob_adjust_logit" in adj and "confidence_mult" in adj, \
        "应能取到 d20 的校准参数：%s" % adj
    mult = float(adj["confidence_mult"])
    assert 0.4 <= mult <= 1.4, "置信度乘子应在合理范围：%s" % mult
    rep = L.calibration_report("VERIFY")
    assert rep["evaluated"] > 0 and rep["table"], "学习报告应有内容"
    L.reset("all")
    return ("回填 %d 条 → 核账 %d 条 → 校准生效（d20 置信度乘子 %.3f）"
            % (bf.get("added", 0), rep["evaluated"], mult))


def check_font_coverage() -> str:
    """内置字体是否覆盖源码里所有中文字符（防云端图表出现方块）。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_font_coverage.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_font_coverage.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-6:])
        raise AssertionError("字体覆盖不足（云端图表会显示方块）：\n%s" % tail)
    line = [x for x in out.splitlines() if "字形数" in x]
    return line[0].strip() if line else "字体覆盖检查通过"


def check_cache_signature() -> str:
    """@st.cache_data 函数的参数必须可哈希（回调类参数须带下划线前缀）。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_cache_signature.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_cache_signature.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=60)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-6:])
        raise AssertionError("缓存函数签名不合规（真机会报 UnhashableParamError）：\n%s"
                             % tail)
    line = [x for x in out.splitlines() if "检查了" in x]
    return line[0].strip() if line else "缓存签名检查通过"


def check_labels() -> str:
    """界面字段标签不得残留英文（金融缩写白名单放行）。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_labels.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_labels.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=180)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-8:])
        raise AssertionError("字段标签仍残留英文：\n%s" % tail)
    line = [x for x in out.splitlines() if "字段标签总数" in x]
    return line[0].strip() if line else "字段标签检查通过"


def check_factor_display() -> str:
    """因子展示值不得出现 None/nan/inf 字样。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_factor_display.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_factor_display.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=180)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-8:])
        raise AssertionError("因子展示值含 None/nan：\n%s" % tail)
    line = [x for x in out.splitlines() if "检查因子" in x]
    return line[0].strip() if line else "因子展示值检查通过"


def check_render_rules() -> str:
    """渲染层收敛：不得再出现手写渲染写法（英文/None/原始字典的根源）。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_render_rules.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_render_rules.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\\n".join(out.strip().splitlines()[-10:])
        raise AssertionError("仍存在手写渲染写法：\\n%s" % tail)
    line = [x for x in out.splitlines() if "扫描 app.py" in x]
    return line[0].strip() if line else "渲染规则检查通过"


def check_render_output() -> str:
    """端到端渲染校验：真跑一遍 9 个页签，扫描渲染出来的文本。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_render_output.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_render_output.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\\n".join(out.strip().splitlines()[-14:])
        raise AssertionError("渲染输出存在问题：\\n%s" % tail)
    keep = [x.strip() for x in out.splitlines()
            if "已渲染" in x or "检查标签" in x]
    return "；".join(keep) if keep else "渲染输出检查通过"


def check_fund() -> str:
    """场内基金模块：类型识别、折溢价、跟踪误差、因子生成（离线合成数据）。"""
    from core import fund as FD
    from core.contracts import Category
    from core.selftest_data import make_ohlcv

    import numpy as np
    import pandas as pd

    cases = [("510300", "沪深300ETF华泰柏瑞", "ETF"),
             ("159915", "创业板ETF易方达", "ETF"),
             ("588000", "科创50ETF华夏", "ETF"),
             ("518880", "黄金ETF华安", "ETF"),
             ("161725", "白酒基金LOF", "LOF"),
             ("508000", "华安张江产业园REIT", "REITs"),
             ("600519", "贵州茅台", None),
             ("000001", "平安银行", None)]
    wrong = []
    for code, nm, want in cases:
        got = FD.classify_fund(code, nm)
        if want is None:
            if got["is_fund"]:
                wrong.append((code, nm, got["type"], "应判为非基金"))
        elif got["type"] != want:
            wrong.append((code, nm, got["type"], want))
    assert not wrong, "类型识别错误：%s" % wrong

    # 折溢价：构造 +0.2% 溢价
    n = 160
    df = make_ohlcv(n, seed=11, start_price=4.0)
    nav = pd.DataFrame({"date": df["date"], "nav": df["close"] * 0.998,
                        "acc_nav": df["close"],
                        "nav_pct": df["close"].pct_change() * 100,
                        "buy_state": ["开放申购"] * n,
                        "sell_state": ["开放赎回"] * n})
    nm2 = FD.nav_metrics(nav, df)
    assert abs(nm2["premium_pct"] - 0.2) < 0.06, nm2.get("premium_pct")
    # 跟踪误差：指数略强于基金
    idx = df.copy(); idx["close"] = df["close"] * 1.0004
    trk = FD.tracking_metrics(nav, idx)
    assert trk["trk_corr"] > 0.99, trk.get("trk_corr")
    # 因子
    fs = FD.fund_factors("510300", "沪深300ETF", df, nav, idx,
                         {"fund_size": 1200e8, "fund_size_text": "1200亿元"},
                         {"turnover": 3.5})
    assert len(fs) >= 5, "基金因子偏少：%d" % len(fs)
    for f in fs:
        assert f.category == Category.FUNDSPEC
        assert 0 <= f.strength <= 1 and 0 <= f.reliability <= 1
        assert f.detail and f.theory
    # 非基金代码应不产生基金因子
    assert FD.fund_factors("600519", "贵州茅台", df) == []
    assert FD.nav_metrics(None, None)["detail"]
    assert FD.tracking_metrics(None, None)["detail"]
    return ("类型识别 %d/%d 正确；折溢价 %+.3f%%；跟踪相关性 %.4f；因子 %d 条"
            % (len(cases), len(cases), nm2["premium_pct"], trk["trk_corr"], len(fs)))


def check_market_structure() -> str:
    """大盘三层嵌套 + 板块轮动（离线合成数据，不发网络请求）。"""
    import numpy as np
    import pandas as pd

    from core import market_board as MB
    from core import market_structure as MS
    from core.selftest_data import make_ohlcv

    n = 500
    # 牛市主升
    up = np.linspace(3000, 4200, n) + np.sin(np.arange(n) / 9) * 30
    df = make_ohlcv(n, seed=3, start_price=3000.0)
    df["close"] = up
    df["high"] = up * 1.012
    df["low"] = up * 0.988
    v = df["volume"].to_numpy(float).copy()
    v[:int(n * 0.7)] *= 0.8
    v[int(n * 0.7):] *= 1.6
    df["volume"] = v
    r = MS.analyze_market_structure(df, name="合成")
    assert r["regime"]["regime"] in ("牛市", "熊市", "震荡市")
    assert r["phase"]["label"], "应识别出波段"
    assert r["swing"]["structure"] in ("上升波段", "下降波段", "震荡波段")
    assert r["factors"], "应生成三层结构因子"
    for f in r["factors"]:
        assert 0 <= f.strength <= 1 and 0 <= f.reliability <= 1
        assert f.detail and f.theory and "None" not in str(f.value)
    assert r["discipline"]["position_advice"], "应给出仓位建议"
    # 熊市
    dn = np.linspace(4200, 3000, n) + np.sin(np.arange(n) / 8) * 25
    d2 = make_ohlcv(n, seed=7, start_price=4200.0)
    d2["close"] = dn
    d2["high"] = dn * 1.008
    d2["low"] = dn * 0.992
    r2 = MS.analyze_market_structure(d2)
    assert r2["regime"]["regime"] == "熊市", r2["regime"]["regime"]
    assert r2["phase"]["label"] in ("熊初（出清段）", "熊中（阴跌段）", "熊末（筑底段）"), \
        r2["phase"]["label"]
    # 板块宽度：双向采样应给出合理宽度（不能恒为 100%）
    boards = ([{"code": "BK%04d" % i, "name": "涨%d" % i, "pct": 2.0}
               for i in range(50)] +
              [{"code": "DK%04d" % i, "name": "跌%d" % i, "pct": -2.0}
               for i in range(50)])
    rot = MB.board_rotation(boards, {})
    # 50 涨 + 50 跌 → 宽度必须是 50%，不能像旧版那样恒报 100%
    assert 40 <= (rot.get("board_breadth_pct") or 0) <= 60, rot.get("board_breadth_pct")
    # 单向（全涨）时应判定为"不可靠"——这才是幸存者偏差的防线
    onesided = [{"code": "BK%04d" % i, "name": "涨%d" % i, "pct": 2.0}
                for i in range(100)]
    rot2 = MB.board_rotation(onesided, {})
    assert rot2.get("breadth_reliable") is False, "全是上涨时应标注为不可靠"
    assert "不完整" in (rot2.get("breadth_quality") or ""), rot2.get("breadth_quality")
    # 边界
    assert MS.analyze_market_structure(None)["detail"]
    assert MB.market_overview({})["detail"]
    assert MB.board_rotation([], {})["detail"]
    return ("大周期=%s｜波段=%s｜短线=%s｜板块宽度=%.0f%%（完整性已标注）"
            % (r["regime"]["regime"], r["phase"]["label"],
               r["swing"]["structure"], rot.get("board_breadth_pct") or 0))


def check_feature_wiring() -> str:
    """功能完整性：关键功能的界面接入点不得静默丢失。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_feature_wiring.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_feature_wiring.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\\n".join(out.strip().splitlines()[-12:])
        raise AssertionError("关键功能界面接入点缺失：\\n%s" % tail)
    line = [x for x in out.splitlines() if "app.py：" in x]
    return line[0].strip() if line else "功能完整性检查通过"


def check_v8_safety() -> str:
    """V8（py_mini_racer）并发安全：并行分析不得让进程崩溃。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_v8_safety.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_v8_safety.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    # 进程崩溃时 returncode 会是非 0 且输出含 FATAL/abort
    if r.returncode != 0 or "FATAL" in out:
        tail = "\\n".join(out.strip().splitlines()[-12:])
        raise AssertionError("V8 并发不安全（会导致进程崩溃）：\\n%s" % tail)
    keep = [x.strip() for x in out.splitlines()
            if "已加锁" in x or "进程存活" in x or "预热身" in x]
    return "；".join(keep) if keep else "V8 并发安全检查通过"


def check_cache_purity() -> str:
    """缓存函数必须纯粹：不得调用 Streamlit UI（否则缓存命中会崩）。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_cache_purity.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_cache_purity.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\\n".join(out.strip().splitlines()[-10:])
        raise AssertionError("缓存函数含 UI 调用（会抛 CacheReplayClosureError）：\\n%s"
                             % tail)
    line = [x for x in out.splitlines() if "检查了" in x]
    return line[0].strip() if line else "缓存纯净性检查通过"


def check_progress_tracker() -> str:
    """进度追踪器：单调递增、夹取、多线程并发安全。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "core", "progress.py")
    r = subprocess.run([sys.executable, "-m", "core.progress"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="ignore", timeout=120, cwd=_HERE)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "self-test OK" not in out:
        tail = "\\n".join(out.strip().splitlines()[-8:])
        raise AssertionError("进度追踪器自测失败：\\n%s" % tail)
    line = [x for x in out.splitlines() if "self-test OK" in x]
    return line[0].strip() if line else "进度追踪器 OK"


def check_render_smoke() -> str:
    """渲染函数冒烟：用真实 payload 实际调用每个 render_*（含全空降级）。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_render_smoke.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_render_smoke.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=900, cwd=_HERE)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\\n".join(out.strip().splitlines()[-16:])
        raise AssertionError("渲染函数冒烟失败（有崩溃或脏数据）：\\n%s" % tail)
    keep = [x.strip() for x in out.splitlines()
            if "发现" in x or "共调用" in x]
    return "；".join(keep) if keep else "渲染冒烟通过"


def check_prediction_basis() -> str:
    """预测基准语义：必须相对分析当日股价，不是目标日前一交易日。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_prediction_basis.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_prediction_basis.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=120, cwd=_HERE)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\\n".join(out.strip().splitlines()[-12:])
        raise AssertionError("预测基准语义不符：\\n%s" % tail)
    keep = [x.strip() for x in out.splitlines() if "累计（相对分析日）" in x]
    return keep[0] if keep else "预测基准语义正确"


def check_keypoint() -> str:
    """关键点动态加权：识别事件、调权、总权重守恒。"""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "core.keypoint"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="ignore", timeout=180, cwd=_HERE)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "self-test OK" not in out:
        tail = "\\n".join(out.strip().splitlines()[-10:])
        raise AssertionError("关键点模块自测失败：\\n%s" % tail)
    keep = [x.strip() for x in out.splitlines()
            if "聚焦度" in x or "总权重" in x]
    return "；".join(keep[:2]) if keep else "关键点加权 OK"


def check_market_forecast() -> str:
    """大盘/板块概率预测：8 周期、概率合法、区间自洽。"""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "core.market_forecast"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="ignore", timeout=300, cwd=_HERE)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "self-test OK" not in out:
        tail = "\\n".join(out.strip().splitlines()[-10:])
        raise AssertionError("大盘/板块预测自测失败：\\n%s" % tail)
    line = [x for x in out.splitlines() if "个周期" in x]
    return line[0].strip() if line else "大盘/板块预测 OK"


def check_market_fundamental() -> str:
    """大盘基本面替代物：估值分位/两融/北向/开户数 + 新鲜度过滤。"""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "core.market_fundamental"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="ignore", timeout=300, cwd=_HERE)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "self-test OK" not in out:
        tail = "\\n".join(out.strip().splitlines()[-10:])
        raise AssertionError("大盘基本面替代物自测失败：\\n%s" % tail)
    keep = [x.strip() for x in out.splitlines() if "生成" in x and "因子" in x]
    return keep[0] if keep else "大盘基本面替代物 OK"


def check_freshness() -> str:
    """通用数据新鲜度校验机制。"""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "core.freshness"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="ignore", timeout=120, cwd=_HERE)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "self-test OK" not in out:
        tail = "\\n".join(out.strip().splitlines()[-10:])
        raise AssertionError("新鲜度校验机制自测失败：\\n%s" % tail)
    return "通用新鲜度校验 OK（日5/周14/月70/季130 天）"


def check_retail_heat() -> str:
    """散户入场热度（开户数代理指标）。"""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "core.retail_heat"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="ignore", timeout=180, cwd=_HERE)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "self-test OK" not in out:
        tail = "\\n".join(out.strip().splitlines()[-10:])
        raise AssertionError("散户热度模块自测失败：\\n%s" % tail)
    return "散户入场热度代理 OK（成交额水位/振幅/两融合成）"


def check_render_wiring() -> str:
    """面板调用检查：每个 render_* 都必须真的被调用过（防"写了没挂"）。"""
    import subprocess
    import sys

    script = os.path.join(_HERE, "tests", "test_render_wiring.py")
    if not os.path.exists(script):
        raise FileNotFoundError("缺少 tests/test_render_wiring.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=120, cwd=_HERE)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\\n".join(out.strip().splitlines()[-14:])
        raise AssertionError("有渲染函数写了但没挂：\\n%s" % tail)
    line = [x.strip() for x in out.splitlines() if "渲染函数" in x]
    return line[0] if line else "面板调用检查通过"


def check_ai_analyst() -> str:
    """AI 复核模块：未配置 Key 时安全降级、摘要完整、不冒充数据。"""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "core.ai_analyst"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="ignore", timeout=180, cwd=_HERE)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "self-test OK" not in out:
        tail = "\\n".join(out.strip().splitlines()[-10:])
        raise AssertionError("AI 复核模块自测失败：\\n%s" % tail)
    keep = [x.strip() for x in out.splitlines() if "字符数" in x]
    return ("AI 复核 OK（未配置 Key 安全降级）" +
            ("；" + keep[0] if keep else ""))


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
    check("内置字体中文覆盖", check_font_coverage)
    check("缓存函数签名（防 UnhashableParamError）", check_cache_signature)
    check("缓存函数纯粹性（防 CacheReplayClosureError）", check_cache_purity)
    check("界面字段标签无英文残留", check_labels)
    check("因子展示值无 None/nan", check_factor_display)
    check("预测基准=分析当日股价", check_prediction_basis)
    check("关键点动态加权", check_keypoint)
    check("大盘/板块概率预测", check_market_forecast)
    check("大盘基本面替代物", check_market_fundamental)
    check("通用数据新鲜度校验", check_freshness)
    check("散户入场热度代理", check_retail_heat)
    check("AI 复核模块", check_ai_analyst)
    check("渲染层收敛（无手写渲染）", check_render_rules)
    check("关键功能界面接入完整", check_feature_wiring)
    check("端到端渲染输出扫描", check_render_output)
    check("渲染函数冒烟（真实payload×5）", check_render_smoke)
    check("面板调用检查（防写了没挂）", check_render_wiring)

    print("\n【2】数值内核正确性")
    check("特征/规则分因果性（无未来函数）", check_causality)
    check("特征数值健康（无 ±inf）", check_no_inf)
    check("概率/区间/波动率自洽性", check_probability_consistency)
    check("大盘/外围联动 + 主力意图", check_market_and_intent)
    check("自我学习闭环（回填→核账→校准）", check_self_learning)
    check("进度追踪器（并发安全）", check_progress_tracker)
    check("V8 并发安全（防进程崩溃）", check_v8_safety)
    check("场内基金（折溢价/跟踪/因子）", check_fund)
    check("大盘三层结构 + 板块轮动", check_market_structure)
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

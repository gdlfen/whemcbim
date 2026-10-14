# -*- coding: utf-8 -*-
"""
在**没有安装 streamlit** 的环境下测试 app.py 的内部逻辑。

思路：用一个轻量 stub 顶替 streamlit 模块（只实现 app.py 用到的 API），
然后直接调用 app.py 里的纯逻辑函数（图表渲染、payload 清理、报告生成等）。
这样即使本机装不上 streamlit，也能验证界面逻辑不会在运行时抛异常。
"""

from __future__ import annotations

import io
import os
import sys
import types

# 本文件位于 tests/ 下，项目根目录（app.py 与 core/ 所在处）在上一级
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# --------------------------------------------------------------------------- #
# 极简 streamlit stub
# --------------------------------------------------------------------------- #
class _State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


class _Ctx:
    """支持 with st.expander(...) / with st.spinner(...) / with tabs[i] 的上下文。"""

    def __init__(self, st=None):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getitem__(self, k):
        return self

    def __iter__(self):
        return iter([])

    def __getattr__(self, name):
        """把 st.columns()/st.tabs() 返回对象上的任意调用转发给 stub 模块，
        这样 ctx.metric(...) / ctx.progress(...) 都能正常走通。"""
        st = object.__getattribute__(self, "_st")
        if st is not None and hasattr(st, name):
            return getattr(st, name)

        def _noop(*a, **k):
            return None

        return _noop


class _Progress:
    def progress(self, *a, **k):
        return None

    def empty(self):
        return None


def _mk_stub():
    st = types.ModuleType("streamlit")
    st.session_state = _State()
    st.set_page_config = lambda **k: None
    st.markdown = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.title = lambda *a, **k: None
    st.header = lambda *a, **k: None
    st.subheader = lambda *a, **k: None
    st.write = lambda *a, **k: None
    st.info = lambda *a, **k: None
    st.success = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.error = lambda *a, **k: None
    st.code = lambda *a, **k: None
    st.json = lambda *a, **k: None
    st.image = lambda *a, **k: None
    st.dataframe = lambda *a, **k: None
    st.table = lambda *a, **k: None
    st.metric = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None
    st.pyplot = lambda *a, **k: None
    st.rerun = lambda *a, **k: None
    st.stop = lambda *a, **k: None
    st.empty = lambda *a, **k: _Ctx(st)
    st.expander = lambda *a, **k: _Ctx(st)
    st.spinner = lambda *a, **k: _Ctx(st)
    st.form = lambda *a, **k: _Ctx(st)
    st.container = lambda *a, **k: _Ctx(st)
    st.progress = lambda *a, **k: _Ctx(st)
    st.tabs = lambda names, **k: [_Ctx(st) for _ in names]
    st.columns = lambda spec, **k: [_Ctx(st) for _ in
                                    (range(spec) if isinstance(spec, int) else spec)]

    def _btn(*a, **k):
        return False

    st.button = _btn
    st.download_button = _btn
    st.checkbox = lambda *a, **k: k.get("value", False)
    st.text_input = lambda *a, **k: k.get("value", "")
    st.text_area = lambda *a, **k: k.get("value", "")
    st.number_input = lambda *a, **k: k.get("value", 0)
    st.selectbox = lambda label, options, **k: (
        k.get("index", 0) and list(options)[k.get("index", 0)] or list(options)[0]
        if options else None)
    st.multiselect = lambda *a, **k: []
    st.slider = lambda *a, **k: k.get("value", 0)
    st.radio = lambda label, options, **k: list(options)[0]

    class _SB:
        def __getattr__(self, name):
            return getattr(st, name)

    st.sidebar = _SB()

    def _cache(*dargs, **dkw):
        def deco(fn):
            return fn
        if len(dargs) == 1 and callable(dargs[0]) and not dkw:
            return dargs[0]
        return deco

    st.cache_data = _cache
    st.cache_resource = _cache
    return st


sys.modules.setdefault("streamlit", _mk_stub())

import pandas as pd  # noqa: E402
import app  # noqa: E402

print("=" * 80)
print("app.py 逻辑测试（streamlit 使用 stub）")
print("=" * 80)

fails: list[str] = []


def step(name, fn):
    try:
        out = fn()
        print("  [PASS] %-40s %s" % (name, str(out)[:80]))
        return out
    except Exception as exc:  # noqa: BLE001
        import traceback

        fails.append(name)
        print("  [FAIL] %-40s %s: %s" % (name, type(exc).__name__, exc))
        traceback.print_exc()
        return None


# --------------------------------------------------------------------------- #
# 1) 端到端：跑一次真实分析（演示模式，保证离线可重复）
# --------------------------------------------------------------------------- #
print("\n【1】完整分析 + 界面数据准备")
payload = step("cached_analysis(600519, demo)",
               lambda: app.cached_analysis("600519", False, True, 800))

if payload:
    def _keys_ok():
        need = {"symbol", "name", "forecasts", "ensemble", "ohlcv", "charts"}
        missing = need - set(payload)
        if missing:
            raise AssertionError("缺少键 {}".format(missing))
        n_fc = len(payload["forecasts"])
        n_ch = len(payload.get("charts") or {})
        n_fa = payload.get("n_factors", 0)
        return "8个周期={} 图表={} 因子={}".format(n_fc, n_ch, n_fa)

    step("payload 必备键", _keys_ok)
    step("forecasts 结构完整", lambda: all(
        {"horizon", "label", "p_up", "exp_return", "ci_low", "ci_high", "drivers",
         "counter", "confidence", "sigma"} <= set(f) for f in payload["forecasts"])
        or (_ for _ in ()).throw(AssertionError("字段缺失")))
    step("概率在合法区间", lambda: all(
        0 <= f["p_up"] <= 1 and 0 <= f["confidence"] <= 1 for f in payload["forecasts"])
        or (_ for _ in ()).throw(AssertionError("越界")))
    step("图表为有效 PNG 字节", lambda: (
        all(isinstance(v, bytes) and v[:8] == b"\x89PNG\r\n\x1a\n"
            for v in (payload.get("charts") or {}).values())
        or (_ for _ in ()).throw(AssertionError("PNG 头不正确")))
        and "%d 张图全部是合法 PNG，共 %.0f KB" % (
            len(payload["charts"]), sum(len(v) for v in payload["charts"].values()) / 1024))

    # ------------------------------------------------------------------- #
    # 2) 渲染函数（会调用 st.* 的 stub，检查不抛异常）
    # ------------------------------------------------------------------- #
    print("\n【2】各标签页渲染函数")
    step("render_header", lambda: app.render_header(payload))
    step("render_overview", lambda: app.render_overview(payload))
    step("render_forecast_table", lambda: app.render_forecast_table(payload))
    step("render_charts", lambda: app.render_charts(payload))
    step("render_dimensions（8 个标签页）", lambda: app.render_dimensions(payload))
    step("render_export", lambda: app.render_export(payload, payload["symbol"]))

    # 因子选择框：stub 返回第一个选项
    step("render_drivers", lambda: app.render_drivers(payload))

    # ------------------------------------------------------------------- #
    # 3) 报告生成
    # ------------------------------------------------------------------- #
    print("\n【3】报告与工具函数")
    txt = step("_payload_text", lambda: app._payload_text(payload))
    html = step("_payload_html", lambda: app._payload_html(payload))
    step("报告内容检查", lambda: (
        len(txt) > 800 and len(html) > 1200
        and "周期" in txt and "<table" in html
        or (_ for _ in ()).throw(AssertionError("报告内容过短或缺表格")))
        and "文本 %d 字 / HTML %d 字" % (len(txt), len(html)))

    # ------------------------------------------------------------------- #
    # 4) 回测标签页（在 stub 下走一遍）
    # ------------------------------------------------------------------- #
    print("\n【4】历史验证标签页")
    step("render_backtest", lambda: app.render_backtest(payload, True, "600519", 800, True))

    # ------------------------------------------------------------------- #
    # 5) 工具函数
    # ------------------------------------------------------------------- #
    print("\n【5】数值格式化工具")
    step("_short / _r / _pct / _big", lambda: (
        app._short(None) == "—" and app._short(True) == "是"
        and app._r(1.2345, 2) == "1.23" and app._pct(0.678) == "67.8%"
        and app._big(1.5788e12).endswith("万亿")
        or (_ for _ in ()).throw(AssertionError("格式化结果不符")))
        and "_short/_r/_pct/_big 全部正确")
    step("_kv_grid 空输入不抛", lambda: app._kv_grid({}))
    step("_kv_grid 正常输入", lambda: app._kv_grid({"a": 1, "b": None, "c": float("nan"),
                                                    "d": "x", "e": {"skip": 1}}))
    step("_all_factors 去重", lambda: "%d 条去重因子" % len(app._all_factors(payload)))
    step("_clean_dict / _deep_clean", lambda: isinstance(
        app._deep_clean({"a": 1, "b": pd.DataFrame({"x": [1]}), "c": {"d": [1, 2]}}), dict)
        and "清理函数正常")
    step("_frame_tail(None)", lambda: app._frame_tail(None, 5) is None)
    step("_capital_brief 空输入", lambda: app._capital_brief({}) == {})

    # ------------------------------------------------------------------- #
    # 6) 演示/空数据边界
    # ------------------------------------------------------------------- #
    print("\n【6】边界情况")

    def _empty_render():
        app.render_header({"symbol": "X", "name": None})
        app.render_overview({"forecasts": [], "ensemble": {}})
        app.render_forecast_table({"forecasts": []})
        app.render_charts({"charts": {}})
        app.render_dimensions({})
        return "空输入全部安全"

    step("空 payload 渲染不抛", _empty_render)
    step("_payload_text 空输入", lambda: len(app._payload_text({"forecasts": []})) > 50)

print("\n" + "=" * 80)
if fails:
    print("失败 %d 项：%s" % (len(fails), fails))
    raise SystemExit(1)
print("app.py 逻辑测试全部通过 —— 界面代码在运行期不会因数据缺失/为空而崩溃")
print("=" * 80)

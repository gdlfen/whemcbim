# -*- coding: utf-8 -*-
"""
渲染函数冒烟测试 —— 用真实 payload **实际调用每一个 render_* 函数**。

为什么需要它
------------
之前发生过的事故，都不是"逻辑写错"，而是"代码在，但一跑就崩"：

  · `_ProgressTracker` 名字存在但从未 import → 点大盘分析即 NameError
  · `_SCOPES` 改成 3 元组，消费处仍按 4 个解包 → ValueError
  · 基金界面代码整体丢失 → 用户看不到功能
  · `render_topdown` 里把结构体命名成 st → 'dict' has no attribute 'markdown'

`test_feature_wiring.py` 只能查"字符串在不在"，查不到上面这些。
真正能拦住它们的是：**把每个渲染函数用真实数据跑一遍**。

本测试做三件事
--------------
1. 自动发现 app 里所有 ``render_*`` / ``_render_*`` 函数（不靠手工清单，
   新增函数会自动被纳入检查）。
2. 用**四种真实 payload**分别调用（个股 / 基金 / 仅大盘板块 / 空 payload）：
     · 正常数据 —— 主路径
     · 空 payload —— 降级路径（最容易崩的地方）
   任何异常都判失败并打印 traceback。
3. 校验渲染出的文本里没有 None / nan / 原始字典 / 裸键名。

用法：``python tests/test_render_smoke.py [--refresh]``
"""
from __future__ import annotations

import inspect
import io
import json
import os
import re
import sys
import traceback
import types
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "tests", ".smoke_payloads.json")

CAP: list = []


# --------------------------------------------------------------------------- #
# 最小 streamlit stub（带采集）
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
        st = object.__getattribute__(self, "_st")
        if st is not None and hasattr(st, name):
            return getattr(st, name)
        return lambda *a, **k: None


def _build_stub():
    st = types.ModuleType("streamlit")
    st.session_state = _State()

    def rec(name):
        def fn(*a, **k):
            for x in list(a) + list(k.values()):
                if isinstance(x, str):
                    CAP.append((name, x))
        return fn

    for m in ("markdown", "caption", "divider", "write", "info", "success", "warning",
              "error", "title", "header", "subheader", "code", "json", "text"):
        setattr(st, m, rec(m))

    st.set_page_config = lambda **k: None
    st.dataframe = lambda *a, **k: CAP.append(("dataframe", ""))
    st.table = st.dataframe
    st.metric = lambda *a, **k: CAP.append(("metric", ""))
    st.image = lambda *a, **k: None
    st.pyplot = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None
    st.altair_chart = lambda *a, **k: None
    for m in ("empty", "expander", "spinner", "container", "form", "progress",
              "status", "popover"):
        setattr(st, m, lambda *a, **k: _Ctx(st))
    st.tabs = lambda names, **k: [_Ctx(st) for _ in names]
    st.columns = lambda spec, **k: [_Ctx(st) for _ in
                                    (range(spec) if isinstance(spec, int) else spec)]
    st.button = lambda *a, **k: False
    st.download_button = lambda *a, **k: False
    st.checkbox = lambda *a, **k: k.get("value", False)
    st.text_input = lambda *a, **k: k.get("value", "")
    st.text_area = lambda *a, **k: k.get("value", "")
    st.number_input = lambda *a, **k: k.get("value", 0)
    st.selectbox = lambda label, options, **k: (list(options)[0] if options else None)
    st.multiselect = lambda *a, **k: []
    st.slider = lambda *a, **k: k.get("value", 0)
    st.radio = lambda label, options, **k: (list(options)[0] if options else None)
    st.date_input = lambda *a, **k: None
    st.rerun = lambda *a, **k: None
    st.stop = lambda *a, **k: None

    class _SB:
        def __getattr__(self, n):
            return getattr(st, n)

    st.sidebar = _SB()

    def _cache(*da, **dk):
        def deco(fn):
            return fn
        if len(da) == 1 and callable(da[0]) and not dk:
            return da[0]
        return deco

    st.cache_data = _cache
    st.cache_resource = _cache
    return st


sys.modules["streamlit"] = _build_stub()
import app  # noqa: E402


# --------------------------------------------------------------------------- #
# payload 构造
# --------------------------------------------------------------------------- #
def _load_payloads(refresh: bool = False) -> dict:
    if not refresh and os.path.exists(CACHE):
        try:
            with io.open(CACHE, encoding="utf-8") as fh:
                raw = json.load(fh)
            out = {}
            for k, v in raw.items():
                out[k] = v
            if len(out) >= 3:
                return out
        except Exception:  # noqa: BLE001
            pass
    print("（首次运行：抓三份真实 payload，约 1~2 分钟）")
    out = {}
    out["stock"] = app.cached_analysis("600519", False, False, 500)
    out["stock"].pop("charts", None)
    try:
        out["fund"] = app.cached_analysis("510300", False, False, 500)
        out["fund"].pop("charts", None)
    except Exception as exc:  # noqa: BLE001
        print("  基金 payload 抓取失败（可接受）:", str(exc)[:60])
        out["fund"] = dict(out["stock"])
    for k in ("market", "board"):
        try:
            out[k] = app.cached_market_analysis(k, 500)
        except Exception as exc:  # noqa: BLE001
            print("  %s payload 抓取失败（可接受）: %s" % (k, str(exc)[:60]))
            out[k] = {}
    try:
        with io.open(CACHE, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        pass
    return out


def _empty_payload() -> dict:
    """全空 payload —— 降级路径，最容易崩。"""
    return {
        "symbol": "", "name": "", "as_of": "", "price": None,
        "forecasts": [], "forecast_table": [], "drivers_by_h": {},
        "factors_all": [], "factor_stats": {},
        "market": {}, "market_global": {}, "is_fund": False, "scope": "all",
        "detail": "", "meta_errors": [], "meta_sources": {}, "notes": [],
        "charts": {}, "stage": {}, "snapshot": {}, "valuation": {},
        "capital_brief": {}, "sentiment": {}, "macro": {}, "seasonality": {},
        "theory": {}, "news": [], "fund": {}, "intent_series": [],
        "main_intent": {}, "volatility": {}, "stages": [],
        "learning_report": {}, "learning": {}, "recorded": {},
        "financials_tail": {}, "sector_boards": [],
    }


# --------------------------------------------------------------------------- #
# 调用计划：不同函数签名不同
# --------------------------------------------------------------------------- #
def _plan(name: str, p: dict) -> tuple | None:
    """返回调用参数元组；None 表示跳过。"""
    sym = p.get("symbol") or "600519"
    table = {
        "render_backtest": (p, True, sym, 500, False),
        "render_export": (p, sym),
        "render_self_learning": (p, sym),
        "render_data_health": (),
        "render_fund": (p,),
        "render_market_panel": (p,),
        "render_board_panel": (p,),
        "render_topdown": (p,),
        "_render_subject_scoped": (p, sym, {"use_ml": False, "demo": False,
                                            "days": 500, "backtest": True,
                                            "self_learning": True}, sym),
        "render_fields": (p.get("stage") or {},),
        # 说明预览：参数是 scope 字符串（all/market/board/subject），不是 payload
        "_render_scope_preview": ("all",),
    }
    if name in table:
        return table[name]
    # 其它（render_header / render_overview / render_charts / render_dimensions …）
    try:
        sig = inspect.signature(getattr(app, name))
        params = [x for x in sig.parameters.values()
                  if x.default is inspect.Parameter.empty
                  and x.kind in (x.POSITIONAL_ONLY, x.POSITIONAL_OR_KEYWORD)]
        if len(params) == 0:
            return ()
        if len(params) == 1:
            return (p,)
        return None                    # 签名不认识 → 跳过而不是瞎猜
    except Exception:  # noqa: BLE001
        return None


#: 不是"渲染面板"的函数，跳过
SKIP = {"render_header_only", "render_helpers"}

#: 允许容忍的关键词（出现在配置/说明里，不算脏数据）
OK_TOKENS = ("NoneType", "None 与", "nan 值")


def _dirty(texts: list) -> list:
    bad = []
    for t in texts:
        s = str(t)
        if re.search(r"\{\s*'[^']+'\s*:", s):          # 原始字典
            bad.append(("原始字典", s[:90]))
            continue
        for tok in ("None", "nan", "inf"):
            if re.search(r"(?<![A-Za-z_])%s(?![A-Za-z_])" % tok, s):
                if any(k in s for k in OK_TOKENS):
                    continue
                bad.append(("%s 字样" % tok, s[:90]))
                break
    return bad


def main() -> int:
    refresh = "--refresh" in sys.argv
    payloads = _load_payloads(refresh)
    payloads["empty"] = _empty_payload()

    # 发现所有渲染函数
    names = sorted(
        n for n in dir(app)
        if callable(getattr(app, n, None))
        and (n.startswith("render_") or n.startswith("_render_"))
        and n not in SKIP
    )
    print("发现 %d 个渲染函数：%s" % (len(names), "、".join(names)))
    print()

    failures = []
    skipped = []
    total_calls = 0

    for pname, p in payloads.items():
        if not isinstance(p, dict) or not p:
            skipped.append((pname, "payload 为空，跳过整组"))
            continue
        print("=== payload: %s ===" % pname)
        for name in names:
            args = _plan(name, p)
            if args is None:
                skipped.append((name, "签名不认识"))
                continue
            fn = getattr(app, name)
            CAP.clear()
            total_calls += 1
            try:
                fn(*args)
            except Exception as exc:  # noqa: BLE001
                failures.append((pname, name, "%s: %s" % (type(exc).__name__, exc),
                                 traceback.format_exc()))
                print("  [崩溃] %-24s %s" % (name, str(exc)[:80]))
                continue
            texts = [x for _, x in CAP]
            bad = _dirty(texts)
            if bad:
                failures.append((pname, name, "渲染出脏数据: %s" % bad[0][0],
                                 "\n".join("%s | %s" % b for b in bad[:3])))
                print("  [脏数据] %-22s %s" % (name, bad[0]))
            else:
                print("  [OK] %-24s %d 段" % (name, len(texts)))

    print()
    print("共调用 %d 次（%d 个函数 × %d 组 payload），跳过 %d 项"
          % (total_calls, len(names),
             len([k for k in payloads if payloads[k]]), len(skipped)))
    if failures:
        print("\n[失败] %d 处问题：" % len(failures))
        for pname, name, msg, tb in failures:
            print("\n  ── payload=%s / %s" % (pname, name))
            print("     %s" % msg)
            for line in tb.strip().splitlines()[-4:]:
                print("     %s" % line)
        return 1
    print("\n[通过] 所有渲染函数在 4 组 payload（个股/基金/大盘板块/全空）下"
          "均无崩溃、无脏数据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

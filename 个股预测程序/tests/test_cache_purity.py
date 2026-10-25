# -*- coding: utf-8 -*-
"""
缓存函数纯净性校验 —— 被 @st.cache_data / @st.cache_resource 装饰的函数
**不得调用任何 Streamlit UI**。

事故
----
给缓存函数传入进度回调，回调里执行 `prog.progress(...)`，
而 `prog` 是缓存函数外部创建的布局块。缓存命中时 Streamlit 需重放消息：

    KeyError: '1774911944144'
    CacheReplayClosureError: While running cached_market_analysis(),
    a streamlit element is called on some layout block created outside the function.

Streamlit 的硬性规则：缓存函数必须纯粹（无 UI、无副作用），
所以进度提示必须由**调用方**用 st.spinner / st.progress 包裹。

本测试用 AST 精确检查（只查真实调用，跳过文档字符串与注释）：
  · 直接调用 st.*
  · 调用 ctx.metric / prog.progress 这类布局块方法
  · 调用名为 *_cb / callback 的回调参数

用法：``python tests/test_cache_purity.py``
"""

from __future__ import annotations

import ast
import io
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, "app.py")

#: Streamlit 的 UI 函数名（缓存函数里出现即违规）
ST_UI = {
    "markdown", "write", "caption", "info", "success", "warning", "error",
    "title", "header", "subheader", "text", "code", "json", "dataframe",
    "table", "metric", "image", "progress", "spinner", "expander", "tabs",
    "columns", "container", "form", "button", "checkbox", "radio", "selectbox",
    "slider", "text_input", "text_area", "number_input", "multiselect",
    "date_input", "time_input", "file_uploader", "color_picker", "download_button",
    "pyplot", "plotly_chart", "altair_chart", "line_chart", "bar_chart",
    "area_chart", "map", "toast", "balloons", "snow", "empty", "sidebar",
    "set_page_config", "rerun", "stop",
}

#: 回调类参数名特征
CB_NAMES = ("cb", "callback")


def _call_name(node: ast.Call) -> tuple[str, str]:
    f = node.func
    if isinstance(f, ast.Name):
        return "", f.id
    if isinstance(f, ast.Attribute):
        base = f.value
        bname = base.id if isinstance(base, ast.Name) else ""
        return bname, f.attr
    return "", ""


def _docstring_nodes(fn: ast.FunctionDef) -> set:
    """收集函数文档字符串所在的节点，避免误判。"""
    out = set()
    if fn.body and isinstance(fn.body[0], ast.Expr):
        v = fn.body[0].value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            out.add(id(v))
    return out


def main() -> int:
    src = io.open(APP, encoding="utf-8").read()
    tree = ast.parse(src)
    problems = []
    checked = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decos = []
        for d in node.decorator_list:
            if isinstance(d, ast.Call):
                d = d.func
            decos.append(getattr(d, "attr", getattr(d, "id", "")))
        if not any(x in ("cache_data", "cache_resource") for x in decos):
            continue
        checked += 1
        skip = _docstring_nodes(node)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and id(sub) in skip:
                continue
            if not isinstance(sub, ast.Call):
                continue
            base, name = _call_name(sub)
            if base in ("st", "streamlit") and name in ST_UI:
                problems.append((node.name, sub.lineno,
                                 "直接调用 st.%s（缓存函数必须纯粹）" % name))
            elif name in ("progress", "empty", "spinner", "metric") and base:
                problems.append((node.name, sub.lineno,
                                 "调用布局块方法 %s.%s()" % (base, name)))
        # 回调参数
        for a in list(node.args.args) + list(node.args.kwonlyargs):
            nm = a.arg
            if nm.startswith("_") and nm.lstrip("_").endswith(CB_NAMES):
                pass
            if any(nm.rstrip("_").endswith(c) or nm.rstrip("_") == c for c in CB_NAMES):
                problems.append((node.name, a.lineno,
                                 "接收回调参数 %r（会绕过纯净性检查）" % nm))

    print("检查了 %d 个缓存函数（AST 精确检查，已跳过文档字符串）" % checked)
    if problems:
        print("\n[失败] 以下缓存函数不是纯粹的，缓存命中时会抛"
              " CacheReplayClosureError：")
        for fn, ln, msg in problems:
            print("   第 %-5d 行 %s：%s" % (ln, fn, msg))
        print("\n修法：把 UI 移到调用方（st.spinner 包裹），"
              "缓存函数只做纯计算。")
        return 1
    print("\n[通过] 所有缓存函数均为纯计算，未调用任何 Streamlit UI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

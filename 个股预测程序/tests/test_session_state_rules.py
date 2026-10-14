# -*- coding: utf-8 -*-
"""
静态检查：确保没有"在部件实例化之后写入它的 session_state 键"这种写法。

这是 Streamlit 最容易被忽视、又最容易在真机上炸掉的错误，典型报错：

    st.session_state.symbol_input cannot be modified after the widget
    with that key is instantiated.

规则：写入某个 **部件 key** 的代码，必须位于 ``on_click`` / ``on_change``
回调函数里（回调在脚本重跑前执行，因此合法）。

用法：``python tests/test_session_state_rules.py``
"""

from __future__ import annotations

import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, "app.py")

#: 已知的安全回调函数名（在其中写 session_state 是允许的）
CALLBACKS = {"_apply_symbol"}


def widget_keys(tree: ast.AST) -> set:
    """收集所有 st.xxx(..., key="yyy") 的 key 名。"""
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            v = node.func.value
            if isinstance(v, ast.Name) and v.id == "st":
                for kw in node.keywords:
                    if kw.arg == "key" and isinstance(kw.value, ast.Constant):
                        keys.add(str(kw.value.value))
    return keys


def session_state_writes(tree: ast.AST) -> list:
    """收集所有 st.session_state[<常量>] = ... 的 (函数名, 行号, 键名)。"""
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for sub in ast.walk(fn):
            if not isinstance(sub, ast.Assign):
                continue
            for t in sub.targets:
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Attribute)
                        and t.value.attr == "session_state"
                        and isinstance(t.slice, ast.Constant)):
                    out.append((fn.name, sub.lineno, str(t.slice.value)))
    return out


def main() -> int:
    if not os.path.exists(APP):
        print("找不到 app.py:", APP)
        return 1
    tree = ast.parse(io.open(APP, encoding="utf-8").read())
    keys = widget_keys(tree)
    writes = session_state_writes(tree)

    print("带 key 的部件：%s" % (sorted(keys) or "无"))
    print("session_state 写入点：%s" % (writes or "无"))

    illegal = [(f, ln, k) for f, ln, k in writes if k in keys and f not in CALLBACKS]
    module_level = [(f, ln, k) for f, ln, k in writes if f == "<module>"]

    if illegal:
        print("\n[失败] 以下写入发生在非回调函数中，真机上会报错：")
        for f, ln, k in illegal:
            print("   %s() 第 %d 行 -> session_state['%s']" % (f, ln, k))
        return 1
    if module_level:
        print("\n[失败] 模块级（函数外）写入了 session_state：%s" % module_level)
        return 1

    print("\n[通过] 所有 session_state 写入都在回调函数内，"
          "不会触发 'cannot be modified after the widget' 错误")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

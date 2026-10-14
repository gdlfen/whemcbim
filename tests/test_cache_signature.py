# -*- coding: utf-8 -*-
"""
校验：被 @st.cache_data 装饰的函数，其参数必须都能被 Streamlit 哈希。

为什么需要这个测试
------------------
真实踩过的坑：给 cached_analysis 加了一个 progress_cb 回调参数后，
Streamlit 报 UnhashableParamError（函数对象无法哈希/序列化）。
本地测试之所以没发现，是因为测试**直接调用**了函数、绕过了缓存装饰器。

规则：任何传给 @st.cache_data 函数的参数，若不能被 pickle 化，
就必须在形参名前加下划线（Streamlit 约定：带下划线的参数不参与缓存键）。

用法：``python tests/test_cache_signature.py``
"""

from __future__ import annotations

import ast
import io
import os
import pickle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, "app.py")

#: 已知可安全跳过哈希的"困难类型"（Streamlit 对这些类型本身有内置哈希器）
SELF_HASHED = {"DataFrame", "Series", "ndarray", "Figure", "Timestamp", "date",
               "datetime", "bytes", "bytearray"}


def main() -> int:
    tree = ast.parse(io.open(APP, encoding="utf-8").read())
    problems: list = []
    checked = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # 只检查被 st.cache_data / st.cache_resource 装饰的函数
        deco_names = []
        for d in node.decorator_list:
            if isinstance(d, ast.Call):
                d = d.func
            if isinstance(d, ast.Attribute):
                deco_names.append(d.attr)
            elif isinstance(d, ast.Name):
                deco_names.append(d.id)
        if not any(n in ("cache_data", "cache_resource") for n in deco_names):
            continue
        checked += 1
        args = list(node.args.args) + list(node.args.kwonlyargs)
        for a in args:
            name = a.arg
            if name in ("self", "cls"):
                continue
            if name.startswith("_"):
                continue          # Streamlit 约定：下划线前缀 = 不参与哈希
            # 该参数若只能接收到不可 pickle 的值，就会出问题；
            # 这里只能做静态判断——凡是名字像"回调/函数/连接/客户端"的都提示
            suspicious = ("cb", "callback", "func", "fn", "hook", "handler",
                          "client", "conn", "session", "logger", "progress")
            if any(s in name.lower() for s in suspicious):
                problems.append((node.name, name, a.lineno))
    print("检查了 %d 个缓存函数" % checked)

    # 额外做一次真实 pickle 探测：函数对象确实不可 pickle
    try:
        pickle.dumps(lambda x: x)
        print("[警告] 本环境 lambda 竟然可 pickle，测试前提不成立")
    except Exception:
        pass

    if problems:
        print("\n[失败] 以下缓存函数参数属于易引发 UnhashableParamError 的类型，"
              "但没有下划线前缀：")
        for fn, arg, ln in problems:
            print("   %s(...) 第 %d 行 -> 参数 %r（应改名为 _%s）" % (fn, ln, arg, arg))
        return 1
    print("\n[通过] 所有 @st.cache_data 函数的回调类参数都带下划线前缀，"
          "不会触发 UnhashableParamError")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""
面板调用检查 —— 确保每个 ``render_*`` 函数都**真的被调用过**。

为什么需要它
------------
项目里出现过三次同类事故，都是"代码写了但没挂上"：
  1. 基金界面代码整体丢失（PowerShell here-string 替换静默失败）
  2. 进度条 ``_ProgressTracker`` 未导入（锚点写成 ``as config``，实为 ``as cfg``）
  3. 手动输入区 ``render_macro_manual`` 定义了但从未调用
     （``replace`` 没加断言，没匹配上就静默跳过）

现有的 ``test_feature_wiring.py`` 只能查"某些字符串在不在"，
查不出"函数定义了但没被调用"——这正是第 3 类事故的形态。

本测试用 AST 做两件事：
  · 收集 app.py 里所有 ``render_*`` / ``_render_*`` 函数定义
  · 收集所有调用点（``ast.Call``）
  · 差集即为"写了没挂"的函数，直接判失败

用法：``python tests/test_render_wiring.py``
"""
from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, "app.py")

#: 允许"只定义不调用"的函数（正常应为空；留出白名单以防误报）
ALLOW_UNCALLED: set[str] = set()

#: 名字里含这些片段的算"渲染面板"，必须被调用
PREFIXES = ("render_", "_render_")


def _is_render(name: str) -> bool:
    return any(name.startswith(p) or ("_" + p) in name for p in PREFIXES)


def main() -> int:
    if not os.path.exists(APP):
        print("[失败] 找不到 app.py")
        return 1
    src = open(APP, encoding="utf-8").read()
    tree = ast.parse(src)

    defs: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_render(node.name):
                defs[node.name] = node.lineno

    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                calls.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                calls.add(fn.attr)

    if not defs:
        print("[失败] 没有发现任何 render_* 函数，检查规则可能失效")
        return 1

    uncalled = sorted(
        (name, ln) for name, ln in defs.items()
        if name not in calls and name not in ALLOW_UNCALLED
    )

    print("扫描 %s" % APP)
    print("  渲染函数 %d 个，调用点 %d 个"
          % (len(defs), len([c for c in calls if c in defs])))
    for name, ln in sorted(defs.items(), key=lambda kv: kv[1]):
        mark = "  " if name in calls else "!!"
        print("  %s %-28s 定义于第 %-5d %s"
              % (mark, name, ln, "已调用" if name in calls else "**从未调用**"))

    if uncalled:
        print("\n[失败] %d 个渲染函数定义了但从未被调用"
              "（写了没挂，用户看不到）：" % len(uncalled))
        for name, ln in uncalled:
            print("  · %-28s 定义于第 %d 行" % (name, ln))
        print("\n修法：在 sidebar() 或对应的面板函数里调用它；")
        print("      若确属内部辅助函数且无需直接调用，加入 ALLOW_UNCALLED。")
        return 1

    print("\n[通过] %d 个渲染函数全部有调用点，无「写了没挂」" % len(defs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

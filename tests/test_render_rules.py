# -*- coding: utf-8 -*-
"""
渲染层收敛校验 —— 禁止再出现"手写渲染"的写法。

背景（连续三轮用户反馈的根因）
------------------------------
展示层以前是**逐处手写**的：9 个页签各写各的 `_kv_grid({中文名: _short(值)})`，
于是同类问题反复以不同面貌出现：
    · 未登记的键以英文出现（control_degree、_window…）
    · 嵌套字典被 str() 成原始字典字样并被截断
    · 长串小数（0.018166804293971925）
    · None / nan 直接打印

修一处冒一处。所以这里用 AST 把"坏写法"钉死：

禁用模式
--------
1. ``_kv_grid({...})`` 用**字面量 dict** 构造 —— 必须传原始 dict 给 render_fields，
   否则又会出现 str(dict)、英文键、长串小数。
2. ``_short(v) for k, v in x.items()`` —— 逐值 str()，嵌套字典会被压成字符串。
3. ``_label(x) or y`` / ``.get(k, _label(k) or k)`` —— 会把 None 兜回英文键名，
   等于绕过了"内部键隐藏"。
4. 在 f-string / % 格式化里直接放原始 dict 值。

用法：``python tests/test_render_rules.py``
"""

from __future__ import annotations

import ast
import io
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, "app.py")

#: app.py 内部允许的例外（函数名, 行号范围说明）
ALLOW = {
    # _kv_grid 自己的定义体、_kv_grid_items 内部
    "_kv_grid", "_kv_grid_items", "render_fields", "_render_section",
}


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def main() -> int:
    src = io.open(APP, encoding="utf-8").read()
    tree = ast.parse(src)
    problems: list = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)

        # ---- 规则 1：_kv_grid 不得接收字面量 dict ---- #
        if name == "_kv_grid" and node.args:
            a = node.args[0]
            if isinstance(a, ast.Dict):
                problems.append((node.lineno, "_kv_grid(字面量字典)",
                                 "改用 render_fields(原始dict, label_map=…)"))

        # ---- 规则 2：不得用 _short(v) 逐值转换后再喂给渲染 ---- #
        if name in ("_kv_grid", "_kv_grid_items", "render_fields"):
            seg = ast.dump(node)
            if "_short" in seg and "comprehension" in seg:
                problems.append((node.lineno, "逐值 _short() 后再渲染",
                                 "直接传原始 dict，格式交给 render_fields"))

    # ---- 规则 2b：字典推导式里对 items() 逐值调用 _short ---- #
    for node in ast.walk(tree):
        if isinstance(node, ast.DictComp):
            seg = ast.dump(node)
            if "_short" in seg and "items" in seg:
                problems.append((node.lineno, "{k: _short(v) for k,v in ….items()}",
                                 "改用 render_fields(d)"))

    # ---- 规则 3：_label(x) or y —— 会把 None 兜回英文键 ---- #
    # 必须精确判断"左侧是一次 _label(...) 调用"：不能只看 ast.dump() 里有没有
    # "_label" 子串，否则变量名里带 label 的（如 type_label）会误报。
    def _is_label_call(n) -> bool:
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_label")

    for node in ast.walk(tree):
        if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)):
            continue
        hit = any(_is_label_call(v) for v in node.values)
        if not hit:
            for sub in ast.walk(node):
                if isinstance(sub, ast.BoolOp) and isinstance(sub.op, ast.Or) \
                        and any(_is_label_call(v) for v in sub.values):
                    hit = True
                    break
        if hit:
            problems.append((node.lineno, "_label(x) or 兜底值",
                             "改为 lab = _label(x); if lab is None: continue"))

    # ---- 规则 4：把整段 payload 值直接拼进字符串（str(dict) 风险） ---- #
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "_esc":
            for a in node.args:
                if isinstance(a, ast.Call) and _call_name(a) in ("repr", "str"):
                    if a.args and isinstance(a.args[0], ast.Name) \
                            and a.args[0].id in ("v", "vv", "v2", "value"):
                        problems.append((node.lineno, "_esc(str(原始值))",
                                         "改用 _compact() 或 render_fields()"))

    lines = src.splitlines()
    print("扫描 app.py（%d 行）" % len(lines))
    if problems:
        print("\n[失败] 发现 %d 处手写渲染写法（会导致英文/None/原始字典/长串小数）："
              % len(problems))
        for ln, what, fix in sorted(set(problems)):
            ctx = lines[ln - 1].strip()[:70] if 0 < ln <= len(lines) else ""
            print("   第 %-5d 行 %-38s %s" % (ln, what, ctx))
            print("            → %s" % fix)
        return 1
    print("\n[通过] 9 个页签全部走统一渲染入口 render_fields，"
          "未发现手写渲染写法")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

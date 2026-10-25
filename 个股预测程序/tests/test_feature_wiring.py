# -*- coding: utf-8 -*-
"""
功能完整性校验 —— 防止"功能代码静默丢失"。

真实事故
--------
在"场内基金"那次提交里，core/fund.py 与 engine 接入都进了库，
但 app.py 的界面部分因为用 PowerShell here-string 做替换时**静默失败**，
导致 `render_fund` 从未真正写入。用户以为"基金功能没做"，
实际上是我把已完成的界面代码弄丢了，而且没有任何测试能发现。

这类问题的特征是：**模块都在、测试都过、但用户看不到功能**。
所以本测试直接检查"关键功能的界面接入点是否齐备"，
任何一处缺失就报错——它是给人看的功能清单，不只是给代码看的。

用法：``python tests/test_feature_wiring.py``
"""

from __future__ import annotations

import ast
import io
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, "app.py")

#: (名称, 检查函数) —— 每一项都对应一个用户能感知的功能
def _read() -> str:
    return io.open(APP, encoding="utf-8").read()


def check_fund_ui(s: str) -> list:
    """场内基金：界面必须真的渲染出来。"""
    miss = []
    if "def render_fund(" not in s:
        miss.append("缺少 render_fund() 函数")
    if not s.count("render_fund(p)") >= 1:
        miss.append("render_fund 从未被调用（函数存在但页面不显示）")
    if '"is_fund": bool' not in s:
        miss.append("payload 未带 is_fund 标志")
    if "is_fund = bool(p.get" not in s:
        miss.append("render_dimensions 未读取 is_fund")
    if "基金特性/类型" not in s:
        miss.append("页签标题未按基金切换")
    if "if val and not is_fund" not in s:
        miss.append("基金未跳过个股基本面（会给用户看一堆空 PE/PB）")
    return miss


def check_market_ui(s: str) -> list:
    """大盘三层结构 + 板块：必须是**顶层可见面板**，不能埋在子页签里。"""
    miss = []
    # 拆成两个独立面板（大盘 / 板块），并都在主界面被调用
    if "def render_market_panel(" not in s:
        miss.append("缺少 render_market_panel() 函数")
    if "def render_board_panel(" not in s:
        miss.append("缺少 render_board_panel() 函数")
    if "render_market_panel(payload)" not in s:
        miss.append("大盘面板未在主界面调用（用户会看不到）")
    if "render_board_panel(payload)" not in s:
        miss.append("板块面板未在主界面调用（用户会看不到）")
    # 四面板切换器必须存在
    if "选择分析范围" not in s:
        miss.append("缺少分析范围选择器（初始界面应能选综合/大盘/板块/个股基金）")
    for kw in ("大周期", "中级波段", "短线结构"):
        if kw not in s:
            miss.append("缺少「%s」展示" % kw)
    if "操作技巧" not in s:
        miss.append("缺少操作技巧提示区块")
    if "板块" not in s:
        miss.append("缺少板块情况展示")
    return miss


def check_fund_payload(s: str) -> list:
    """基金/大盘数据必须进 payload，否则前端拿不到。"""
    miss = []
    if '"fund": _deep_clean' not in s:
        miss.append("payload 未带 fund 数据")
    if '"market": _deep_clean' not in s:
        miss.append("payload 未带 market 数据（大盘结构拿不到）")
    return miss


def check_learning_ui(s: str) -> list:
    miss = []
    if "def render_self_learning(" not in s:
        miss.append("缺少自我学习面板函数")
    if "render_self_learning(" not in s.replace("def render_self_learning(", ""):
        miss.append("自我学习面板未被调用")
    return miss


def check_render_contract(s: str) -> list:
    """统一渲染入口必须仍在（防止有人改回手写渲染）。"""
    miss = []
    if "def render_fields(" not in s:
        miss.append("缺少统一渲染入口 render_fields()")
    if "def _fmt_value(" not in s:
        miss.append("缺少数字格式化 _fmt_value()")
    return miss


CHECKS = [
    ("场内基金界面", check_fund_ui),
    ("大盘三层界面", check_market_ui),
    ("数据进 payload", check_fund_payload),
    ("自我学习界面", check_learning_ui),
    ("统一渲染契约", check_render_contract),
]


def main() -> int:
    s = _read()
    try:
        ast.parse(s)
    except SyntaxError as exc:
        print("[失败] app.py 语法错误：第 %s 行 %s" % (exc.lineno, exc.msg))
        return 1
    # 客户端代码里也要有对应函数（否则函数写了但没挂到页面上）
    tree = ast.parse(s)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    print("app.py：%d 行，%d 个函数" % (s.count("\n") + 1, len(funcs)))
    bad = 0
    for name, fn in CHECKS:
        miss = fn(s)
        if miss:
            bad += 1
            print("\n[失败] %s 不完整：" % name)
            for m in miss:
                print("   · %s" % m)
        else:
            print("  [OK] %s" % name)
    if bad:
        print("\n共 %d 组功能不完整。这类问题用户能感知、但普通单测发现不了。"
              % bad)
        return 1
    print("\n[通过] 所有关键功能的界面接入点齐备（无静默丢失）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

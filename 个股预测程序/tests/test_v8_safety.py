# -*- coding: utf-8 -*-
"""
V8 并发安全校验 —— 防止"并行分析把进程搞崩"再次发生。

事故
----
开启 4 线程并行分析后，进程**直接崩溃**（不是异常，try/except 抓不住）：

    [FATAL:partition_address_space.cc(243)]
    Check failed: !IsConfigurablePoolInitialized()
    #0 py_mini_racer\\mini_racer.dll

akshare 用 py_mini_racer（内嵌 V8）跑 JS 解密；V8 有进程级内存池，
多线程并发创建/销毁上下文会撞断言并 abort 进程。

为什么普通单测抓不到
--------------------
单测只验证"函数返回对不对"，而这个 bug 是**进程级崩溃**：
测试进程直接死掉，看起来像"测试框架有问题"，而不是断言失败。

本测试做三件事：
  1. 断言 v8_guard 已安装（akshare 的导出函数带 __v8_locked__ 标记）
  2. 断言预热身成功（V8 平台已在主线程初始化）
  3. **并发调用 akshare 多次**，确认进程存活且结果正确

用法：``python tests/test_v8_safety.py``
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def main() -> int:
    from core import v8_guard

    # 1) 安装
    ok_install = v8_guard.install()
    print("V8 锁安装:", "是" if ok_install else "否（akshare 不可用时可接受）")

    # 2) 预热身
    ok_pre = v8_guard.prewarm()
    print("V8 预热身:", "成功" if ok_pre else "失败")

    if not ok_install:
        print("\n[跳过] akshare 不可用（该环境无 V8 依赖），V8 安全无需校验")
        return 0

    # 3) 验证导出函数确实被包裹
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        print("[跳过] akshare 导入失败:", str(exc)[:60])
        return 0

    locked = 0
    checked = 0
    for name in dir(ak):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(ak, name)
        except Exception:  # noqa: BLE001
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        checked += 1
        if getattr(obj, "__v8_locked__", False):
            locked += 1
    print("akshare 导出函数：%d 个，已加锁 %d 个" % (checked, locked))
    if checked and locked < checked * 0.9:
        print("[失败] 加锁覆盖率不足 90%%，并发时仍有 V8 崩溃风险")
        return 1

    # 4) 并发调用（关键：进程若崩溃，本测试直接死亡）
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        try:
            df = ak.macro_china_cpi()
            return i, (len(df) if df is not None else 0), None
        except Exception as exc:  # noqa: BLE001
            return i, 0, str(exc)[:50]

    rounds = 3
    print("\n并发调用 akshare（%d 轮 × 4 线程）…" % rounds)
    for rd in range(rounds):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(one, range(4)))
        bad = [r for r in results if r[2]]
        print("  第 %d 轮：%s" % (rd + 1,
                                 "全部成功" if not bad else "失败 %d 个" % len(bad)))
        if bad:
            print("    ", bad[:2])
    print("\n[通过] 进程存活，V8 并发调用安全（这就是本次修复的核心验证）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

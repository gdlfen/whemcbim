# -*- coding: utf-8 -*-
"""
V8 引擎并发保护 —— 修掉并行分析导致的进程级崩溃。

事故现象
--------
开启并行分析后，进程**直接崩溃**（不是 Python 异常，try/except 抓不住）：

    [FATAL:partition_address_space.cc(243)]
    Check failed: !IsConfigurablePoolInitialized()
    #0 py_mini_racer\\mini_racer.dll

原因
----
akshare 的部分接口用 `py_mini_racer`（内嵌 V8 引擎）执行 JS 解密。
V8 有自己的**进程级内存池**，在多线程里并发初始化/销毁大量 MiniRacer 上下文时，
会撞上 `IsConfigurablePoolInitialized` 断言而 abort 整个进程。
这是 V8 的已知限制：它假设自己由单一线程主导。

为什么单元测试抓不到
--------------------
单独用线程池并发创建 4 个 MiniRacer 是**能过**的；
只有 akshare 在十几秒内反复创建/销毁几十个上下文时才触发。
所以必须从"串行化 V8 使用"这个层面解决，而不是减少线程数。

解法
----
1. **预热身**：进程启动时在主线程创建一次 MiniRacer，让 V8 平台完成初始化。
   之后各线程只复用已初始化的平台，不再竞争首次初始化。
2. **全局锁**：给 akshare 的每个导出函数套一把可重入锁，
   保证任何时刻只有一个线程在跑 JS。requests 类请求不受影响，
   因此并行提速基本保留（V8 只占各步骤的一小部分时间）。
"""
from __future__ import annotations

import sys
import threading

from .utils import get_logger

log = get_logger()

#: 保护所有 akshare（内含 V8）调用的全局可重入锁
V8_LOCK = threading.RLock()

_installed = False
_prewarmed = False


def prewarm() -> bool:
    """在主线程预初始化 V8（只做一次），避免多线程竞争首次初始化。"""
    global _prewarmed
    if _prewarmed:
        return True
    with V8_LOCK:
        try:
            import py_mini_racer as _mr

            # 不同版本导出位置不同，逐个尝试
            engine = getattr(_mr, "MiniRacer", None)
            if engine is None:
                from py_mini_racer.py_mini_racer import MiniRacer as engine  # type: ignore
            ctx = engine()
            ctx.eval("1+1")
            del ctx
            _prewarmed = True
            log.debug("V8 预热身完成")
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("V8 预热身失败（不影响无 JS 的接口）: %s", exc)
            return False


def _wrap(fn):
    """把函数包成"持锁执行"。"""

    def wrapper(*args, **kwargs):
        with V8_LOCK:
            return fn(*args, **kwargs)

    try:
        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        wrapper.__module__ = getattr(fn, "__module__", None)
    except Exception:  # noqa: BLE001
        pass
    return wrapper


def install(ak=None) -> bool:
    """给 akshare 的所有导出函数套上 V8 锁；并预热身。

    幂等：重复调用无副作用。
    """
    global _installed
    if _installed:
        return True
    prewarm()
    if ak is None:
        ak = sys.modules.get("akshare")
        if ak is None:
            try:
                import akshare as ak  # noqa: PLC0415
            except Exception as exc:  # noqa: BLE001
                log.debug("akshare 不可用，跳过 V8 锁: %s", exc)
                _installed = True
                return False
    wrapped = 0
    try:
        names = [n for n in dir(ak) if not n.startswith("_")]
    except Exception:  # noqa: BLE001
        names = []
    for name in names:
        try:
            obj = getattr(ak, name)
        except Exception:  # noqa: BLE001
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        if getattr(obj, "__v8_locked__", False):
            continue
        try:
            w = _wrap(obj)
            w.__v8_locked__ = True          # 标记，避免重复包裹
            setattr(ak, name, w)
            wrapped += 1
        except Exception:  # noqa: BLE001
            continue
    _installed = True
    log.debug("akshare 已加 V8 锁：%d 个函数", wrapped)
    return wrapped > 0


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    print("=== 1. 预热身 ===")
    print("  prewarm:", prewarm())
    print("=== 2. 安装锁 ===")
    print("  install:", install())
    print("=== 3. 并发调用 akshare（应当串行、不崩溃）===")
    from concurrent.futures import ThreadPoolExecutor

    import akshare as ak

    def one(i):
        try:
            df = ak.macro_china_cpi()
            return i, ("ok %d 行" % len(df)) if df is not None else "ok 空"
        except Exception as exc:  # noqa: BLE001
            return i, "ERR %s" % str(exc)[:60]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(one, range(4)):
            print("  ", r)
    print("ALIVE —— 未崩溃")
    print("\nv8_guard self-test OK")

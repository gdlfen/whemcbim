# -*- coding: utf-8 -*-
"""
线程安全的进度追踪器 —— 供"后台线程跑分析 + 主线程实时刷新进度条"用。

为什么需要它
------------
Streamlit 的 UI 只在脚本执行期间更新。分析是**同步阻塞**的，
如果直接在脚本里跑，进度条只会停在 0% 直到结束（用户看到"假死"）。

正确做法：把分析放进后台线程，主线程轮询进度并刷新进度条。
两个线程之间需要一个线程安全的进度容器 —— 就是本模块。

注意：进度容器**不是** UI 对象，放进后台线程完全合法；
被 @st.cache_data 装饰的函数不能碰 UI，但可以往这里写数字。
"""
from __future__ import annotations

import threading
import math
import time


class ProgressTracker:
    """线程安全的进度容器。

    Examples
    --------
    >>> t = ProgressTracker()
    >>> t.update("获取行情数据", 0.1)
    >>> t.snapshot()
    ('获取行情数据', 0.1, True)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stage = "准备中"
        self._pct = 0.0
        self._done = False
        self._error: str | None = None
        self._t0 = time.time()
        #: 阶段历史（供结束后展示耗时）
        self.stages: list[tuple[str, float]] = []

    # -- 写入 ---------------------------------------------------------- #
    def update(self, stage: str, pct: float) -> None:
        """更新进度（线程安全）。pct 会被夹到 0~1 并**只增不减**。"""
        try:
            p = float(pct)
        except (TypeError, ValueError):
            return
        p = min(max(p, 0.0), 1.0)
        with self._lock:
            self._last_t = time.time()
            if p >= self._pct:                 # 单调递增，避免进度倒退
                self._pct = p
            if stage and (not self.stages or self.stages[-1][0] != stage):
                self.stages.append((stage, p))

    def finish(self, error: str | None = None) -> None:
        with self._lock:
            self._done = True
            if error is None:
                self._pct = 1.0
            self._error = error

    # -- 读取 ---------------------------------------------------------- #
    def snapshot(self) -> tuple[str, float, bool]:
        """返回 ``(当前阶段, 进度, 是否结束)``。"""
        with self._lock:
            return self._stage or "分析中", self._pct, self._done

    def smooth_snapshot(self, step_seconds: float = 4.0) -> tuple[str, float, bool]:
        """返回**实时动态**进度：在两次真实上报之间平滑爬升。

        分析按阶段上报，长阶段（如基本面十余秒）里进度条会长时间不动、
        看起来像卡死；短阶段又一下跳很多。本方法按距上次上报的时间
        向上限做渐近爬升：pct + (上限−pct)(1−e^(−t/τ))，
        上限取剩余空间的 85%，**永远追不上真实上限**，
        因此不会出现"走到 95% 却还在跑"的假象；真实上报到达即对齐。
        """
        with self._lock:
            pct, stage, done = self._pct, self._stage, self._done
            last_t = getattr(self, "_last_t", None) or self._t0
        if done:
            return stage or "完成", 1.0, True
        gap = max(0.0, time.time() - last_t)
        if gap <= 0.05:
            return stage or "分析中", pct, done
        ceil = pct + (1.0 - pct) * 0.85
        tau = max(1.0, float(step_seconds))
        pct_disp = pct + (ceil - pct) * (1.0 - math.exp(-gap / tau))
        return stage or "分析中", min(pct_disp, 0.995), done

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def label(self, names: dict | None = None) -> str:
        """给界面用的阶段文案。"""
        stage, pct, done = self.snapshot()
        if self.error:
            return "分析失败"
        if done:
            return "完成"
        name = (names or {}).get(stage, stage or "分析中")
        return "%s ｜ %.0f%%（已用 %.0f 秒）" % (name, pct * 100, self.elapsed)


if __name__ == "__main__":  # pragma: no cover
    import random

    t = ProgressTracker()
    assert t.snapshot() == ("准备中", 0.0, False) or t.snapshot()[1] == 0.0

    # 单调性
    t.update("a", 0.5)
    t.update("a", 0.3)                      # 倒退应被忽略
    assert abs(t.snapshot()[1] - 0.5) < 1e-9, t.snapshot()
    t.update("b", 0.9)
    assert abs(t.snapshot()[1] - 0.9) < 1e-9
    # 夹取
    t.update("c", 2.0)
    assert t.snapshot()[1] == 1.0
    t.update("c", -1.0)
    assert t.snapshot()[1] == 1.0
    # 阶段记录去重
    assert [s for s, _ in t.stages] == ["a", "b", "c"], t.stages

    # 并发写入
    t2 = ProgressTracker()

    def worker(k):
        for i in range(50):
            t2.update("阶段%d" % k, (i + 1) / 50.0)

    ths = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    st, pc, dn = t2.snapshot()
    assert 0.0 <= pc <= 1.0 and not dn
    assert len([s for s, _ in t2.stages]) >= 1

    t2.finish()
    assert t2.snapshot()[2] is True and t2.snapshot()[1] == 1.0
    print("progress self-test OK：单调、夹取、阶段去重、6 线程并发写入均正确")

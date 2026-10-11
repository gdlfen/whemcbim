# -*- coding: utf-8 -*-
"""
可视化模块 —— 生成 K 线与指标图表（matplotlib）。

所有函数遵循同一约定：
* 输入 ``df`` 为日线 DataFrame（含 date/open/high/low/close/volume）；
* 返回 ``matplotlib.figure.Figure``；**失败时返回 None**，绝不抛异常；
* 不调用 ``plt.show()``，由调用方决定渲染方式（保存文件 / ``st.pyplot``）；
* 中文字体在 Windows 与 Linux 上都做尝试，找不到则退回英文标签，
  但**不会因为字体缺失而报错**（这是云端部署最常见的踩坑点）。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger()

_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
    "WenQuanYi Zen Hei", "Source Han Sans SC", "PingFang SC", "Arial Unicode MS",
]
_FONT_READY: bool | None = None


def _font_has_cjk(font_name: str) -> bool:
    """检查某个字体是否真的包含中文字形。

    只检查"字体名在列表里"是不够的——Linux 上常见的情况是字体被 fontconfig
    提供一个不带中文字形的替身，结果图里全是方框（tofu）。这里直接读字体的
    cmap，确认它是否覆盖常用汉字区（U+4E00 起）。
    """
    try:
        from matplotlib import font_manager

        path = font_manager.findfont(font_manager.FontProperties(family=font_name),
                                     fallback_to_default=False)
        if not path or not os.path.exists(path):
            return False
        try:
            from fontTools.ttLib import TTFont  # 可选依赖

            with TTFont(path, fontNumber=0, lazy=True) as ft:
                for table in ft["cmap"].tables:
                    if 0x4E2D in table.cmap and 0x6587 in table.cmap:  # 中文
                        return True
            return False
        except Exception:  # noqa: BLE001
            # 没有 fontTools 时退化为"文件名启发式"判断
            low = os.path.basename(path).lower()
            return any(k in low for k in ("cjk", "hei", "song", "ming", "yahei",
                                          "noto", "wqy", "droidsansfallback"))
    except Exception:  # noqa: BLE001
        return False


#: 随仓库分发的字体（子集化，0.27MB，含 ASCII + 源码里出现的全部汉字/符号）。
#: 云端 Linux 常常"看起来装了中文字体、实际渲染成方框"，所以自带一份最可靠。
_BUNDLED_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                             "NotoSansSC-subset.otf")


def register_bundled_font() -> str | None:
    """把随仓库分发的字体注册进 matplotlib，返回其字体名（失败返回 None）。"""
    try:
        if not os.path.exists(_BUNDLED_FONT):
            return None
        from matplotlib import font_manager

        font_manager.fontManager.addfont(_BUNDLED_FONT)
        name = font_manager.FontProperties(fname=_BUNDLED_FONT).get_name()
        return name or None
    except Exception as exc:  # noqa: BLE001
        log.debug("注册内置字体失败: %s", exc)
        return None


def setup_chinese_font() -> bool:
    """配置 matplotlib 中文字体；返回是否成功找到**确实含中文字形**的字体。

    优先级：**随仓库分发的子集字体** > 系统里经验证含汉字的字体 > 英文回退。
    每个候选字体都会读 cmap 校验；都不行时返回 False（调用方改用英文标签），
    避免在图上画出一堆方框。
    """
    global _FONT_READY
    if _FONT_READY is not None:
        return _FONT_READY
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        from matplotlib import font_manager

        picked: list[str] = []
        bundled = register_bundled_font()
        if bundled and _font_has_cjk(bundled):
            picked.append(bundled)

        available = {f.name for f in font_manager.fontManager.ttflist}
        candidates = [f for f in _FONT_CANDIDATES if f in available]
        if not candidates:
            for f in font_manager.fontManager.ttflist:
                nm = f.name or ""
                if any(k in nm for k in ("CJK", "Noto Sans SC", "Source Han", "WenQuanYi",
                                         "Hei", "Song", "Ming", "YaHei", "PingFang",
                                         "Droid Sans Fallback")):
                    candidates.append(nm)
        for c in dict.fromkeys(candidates):
            if c not in picked and _font_has_cjk(c):
                picked.append(c)

        if picked:
            # DejaVu Sans 兜底：内置子集字体已含 ASCII，但再加一层保险
            matplotlib.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            log.info("图表中文字体：%s", picked[0])
            _FONT_READY = True
        else:
            log.info("未找到含中文字形的字体，图表标签改用英文（数值不受影响）")
            _FONT_READY = False
    except Exception as exc:  # noqa: BLE001
        log.debug("字体配置失败: %s", exc)
        _FONT_READY = False
    return bool(_FONT_READY)


def _label(zh: str, en: str) -> str:
    return zh if setup_chinese_font() else en


def _ensure_df(df) -> pd.DataFrame | None:
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    d = df.copy()
    ren = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
           "收盘": "close", "成交量": "volume", "成交额": "amount", "换手率": "turnover"}
    d = d.rename(columns={k: v for k, v in ren.items() if k in d.columns})
    for col in ("open", "high", "low", "close"):
        if col not in d.columns:
            return None
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    return d


def _tail(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    return df.iloc[-int(bars):].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# K 线主图 + 均线 + 布林 + 成交量
# --------------------------------------------------------------------------- #
def plot_price(df, *, bars: int = 180, title: str = "", mas=(5, 20, 60),
               show_boll: bool = True, show_volume: bool = True,
               marks: list | None = None, support_resistance=None):
    """绘制 K 线主图：蜡烛 + 均线 + 布林带 + 成交量，可选标注支撑压力位。"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        d = _ensure_df(df)
        if d is None or len(d) < 5:
            return None
        d = _tail(d, bars)
        n = len(d)
        x = np.arange(n)

        rows = 2 if show_volume else 1
        fig, axes = plt.subplots(rows, 1, figsize=(12, 6.4 if show_volume else 5.0),
                                 sharex=True,
                                 gridspec_kw={"height_ratios": [3.2, 1.0]} if show_volume else None)
        ax = axes[0] if show_volume else axes

        # ---- 蜡烛 ---------------------------------------------------- #
        width = 0.62
        for i in range(n):
            o, h, l, c = (float(d["open"].iloc[i]), float(d["high"].iloc[i]),
                          float(d["low"].iloc[i]), float(d["close"].iloc[i]))
            color = "#d62728" if c >= o else "#2ca02c"   # A股习惯：红涨绿跌
            ax.vlines(i, l, h, color=color, linewidth=0.7, zorder=2)
            ax.add_patch(Rectangle((i - width / 2, min(o, c)), width, max(abs(c - o), 1e-6),
                                   facecolor=color if c >= o else color,
                                   edgecolor=color, linewidth=0.5, zorder=3))

        # ---- 均线 ---------------------------------------------------- #
        close_full = _ensure_df(df)["close"].astype(float)
        for ma in mas:
            if len(close_full) >= ma:
                series = close_full.rolling(ma).mean().iloc[-n:].to_numpy()
                ax.plot(x, series, linewidth=1.0, label="MA%d" % ma, alpha=0.9)
        # ---- 布林带 -------------------------------------------------- #
        if show_boll and len(close_full) >= 20:
            mid = close_full.rolling(20).mean()
            sd = close_full.rolling(20).std()
            up = (mid + 2 * sd).iloc[-n:].to_numpy()
            low = (mid - 2 * sd).iloc[-n:].to_numpy()
            ax.plot(x, up, linestyle="--", linewidth=0.8, color="#888", alpha=0.8,
                    label=_label("布林上轨", "BOLL UP"))
            ax.plot(x, low, linestyle="--", linewidth=0.8, color="#888", alpha=0.8,
                    label=_label("布林下轨", "BOLL LOW"))
            ax.fill_between(x, low, up, color="#888", alpha=0.06)

        # ---- 支撑压力位 ---------------------------------------------- #
        if isinstance(support_resistance, dict):
            for p in (support_resistance.get("supports") or [])[:3]:
                pv = p.get("price") if isinstance(p, dict) else p
                if pv:
                    ax.axhline(float(pv), color="#1f77b4", linestyle=":", linewidth=0.8,
                               alpha=0.75)
            for p in (support_resistance.get("resistances") or [])[:3]:
                pv = p.get("price") if isinstance(p, dict) else p
                if pv:
                    ax.axhline(float(pv), color="#ff7f0e", linestyle=":", linewidth=0.8,
                               alpha=0.75)

        # ---- 标注（形态/事件） --------------------------------------- #
        for mk in (marks or [])[:40]:
            try:
                idx = mk.get("index")
                if idx is None or not (0 <= idx < n):
                    continue
                price = float(d["high"].iloc[idx]) if mk.get("direction", 1) < 0 \
                    else float(d["low"].iloc[idx])
                ax.annotate(str(mk.get("text", ""))[:6], (idx, price),
                            textcoords="offset points",
                            xytext=(0, 12 if mk.get("direction", 1) < 0 else -18),
                            fontsize=7, ha="center",
                            color="#d62728" if mk.get("direction", 1) > 0 else "#2ca02c")
            except Exception:  # noqa: BLE001
                continue

        step = max(n // 8, 1)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(d["date"].iloc[i])[:10] for i in range(0, n, step)],
                           rotation=25, fontsize=8)
        ax.set_title(title or _label("K线走势", "Price"), fontsize=12)
        ax.grid(alpha=0.18)
        ax.legend(loc="upper left", fontsize=8, ncol=4, framealpha=0.85)
        ax.set_ylabel(_label("价格", "Price"))

        # ---- 成交量 -------------------------------------------------- #
        if show_volume and "volume" in d.columns:
            ax2 = axes[1]
            vol = d["volume"].astype(float).to_numpy()
            colors = ["#d62728" if float(d["close"].iloc[i]) >= float(d["open"].iloc[i])
                      else "#2ca02c" for i in range(n)]
            ax2.bar(x, vol, color=colors, width=0.7, alpha=0.75)
            if len(vol) >= 5:
                ax2.plot(x, pd.Series(vol).rolling(5).mean().to_numpy(), linewidth=0.9,
                         color="#555", label="MA5")
            ax2.set_ylabel(_label("成交量", "Volume"), fontsize=9)
            ax2.grid(alpha=0.18)
            ax2.legend(loc="upper left", fontsize=7)
        fig.tight_layout()
        return fig
    except Exception as exc:  # noqa: BLE001
        log.warning("plot_price 失败: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# 指标副图
# --------------------------------------------------------------------------- #
def plot_indicators(df, *, bars: int = 180, panels=("MACD", "KDJ", "RSI")):
    """绘制 MACD / KDJ / RSI 副图。"""
    try:
        import matplotlib.pyplot as plt

        d = _ensure_df(df)
        if d is None or len(d) < 40:
            return None
        full = d
        d = _tail(d, bars)
        n = len(d)
        x = np.arange(n)
        c = full["close"].astype(float)

        def _tailn(s):
            return np.asarray(s)[-n:]

        panels = [p for p in panels if p]
        if not panels:
            return None
        fig, axes = plt.subplots(len(panels), 1, figsize=(12, 1.9 * len(panels)),
                                 sharex=True)
        if len(panels) == 1:
            axes = [axes]

        for ax, name in zip(axes, panels):
            if name == "MACD":
                e12 = c.ewm(span=12, adjust=False).mean()
                e26 = c.ewm(span=26, adjust=False).mean()
                dif = e12 - e26
                dea = dif.ewm(span=9, adjust=False).mean()
                hist = (dif - dea) * 2
                ax.bar(x, _tailn(hist), color=["#d62728" if v >= 0 else "#2ca02c"
                                               for v in _tailn(hist)], width=0.7, alpha=0.7)
                ax.plot(x, _tailn(dif), linewidth=0.9, label="DIF")
                ax.plot(x, _tailn(dea), linewidth=0.9, label="DEA")
                ax.axhline(0, color="#999", linewidth=0.6)
                ax.legend(loc="upper left", fontsize=7, ncol=3)
                ax.set_title("MACD", fontsize=9)
            elif name == "KDJ":
                h, l = full["high"].astype(float), full["low"].astype(float)
                llv = l.rolling(9).min()
                hhv = h.rolling(9).max()
                rsv = (c - llv) / (hhv - llv).replace(0, np.nan) * 100
                k = rsv.ewm(alpha=1 / 3.0, adjust=False).mean()
                dd = k.ewm(alpha=1 / 3.0, adjust=False).mean()
                j = 3 * k - 2 * dd
                ax.plot(x, _tailn(k), linewidth=0.9, label="K")
                ax.plot(x, _tailn(dd), linewidth=0.9, label="D")
                ax.plot(x, _tailn(j), linewidth=0.9, label="J", alpha=0.7)
                ax.axhline(80, color="#d62728", linestyle=":", linewidth=0.7)
                ax.axhline(20, color="#2ca02c", linestyle=":", linewidth=0.7)
                ax.legend(loc="upper left", fontsize=7, ncol=3)
                ax.set_title("KDJ", fontsize=9)
            elif name == "RSI":
                for per in (6, 14):
                    diff = c.diff()
                    up = diff.clip(lower=0).ewm(alpha=1 / per, adjust=False).mean()
                    dn = (-diff.clip(upper=0)).ewm(alpha=1 / per, adjust=False).mean()
                    rsi = 100 * up / (up + dn).replace(0, np.nan)
                    ax.plot(x, _tailn(rsi), linewidth=0.9, label="RSI%d" % per)
                ax.axhline(70, color="#d62728", linestyle=":", linewidth=0.7)
                ax.axhline(30, color="#2ca02c", linestyle=":", linewidth=0.7)
                ax.legend(loc="upper left", fontsize=7, ncol=2)
                ax.set_title("RSI", fontsize=9)
            ax.grid(alpha=0.18)

        step = max(n // 8, 1)
        axes[-1].set_xticks(x[::step])
        axes[-1].set_xticklabels([str(d["date"].iloc[i])[:10] for i in range(0, n, step)],
                                 rotation=25, fontsize=8)
        fig.tight_layout()
        return fig
    except Exception as exc:  # noqa: BLE001
        log.warning("plot_indicators 失败: %s", exc)
        return None


def plot_multi_horizon(result):
    """把 8 个周期的概率与期望幅度画成一张组合图（横向条形）。"""
    try:
        import matplotlib.pyplot as plt

        fcs = getattr(result, "forecasts", None) or []
        if not fcs:
            return None
        labels = [f.label for f in fcs]
        p_up = [f.p_up * 100 for f in fcs]
        exp = [f.exp_return for f in fcs]
        y = np.arange(len(fcs))

        fig, axes = plt.subplots(1, 2, figsize=(12, 0.52 * len(fcs) + 1.6))
        ax1, ax2 = axes
        n_h = len(fcs)
        ylim = (n_h - 0.5, -0.5)          # 第一行在顶部；两栏用同一 ylim 保证逐行对齐

        colors = ["#d62728" if p >= 50 else "#2ca02c" for p in p_up]
        ax1.barh(y, p_up, color=colors, alpha=0.82, height=0.62)
        ax1.axvline(50, color="#333", linewidth=0.9, linestyle="--")
        for i, p in enumerate(p_up):
            ax1.text(p + (1.2 if p >= 50 else -1.2), i, "%.1f%%" % p,
                     va="center", ha="left" if p >= 50 else "right", fontsize=8)
        ax1.set_yticks(y)
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(*ylim)
        ax1.set_title(_label("各周期上涨概率", "P(up) by horizon"), fontsize=11)
        ax1.grid(alpha=0.18, axis="x")

        colors2 = ["#d62728" if e >= 0 else "#2ca02c" for e in exp]
        ax2.barh(y, exp, color=colors2, alpha=0.82, height=0.62)
        for i, e in enumerate(exp):
            ax2.text(e + (0.05 if e >= 0 else -0.05), i, "%+.2f%%" % e,
                     va="center", ha="left" if e >= 0 else "right", fontsize=8)
        ax2.axvline(0, color="#333", linewidth=0.9)
        ax2.set_yticks(y)
        ax2.set_yticklabels([])          # 与左栏共用行名，避免重复
        ax2.set_ylim(*ylim)
        span = max(max(abs(min(exp)), abs(max(exp))) * 1.35, 0.5)
        ax2.set_xlim(-span, span * 0.55)  # 多留左侧空间给负向标签
        ax2.set_title(_label("各周期期望涨跌幅", "Expected return by horizon"), fontsize=11)
        ax2.grid(alpha=0.18, axis="x")
        fig.tight_layout()
        return fig
    except Exception as exc:  # noqa: BLE001
        log.warning("plot_multi_horizon 失败: %s", exc)
        return None


def plot_confidence_cone(result):
    """画置信锥：随周期扩大的收益区间（带中心期望线）。"""
    try:
        import matplotlib.pyplot as plt

        fcs = getattr(result, "forecasts", None) or []
        if not fcs:
            return None
        x = np.array([f.bars for f in fcs], dtype=float)
        exp = np.array([f.exp_return for f in fcs], dtype=float)
        lo = np.array([f.ci_low for f in fcs], dtype=float)
        hi = np.array([f.ci_high for f in fcs], dtype=float)
        labels = [f.label.replace("未来", "") for f in fcs]
        # 用序号做等距横轴（周期的 bars 跨度差异太大，直接当数值轴会把短线挤在一起）
        xpos = np.arange(len(fcs), dtype=float)

        fig, ax = plt.subplots(figsize=(10, 5.0))
        ax.fill_between(xpos, lo, hi, color="#4c78a8", alpha=0.20,
                        label=_label("约 72% 概率区间", "~72% interval"))
        ax.plot(xpos, exp, "o-", color="#1f4e79", linewidth=1.6, markersize=5,
                label=_label("期望涨跌幅", "Expected"))
        ax.axhline(0, color="#333", linewidth=1.0, linestyle="--")
        for xi, l, h_, e in zip(xpos, lo, hi, exp):
            ax.vlines(xi, l, h_, color="#4c78a8", linewidth=0.8, alpha=0.6)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=8, rotation=20)
        ax.set_xlim(-0.4, len(fcs) - 0.6)
        ax.set_ylabel(_label("预期涨跌幅 (%)", "Expected return (%)"))
        ax.set_title(_label("预测置信锥（周期越长不确定性越大）",
                            "Confidence cone"), fontsize=12)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig
    except Exception as exc:  # noqa: BLE001
        log.warning("plot_confidence_cone 失败: %s", exc)
        return None


def plot_factor_contribution(result, horizon: str = "d20", top: int = 14):
    """画单周期的因子贡献条形图（红=看涨贡献，绿=看跌贡献）。"""
    try:
        import matplotlib.pyplot as plt

        fc = result.by_key(horizon) if hasattr(result, "by_key") else None
        if fc is None:
            return None
        rows = list(fc.drivers) + list(fc.counter)
        if not rows:
            return None
        rows = sorted(rows, key=lambda r: -abs(r.get("贡献", 0)))[:top]
        names = [r["因子"] for r in rows][::-1]
        vals = [r.get("贡献", 0) for r in rows][::-1]
        fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 1.6))
        ax.barh(np.arange(len(vals)), vals,
                color=["#d62728" if v > 0 else "#2ca02c" for v in vals], alpha=0.85)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color="#333", linewidth=0.9)
        ax.set_xlabel(_label("贡献度（%）", "Contribution (%)"))
        ax.set_title(_label("%s 因子贡献" % fc.label, "Factor contribution"), fontsize=11)
        ax.grid(alpha=0.18, axis="x")
        fig.tight_layout()
        return fig
    except Exception as exc:  # noqa: BLE001
        log.warning("plot_factor_contribution 失败: %s", exc)
        return None


def plot_probability_gauge(result):
    """把各周期上涨概率画成横向"仪表条"。"""
    try:
        import matplotlib.pyplot as plt

        fcs = getattr(result, "forecasts", None) or []
        if not fcs:
            return None
        fig, ax = plt.subplots(figsize=(10, 0.46 * len(fcs) + 1.2))
        for i, f in enumerate(fcs):
            ax.barh(i, 100, color="#eeeeee", height=0.55)
            ax.barh(i, f.p_up * 100,
                    color="#d62728" if f.p_up >= 0.5 else "#2ca02c", height=0.55, alpha=0.85)
            ax.text(1.5, i, f.label, va="center", fontsize=9)
            ax.text(98.5, i, "%.1f%%" % (f.p_up * 100), va="center", ha="right",
                    fontsize=9, color="#222")
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_title(_label("上涨概率仪表（红=偏多 绿=偏空）", "P(up) gauge"), fontsize=11)
        ax.grid(alpha=0.12, axis="x")
        fig.tight_layout()
        return fig
    except Exception as exc:  # noqa: BLE001
        log.warning("plot_probability_gauge 失败: %s", exc)
        return None


def plot_backtest(bt_table, grade: dict | None = None):
    """画回测信息系数（IC）柱状图。"""
    try:
        import matplotlib.pyplot as plt

        if bt_table is None or (hasattr(bt_table, "empty") and bt_table.empty):
            return None
        df = bt_table.copy()
        ic_col = "IC" if "IC" in df.columns else None
        if ic_col is None:
            return None
        vals = pd.to_numeric(df[ic_col], errors="coerce").fillna(0.0).to_numpy()
        labels = df["周期"].astype(str).to_numpy()
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.bar(np.arange(len(vals)), vals,
               color=["#d62728" if v > 0 else "#2ca02c" for v in vals], alpha=0.85)
        ax.axhline(0, color="#333", linewidth=0.9)
        ax.axhline(0.03, color="#1f77b4", linestyle=":", linewidth=0.8)
        ax.axhline(-0.03, color="#1f77b4", linestyle=":", linewidth=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=18)
        ax.set_ylabel("IC")
        title = _label("历史回测：信号与未来收益的信息系数", "Backtest IC")
        if grade:
            title += "   [%s]" % grade.get("grade", "")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.18, axis="y")
        fig.tight_layout()
        return fig
    except Exception as exc:  # noqa: BLE001
        log.warning("plot_backtest 失败: %s", exc)
        return None


def save_figure(fig, path: str, dpi: int = 130) -> str | None:
    """保存图片到文件，返回路径（失败返回 None）。"""
    if fig is None:
        return None
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        return path
    except Exception as exc:  # noqa: BLE001
        log.warning("保存图表失败 %s: %s", path, exc)
        return None


if __name__ == "__main__":  # pragma: no cover
    import os

    from .contracts import AnalysisBundle, ProbabilityForecast, PredictionResult
    from .selftest_data import make_ohlcv

    print("中文字体可用:", setup_chinese_font())
    df = make_ohlcv(400, seed=5)
    figs = {}
    figs["price"] = plot_price(df, bars=120, title="测试K线",
                               support_resistance={"supports": [{"price": 28.0}],
                                                   "resistances": [{"price": 32.0}]})
    figs["ind"] = plot_indicators(df, bars=120)
    fcs = [ProbabilityForecast(horizon=h, label=l, p_up=p, exp_return=e,
                               ci_low=e - 4, ci_high=e + 4, confidence=0.5, sigma=4.0)
           for h, l, p, e in [("d1", "未来1天", 0.55, 0.3), ("d5", "未来5天", 0.58, 0.9),
                              ("d10", "未来10天", 0.61, 1.5), ("d20", "未来20天", 0.63, 2.4),
                              ("m1", "未来1个月", 0.62, 2.6), ("q1", "未来1个季度", 0.58, 4.0),
                              ("h1", "未来半年", 0.55, 6.5), ("y1", "未来1年", 0.53, 9.0)]]
    res = PredictionResult(symbol="600519", name="测试", price=30.0, forecasts=fcs)
    res.forecasts[3].drivers = [{"因子": "均线多头排列", "贡献": 12.5},
                                {"因子": "主力净流入", "贡献": 9.1}]
    res.forecasts[3].counter = [{"因子": "估值高位", "贡献": -7.3}]
    figs["multi"] = plot_multi_horizon(res)
    figs["cone"] = plot_confidence_cone(res)
    figs["contrib"] = plot_factor_contribution(res, "d20")
    figs["gauge"] = plot_probability_gauge(res)
    import pandas as pd
    figs["bt"] = plot_backtest(pd.DataFrame({"周期": ["1天", "5天", "20天", "1季"],
                                             "IC": [0.04, 0.06, -0.01, 0.03]}),
                               {"grade": "B（部分周期有效）"})
    # 输出目录放在工作区内（文件沙箱只允许写工作区）
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "_chart_selftest")
    os.makedirs(out_dir, exist_ok=True)
    ok = 0
    for k, f in figs.items():
        p = save_figure(f, os.path.join(out_dir, "%s.png" % k))
        print("  %-8s -> %s (%s)" % (k, "OK" if p else "None",
                                     "%.0f KB" % (os.path.getsize(p) / 1024) if p else "—"))
        ok += 1 if p else 0
        try:
            import matplotlib.pyplot as plt
            plt.close(f)
        except Exception:  # noqa: BLE001
            pass
    assert ok == len(figs), "有图表生成失败（%d/%d）" % (ok, len(figs))
    print("charts self-test OK（%d 张图，输出目录 %s）" % (ok, out_dir))

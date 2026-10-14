# -*- coding: utf-8 -*-
"""
本地命令行入口 —— 无需 Streamlit，直接在终端跑完整分析。

用法示例
--------
::

    python run_local.py 600519                     # 完整分析（含 AI 模型）+ 生成报告
    python run_local.py 600519 --no-ml             # 只用规则模型（快）
    python run_local.py 600519 --demo              # 演示模式（离线合成数据）
    python run_local.py 600519 000001 300750       # 批量分析
    python run_local.py --file stocks.txt          # 从文件读代码列表（每行一个）
    python run_local.py 600519 --backtest          # 额外输出历史回测与可信度评级
    python run_local.py 600519 --charts            # 额外保存图表 PNG
    python run_local.py --health                   # 数据源健康检查
    python run_local.py 600519 --json out.json     # 结果导出为 JSON

所有输出默认保存在 ``reports/`` 与 ``charts/`` 目录下。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd  # noqa: E402

from core import config  # noqa: E402
from core.backtest import (backtest_prediction_history, calibrate_confidence,  # noqa: E402
                           reliability_grade, summarize)
from core.engine import Predictor  # noqa: E402
from core.report import build_text, save_report  # noqa: E402
from core.utils import normalize_symbol  # noqa: E402


def _banner(title: str, width: int = 92) -> str:
    return "\n" + "=" * width + "\n" + title + "\n" + "=" * width


def analyze_one(symbol: str, args) -> dict:
    """分析单只股票，返回摘要 dict。"""
    t0 = time.time()
    print(_banner("分析 %s" % symbol))
    predictor = Predictor(symbol, use_ml=not args.no_ml, demo=args.demo,
                          days=args.days, verbose=not args.quiet)

    def _prog(stage: str, pct: float) -> None:
        if not args.quiet:
            print("  [%3.0f%%] %s" % (pct * 100, stage))

    try:
        result = predictor.run(progress=_prog)
    except Exception as exc:  # noqa: BLE001
        print("✗ 分析失败：%s" % exc)
        if args.verbose:
            traceback.print_exc()
        return {"symbol": symbol, "ok": False, "error": str(exc)}

    bundle = result.bundle
    df = getattr(bundle, "ohlcv", None)

    bt_table = None
    grade = None
    calib = None
    if args.backtest and df is not None and len(df):
        print("  执行历史回测…")
        try:
            bt = backtest_prediction_history(df, getattr(bundle, "index_data", None))
            bt_table = summarize(bt)
            grade = reliability_grade(bt)
            calib = calibrate_confidence(result.forecasts, bt)
        except Exception as exc:  # noqa: BLE001
            print("  ! 回测失败：%s" % exc)

    print()
    print(build_text(result, bt_table=bt_table, grade=grade))

    # ---- 导出 ---------------------------------------------------------- #
    paths: dict = {}
    if args.report:
        try:
            paths.update(save_report(result, out_dir=args.out_dir, bt_table=bt_table,
                                     grade=grade, calibration=calib,
                                     formats=tuple(args.formats)))
            print("\n报告已保存：")
            for k, v in paths.items():
                print("  [%s] %s" % (k, v))
        except Exception as exc:  # noqa: BLE001
            print("! 报告保存失败：%s" % exc)

    if args.charts and df is not None and len(df):
        try:
            from core import charts as ch
            from core import indicators as ind

            cdir = args.chart_dir or config.OUTPUT["chart_dir"]
            os.makedirs(cdir, exist_ok=True)
            sr = None
            try:
                sr = ind.support_resistance(df["high"], df["low"], df["close"],
                                            volume=df.get("volume"))
            except Exception:  # noqa: BLE001
                pass
            figs = {
                "kline": ch.plot_price(df, bars=200, title="%s K线" % (result.name or symbol),
                                       support_resistance=sr),
                "indicators": ch.plot_indicators(df, bars=200),
                "horizons": ch.plot_multi_horizon(result),
                "cone": ch.plot_confidence_cone(result),
                "contribution": ch.plot_factor_contribution(result, "d20"),
            }
            if bt_table is not None:
                figs["backtest"] = ch.plot_backtest(bt_table, grade)
            print("\n图表已保存：")
            for k, fig in figs.items():
                p = ch.save_figure(fig, os.path.join(cdir, "%s_%s.png" % (symbol, k)))
                if p:
                    print("  %s" % p)
                try:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            print("! 图表保存失败：%s" % exc)

    # ---- JSON ---------------------------------------------------------- #
    if args.json:
        try:
            payload = _to_json(result, bt_table, grade, calib)
            target = args.json
            if os.path.isdir(target) or len(args.symbols) > 1:
                os.makedirs(target if os.path.isdir(target) else ".", exist_ok=True)
                target = os.path.join(target, "%s.json" % symbol)
            with open(target, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
            print("\nJSON 已保存：%s" % target)
        except Exception as exc:  # noqa: BLE001
            print("! JSON 导出失败：%s" % exc)

    print("\n耗时 %.1fs" % (time.time() - t0))
    return {
        "symbol": symbol, "ok": True, "name": result.name,
        "price": result.price, "as_of": result.as_of,
        "stance": (result.ensemble or {}).get("stance"),
        "score": (result.ensemble or {}).get("score_pct"),
        "p_up_20d": (result.by_key("d20").p_up if result.by_key("d20") else None),
        "exp_20d": (result.by_key("d20").exp_return if result.by_key("d20") else None),
        "grade": (grade or {}).get("grade"),
        "paths": paths,
    }


def _to_json(result, bt_table, grade, calib) -> dict:
    b = result.bundle
    return {
        "symbol": result.symbol, "name": result.name, "as_of": result.as_of,
        "price": None if result.price != result.price else result.price,
        "forecasts": [
            {**f.to_row(), "horizon": f.horizon, "p_up": f.p_up, "p_down": f.p_down,
             "exp_return": f.exp_return, "ci_low": f.ci_low, "ci_high": f.ci_high,
             "sigma": f.sigma, "confidence": f.confidence,
             "trend": f.trend, "drivers": f.drivers, "counter": f.counter,
             "model_mix": f.model_mix}
            for f in result.forecasts
        ],
        "ensemble": result.ensemble,
        "meta": {"sources": result.meta.get("sources"), "ml": result.meta.get("ml"),
                 "volatility": result.meta.get("volatility"),
                 "consistency": result.meta.get("consistency"),
                 "n_factors": result.meta.get("n_factors"),
                 "errors": result.meta.get("errors"), "notes": result.meta.get("notes")},
        "backtest": (bt_table.to_dict("records") if bt_table is not None
                     and not bt_table.empty else None),
        "grade": grade, "calibration": calib,
        "data_warnings": list(getattr(b, "notes", []) or []),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="个股涨跌概率预测（本地版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("symbols", nargs="*", help="股票代码或名称，可多个")
    ap.add_argument("--file", help="从文本文件读取代码列表（每行一个，# 开头为注释）")
    ap.add_argument("--no-ml", action="store_true", help="禁用 AI 模型，仅用规则因子模型")
    ap.add_argument("--demo", action="store_true", help="演示模式：离线合成数据")
    ap.add_argument("--days", type=int, default=1600, help="历史数据天数（默认1600）")
    ap.add_argument("--backtest", action="store_true", help="执行历史回测与置信度校准")
    ap.add_argument("--charts", action="store_true", help="保存图表 PNG")
    ap.add_argument("--chart-dir", default=None, help="图表输出目录（默认 charts/）")
    ap.add_argument("--out-dir", default=None, help="报告输出目录（默认 reports/）")
    ap.add_argument("--report", action="store_true", default=True,
                    help="生成报告文件（默认开启）")
    ap.add_argument("--no-report", dest="report", action="store_false")
    ap.add_argument("--formats", nargs="+", default=["html", "txt"],
                    choices=["html", "txt", "md"], help="报告格式")
    ap.add_argument("--json", default=None, help="导出 JSON（文件或目录）")
    ap.add_argument("--health", action="store_true", help="数据源健康检查后退出")
    ap.add_argument("--offline", action="store_true", help="离线模式：只用缓存，不发网络请求")
    ap.add_argument("--quiet", action="store_true", help="精简日志")
    ap.add_argument("--verbose", action="store_true", help="打印详细异常")
    args = ap.parse_args(argv)

    if args.offline:
        config.RUNTIME["offline"] = True

    if args.health:
        try:
            from core import datasource as ds

            rep = ds.data_health_check()
            print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001
            print("健康检查失败：%s" % exc)
            return 1
        return 0

    symbols = list(args.symbols)
    if args.file:
        try:
            with open(args.file, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        symbols.append(line.split()[0])
        except Exception as exc:  # noqa: BLE001
            print("读取代码文件失败：%s" % exc)
            return 1
    if not symbols:
        ap.print_help()
        return 0

    args.symbols = symbols
    summaries = []
    for i, sym in enumerate(symbols, 1):
        print("\n[%d/%d] %s" % (i, len(symbols), sym))
        summaries.append(analyze_one(sym, args))

    if len(summaries) > 1:
        print(_banner("汇总"))
        rows = []
        for s in summaries:
            if not s.get("ok"):
                rows.append({"代码": s["symbol"], "状态": "失败", "备注": s.get("error")})
                continue
            rows.append({
                "代码": s["symbol"], "名称": s.get("name"), "现价": s.get("price"),
                "操作倾向": s.get("stance"), "评分": s.get("score"),
                "20天上涨概率": ("%.1f%%" % (s["p_up_20d"] * 100)) if s.get("p_up_20d") else "—",
                "20天期望": ("%+.2f%%" % s["exp_20d"]) if s.get("exp_20d") is not None else "—",
                "历史评级": s.get("grade") or "—",
            })
        try:
            print(pd.DataFrame(rows).to_string(index=False))
        except Exception:  # noqa: BLE001
            for r in rows:
                print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

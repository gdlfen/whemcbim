# -*- coding: utf-8 -*-
"""
历史回测与模型校准 —— 回答一个问题："这套方法在过去有多准？"

三类回测
--------
1. **规则因子分回测**：对历史每一根 K 线计算"快速规则分"（严格因果），
   统计它与未来 1/5/10/20/60/120/244 日实际收益的关系：
   * IC / RankIC（信息系数，>0.03 即算有效，>0.06 优秀）
   * 分组单调性（把分数分 5 组，看收益是否单调递增）
   * 方向胜率、多空组合收益、最大回撤
2. **AI 模型回测**：扩展窗口前向验证（walk-forward），逐段训练逐段测试，
   得到真实的样本外命中率与 AUC（**不允许用未来样本训练**）。
3. **报告**：把上面结果整理成"历史准确率表"，直接放进预测报告，
   让用户知道每个周期的结论应该打几折。

注意：A 股历史样本有限（尤其长周期），回测结果本身也有过拟合风险，
因此报告中会同时给出 **样本量与置信区间**，并明确提示"历史规律不代表未来"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .contracts import HORIZONS, horizon_spec
from .utils import get_logger, norm_ppf

log = get_logger()

BACKTEST_HORIZONS = ("d1", "d5", "d10", "d20", "m1", "q1", "h1", "y1")


@dataclass
class HorizonBacktest:
    """单个周期的回测结果。"""

    horizon: str
    label: str
    bars: int
    n_samples: int = 0
    ic: float = float("nan")            # 皮尔逊信息系数
    rank_ic: float = float("nan")       # 秩相关（更稳健）
    hit_rate: float = float("nan")      # 方向命中率
    hit_rate_ci: tuple = (float("nan"), float("nan"))
    avg_ret_when_bull: float = float("nan")   # 看多信号下的平均收益（%）
    avg_ret_when_bear: float = float("nan")
    long_short: float = float("nan")    # 多空组合平均收益（%）
    group_returns: list = field(default_factory=list)   # 5 分组的平均收益
    monotonic: float = float("nan")     # 分组单调性 -1~1
    max_drawdown: float = float("nan")
    edge: float = float("nan")          # 命中率相对 50% 的优势
    detail: str = ""

    @property
    def reliable(self) -> bool:
        """样本是否足够支撑结论（≥60 个独立样本，或 ≥30 且 |IC| 明显）。"""
        return self.n_samples >= 60 or (self.n_samples >= 30 and abs(self.ic) > 0.05)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return float("nan")
    a, b = a[ok], b[ok]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson 区间：命中率的置信区间（小样本下比正态近似可靠）。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def backtest_factor_score(df: pd.DataFrame, score: np.ndarray,
                          horizons=BACKTEST_HORIZONS) -> dict:
    """用历史规则分序列回测。**score 必须是严格因果的**（见 forecast.fast_rule_score_history）。"""
    out: dict[str, HorizonBacktest] = {}
    if df is None or len(df) < 60:
        return out
    c = df["close"].astype(float).to_numpy()
    n = len(c)
    score = np.asarray(score, dtype=float)
    if len(score) != n:
        return out

    for key in horizons:
        try:
            spec = horizon_spec(key)
        except KeyError:
            continue
        bars = spec.bars
        if n <= bars + 30:
            out[key] = HorizonBacktest(horizon=key, label=spec.label, bars=bars,
                                       detail="历史样本不足（%d 根），无法回测" % n)
            continue
        fwd = np.full(n, np.nan)
        fwd[:n - bars] = c[bars:] / c[:n - bars] - 1.0
        ok = np.isfinite(fwd) & np.isfinite(score)
        s, r = score[ok], fwd[ok] * 100.0
        if len(s) < 30:
            out[key] = HorizonBacktest(horizon=key, label=spec.label, bars=bars,
                                       n_samples=int(len(s)), detail="有效样本不足")
            continue

        # 只用非预热期的样本（score 预热期为 0，会稀释统计）
        warm = np.abs(s) > 1e-9
        s2, r2 = s[warm], r[warm]
        if len(s2) < 30:
            s2, r2 = s, r

        ic = _pearson(s2, r2)
        ric = _spearman(s2, r2)
        # 方向命中：分数为正预测上涨
        pred_up = s2 > 0
        actual_up = r2 > 0
        k = int((pred_up == actual_up).sum())
        hit = k / len(s2)
        ci = _wilson_ci(k, len(s2))

        bull = r2[s2 > 0.05]
        bear = r2[s2 < -0.05]
        ls = float(np.mean(bull) - np.mean(bear)) if len(bull) > 3 and len(bear) > 3 else float("nan")

        # 5 分组单调性
        groups = []
        try:
            q = pd.qcut(pd.Series(s2), 5, labels=False, duplicates="drop")
            for g in sorted(pd.unique(q.dropna())):
                groups.append(float(np.mean(r2[q.to_numpy() == g])))
        except Exception:  # noqa: BLE001
            groups = []
        mono = float("nan")
        if len(groups) >= 3:
            rho = _spearman(np.arange(len(groups), dtype=float), np.asarray(groups))
            mono = rho

        # 多空组合的净值回撤（等权、按信号方向持有）
        dd = float("nan")
        try:
            pos = np.sign(s2)
            pnl = pos * r2 / 100.0
            eq = np.cumprod(1.0 + pnl)
            peak = np.maximum.accumulate(eq)
            dd = float(np.min(eq / peak - 1.0) * 100.0)
        except Exception:  # noqa: BLE001
            pass

        detail = "样本 %d｜IC=%.3f｜方向胜率 %.1f%%（95%%CI %.1f%%~%.1f%%）" % (
            len(s2), ic if ic == ic else float("nan"), hit * 100,
            ci[0] * 100, ci[1] * 100)
        out[key] = HorizonBacktest(
            horizon=key, label=spec.label, bars=bars, n_samples=int(len(s2)),
            ic=ic, rank_ic=ric, hit_rate=hit, hit_rate_ci=ci,
            avg_ret_when_bull=float(np.mean(bull)) if len(bull) else float("nan"),
            avg_ret_when_bear=float(np.mean(bear)) if len(bear) else float("nan"),
            long_short=ls, group_returns=groups, monotonic=mono,
            max_drawdown=dd, edge=hit - 0.5, detail=detail)
    return out


def backtest_ai(df: pd.DataFrame, index_df: pd.DataFrame | None = None,
                horizons=("d1", "d5", "d10", "d20", "q1"), n_splits: int = 4) -> dict:
    """AI 模型的扩展窗口前向验证（样本外）。"""
    from . import mlmodel

    out: dict = {}
    if df is None or len(df) < config.RUNTIME["min_history_for_ml"]:
        return out
    try:
        x = mlmodel.build_features(df, index_df).to_numpy(dtype=float)
        t = mlmodel.make_targets(df, HORIZONS)
    except Exception as exc:  # noqa: BLE001
        log.warning("AI 回测特征构造失败: %s", exc)
        return out
    for key in horizons:
        spec = horizon_spec(key)
        col = "fwd_%s" % key
        if col not in t.columns:
            continue
        y_ret = t[col].to_numpy(dtype=float)
        ok = np.isfinite(y_ret) & (np.isfinite(x).mean(axis=1) >= 0.6)
        if ok.sum() < 120:
            out[key] = {"label": spec.label, "n": int(ok.sum()),
                        "detail": "样本不足，未做样本外验证"}
            continue
        xs, ys = x[ok], (y_ret[ok] > 0).astype(float)
        ev = mlmodel.walk_forward_eval(
            xs, ys, lambda: mlmodel.LogisticModel(l2=1.5, lr=0.3, epochs=300),
            n_splits=n_splits)
        # GBDT 也验证一次（样本足够时）
        ev2 = {}
        if len(ys) >= 250:
            try:
                ev2 = mlmodel.walk_forward_eval(
                    xs, ys, lambda: mlmodel.GBDTModel(n_trees=40, max_depth=2,
                                                      min_leaf=max(10, len(ys) // 40)),
                    n_splits=n_splits)
            except Exception:  # noqa: BLE001
                ev2 = {}
        out[key] = {
            "label": spec.label, "bars": spec.bars, "n": int(len(ys)),
            "logit": ev,
            "gbdt": ev2,
            "best_auc": max([v.get("auc", float("nan")) for v in (ev, ev2)
                             if v and v.get("auc") == v.get("auc")] or [float("nan")]),
            "base_rate": float(ys.mean()),
        }
    return out


def backtest_prediction_history(df: pd.DataFrame, index_df: pd.DataFrame | None = None,
                                horizons=BACKTEST_HORIZONS) -> dict:
    """规则分的滚动回测（对同一只股票、用同一条因果规则分序列）。"""
    from .forecast import fast_rule_score_history

    if df is None or len(df) < 120:
        return {}
    scores = {k: fast_rule_score_history(df, k) for k in horizons}
    results = {}
    for k, s in scores.items():
        bt = backtest_factor_score(df, s, horizons=[k])
        if k in bt:
            results[k] = bt[k]
    return results


# --------------------------------------------------------------------------- #
# 校准：把回测得到的准确率反馈到预测置信度上
# --------------------------------------------------------------------------- #
def calibrate_confidence(forecasts: list, bt: dict) -> dict:
    """用历史回测结果**折扣**每个周期的置信度。

    规则：
    * 样本充足且 IC/RankIC 为正 → 保持或小幅上调置信度；
    * IC 接近 0 或为负 → 显著下调（历史证明这个周期的信号没用）；
    * 样本不足 → 下调（无法验证 = 不确定）。
    """
    applied = {}
    for f in forecasts:
        hb = bt.get(f.horizon)
        if hb is None:
            applied[f.horizon] = {"factor": 0.9, "reason": "无回测样本，保守折扣"}
            f.confidence = _clamp(f.confidence * 0.9)
            continue
        ric = getattr(hb, "rank_ic", float("nan"))
        ic = getattr(hb, "ic", float("nan"))
        n = getattr(hb, "n_samples", 0)
        if n < 30:
            factor, reason = 0.85, "样本过少（%d）" % n
        else:
            ic_use = ric if ric == ric else ic
            if ic_use != ic_use:
                factor, reason = 0.9, "IC 无法估计"
            elif ic_use >= 0.06:
                factor, reason = 1.12, "历史 IC=%.3f 较强" % ic_use
            elif ic_use >= 0.03:
                factor, reason = 1.05, "历史 IC=%.3f 有效" % ic_use
            elif ic_use >= 0.0:
                factor, reason = 0.95, "历史 IC=%.3f 偏弱" % ic_use
            elif ic_use >= -0.03:
                factor, reason = 0.82, "历史 IC=%.3f 接近无效" % ic_use
            else:
                factor, reason = 0.68, "历史 IC=%.3f 反向，已显著折扣" % ic_use
            # 命中率也参与
            hr = getattr(hb, "hit_rate", float("nan"))
            if hr == hr:
                if hr >= 0.55:
                    factor *= 1.05
                elif hr < 0.46:
                    factor *= 0.85
        f.confidence = _clamp(f.confidence * factor)
        applied[f.horizon] = {"factor": round(factor, 3), "reason": reason,
                              "ic": None if ic != ic else round(ic, 4),
                              "rank_ic": None if ric != ric else round(ric, 4),
                              "hit_rate": None if getattr(hb, "hit_rate", float("nan")) !=
                              getattr(hb, "hit_rate", float("nan")) else round(hb.hit_rate, 4),
                              "n": n}
    return applied


def _clamp(x: float, lo: float = 0.03, hi: float = 0.92) -> float:
    return lo if x < lo else (hi if x > hi else x)


def summarize(bt: dict) -> pd.DataFrame:
    """把回测结果整理成表格。"""
    rows = []
    for k, hb in sorted(bt.items(), key=lambda kv: horizon_spec(kv[0]).bars):
        if not isinstance(hb, HorizonBacktest):
            continue
        rows.append({
            "周期": hb.label,
            "样本数": hb.n_samples,
            "IC": None if hb.ic != hb.ic else round(hb.ic, 4),
            "RankIC": None if hb.rank_ic != hb.rank_ic else round(hb.rank_ic, 4),
            "方向胜率": None if hb.hit_rate != hb.hit_rate else "%.1f%%" % (hb.hit_rate * 100),
            "竞率区间": ("%.1f%%~%.1f%%" % (hb.hit_rate_ci[0] * 100, hb.hit_rate_ci[1] * 100))
                        if hb.hit_rate_ci[0] == hb.hit_rate_ci[0] else "—",
            "看多时均收益": None if hb.avg_ret_when_bull != hb.avg_ret_when_bull
                            else "%+.2f%%" % hb.avg_ret_when_bull,
            "看空时均收益": None if hb.avg_ret_when_bear != hb.avg_ret_when_bear
                            else "%+.2f%%" % hb.avg_ret_when_bear,
            "多空差": None if hb.long_short != hb.long_short else "%+.2f%%" % hb.long_short,
            "分组单调性": None if hb.monotonic != hb.monotonic else round(hb.monotonic, 3),
            "说明": hb.detail,
        })
    return pd.DataFrame(rows)


def reliability_grade(bt: dict) -> dict:
    """给整套方法一个总体可信度评级（用于报告顶部的免责说明）。"""
    if not bt:
        return {"grade": "未验证", "score": 0.0,
                "detail": "历史样本不足，未做回测验证，请谨慎使用"}
    ics = [hb.ic for hb in bt.values() if isinstance(hb, HorizonBacktest)
           and hb.ic == hb.ic and hb.n_samples >= 30]
    hits = [hb.hit_rate for hb in bt.values() if isinstance(hb, HorizonBacktest)
            and hb.hit_rate == hb.hit_rate and hb.n_samples >= 30]
    if not ics:
        return {"grade": "未验证", "score": 0.0, "detail": "有效样本不足"}
    mean_ic = float(np.mean(ics))
    mean_hit = float(np.mean(hits)) if hits else 0.5
    pos_ratio = sum(1 for x in ics if x > 0) / len(ics)
    score = 0.5 * min(max(mean_ic / 0.06, -1), 1) + 0.3 * ((mean_hit - 0.5) / 0.08) + \
        0.2 * (pos_ratio * 2 - 1)
    score = max(-1.0, min(1.0, score))
    if score >= 0.55:
        grade, desc = "A（历史验证有效）", "多数周期的历史信号与未来收益显著正相关"
    elif score >= 0.25:
        grade, desc = "B（部分周期有效）", "中短周期有效，长周期信号偏弱"
    elif score >= -0.1:
        grade, desc = "C（信号偏弱）", "历史相关度低，本结论仅供参考"
    else:
        grade, desc = "D（历史不成立）", "历史回测未支持该信号，请勿据此交易"
    return {"grade": grade, "score": round(score, 3), "mean_ic": round(mean_ic, 4),
            "mean_hit": round(mean_hit, 4), "positive_ratio": round(pos_ratio, 3),
            "detail": desc,
            "advice": "历史规律不代表未来；本结果不构成投资建议。"}


if __name__ == "__main__":  # pragma: no cover
    import time

    from .forecast import fast_rule_score_history
    from .selftest_data import make_index, make_ohlcv

    t0 = time.time()
    df = make_ohlcv(1400, seed=42)
    idx = make_index(1400, seed=7)
    print("样本 %d 根，%s ~ %s" % (len(df), str(df['date'].iloc[0])[:10],
                                  str(df['date'].iloc[-1])[:10]))
    bt = backtest_prediction_history(df, idx)
    tbl = summarize(bt)
    print(tbl[["周期", "样本数", "IC", "RankIC", "方向胜率", "多空差", "说明"]].to_string(index=False))
    grade = reliability_grade(bt)
    print("\n可信度评级:", grade["grade"], grade["score"], "|", grade["detail"])

    # 校准演示
    from .contracts import ProbabilityForecast
    fcs = [ProbabilityForecast(horizon=k, label=horizon_spec(k).label, p_up=0.6,
                               confidence=0.6) for k in bt.keys()]
    applied = calibrate_confidence(fcs, bt)
    print("\n置信度校准示例:")
    for f in fcs:
        info = applied[f.horizon]
        print("  %-16s %.3f → %.3f   (%s)" % (f.label, 0.6, f.confidence, info["reason"]))

    # 断言：因果性
    s_full = fast_rule_score_history(df, "d20")
    s_trunc = fast_rule_score_history(df.iloc[:800], "d20")
    assert np.allclose(s_full[:800], s_trunc, atol=1e-9), "规则分存在未来函数"
    # 断言：随机分数不应显示出有效性
    rng = np.random.default_rng(0)
    rand = rng.normal(0, 0.3, len(df))
    bt_rand = backtest_factor_score(df, rand, horizons=["d20"])["d20"]
    print("\n随机信号对照: IC=%.4f 胜率=%.1f%%（应接近 0 / 50%%）" % (
        bt_rand.ic, bt_rand.hit_rate * 100))
    assert abs(bt_rand.ic) < 0.12, "随机信号不该有高 IC，回测实现可能有误"
    assert len(tbl) == 8, "应回测 8 个周期"
    print("耗时 %.1fs" % (time.time() - t0))
    print("backtest self-test OK")

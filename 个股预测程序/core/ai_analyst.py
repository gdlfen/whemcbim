# -*- coding: utf-8 -*-
"""
AI 复核与资料补全（可选功能，需配置 API Key）。

设计原则（非常重要）
--------------------
**AI 绝不编造数字进入打分模型。** 理由：一个幻觉出来的 CPI 值，
比"没有数据"危险得多——程序会把假数据当成真的算出一个漂亮结论。

所以本模块严格区分两类用途：

① **文字分析与交叉复核**（默认用途，安全）
   把程序算出的结论汇总喂给 AI，让它做：
     · 指出结论之间的矛盾（如"技术面全破但大周期判牛市"）
     · 补充程序看不到的语境（政策方向、产业逻辑、海外环境）
     · 对概率与阶段判定提出质疑
     · 列出"还缺什么信息才能下结论"
   输出**纯文字**，只展示给人看，不参与计算。

② **缺失数据的线索**（谨慎用途）
   对确实拿不到的指标（如开户数），可让 AI 给出"大致区间与出处线索"，
   但结果一律标注 **「AI 推测，未经验证」**，且：
     · 不写入 `macro_source` 的手动数据文件
     · 不进入 `fundamental_factors` / 环境打分
     · 界面只作为"参考线索"展示，需要人确认后手动录入
   这样 AI 是"帮你找线索的助手"，不是"数据源"。

配置（环境变量）
----------------
    STOCKAI_AI_KEY     API Key（未配置则整个功能自动禁用）
    STOCKAI_AI_BASE    接口地址，默认 https://api.deepseek.com
    STOCKAI_AI_MODEL   模型名，默认 deepseek-chat
    STOCKAI_AI_TIMEOUT 超时秒数，默认 60

无 Key 时所有函数安全返回"未配置"，不报错、不影响主流程。
"""
from __future__ import annotations

import json
import os
import threading
import time

from .utils import get_logger

log = get_logger()

#: 结果缓存：同一问题 10 分钟内不重复调用（省钱、省时）
_CACHE: dict = {}
_CACHE_TTL = 600.0
_LOCK = threading.RLock()

#: 系统提示词：约束 AI 不要编数据
SYSTEM_PROMPT = """你是A股量化分析师，负责复核一个量化程序的结论。

严格要求：
1. **不要编造任何数字**。需要数据时说明"缺少X数据"，而不是猜一个值。
2. 只基于我给的信息做推理；引用外部知识时要说明"据我所知"。
3. 明确指出程序结论**内部矛盾**之处，这是你最重要的价值。
4. 对概率预测，说明它可能的偏差来源（因子缺失/样本偏差/过拟合）。
5. 用中文，结构化输出，控制在 700 字内。

输出格式：
### 一、结论是否自洽
### 二、我看到的主要矛盾
### 三、缺什么信息才能下更可靠结论
### 四、对程序判定的质疑或补充"""


#: 界面配置的存放位置（含 API Key，已在 .gitignore 中排除，勿提交）
CONFIG_FILE = os.environ.get("STOCKAI_AI_CONFIG") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ai_config.json")


def load_config() -> dict:
    """读 AI 配置（不含环境变量）。

    **只从当前设备读取**：本地落盘 data/ai_config.json；
    云端部署不落盘，改为按浏览器会话隔离的内存存储（见 core/local_store.py）。
    """
    try:
        from . import local_store as LS

        return LS.load("ai_config.json")
    except Exception as exc:  # noqa: BLE001
        log.debug("读 AI 配置失败: %s", exc)
        return {}


def save_config(cfg: dict) -> bool:
    """保存 AI 配置（含 Key）——**仅当前设备/会话可见**，绝不写入仓库。"""
    try:
        from . import local_store as LS

        return LS.save("ai_config.json", cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("保存 AI 配置失败: %s", exc)
        return False


def storage_note() -> str:
    """界面用：AI 配置当前保存在哪里。"""
    try:
        from . import local_store as LS

        return LS.describe("ai_config.json")
    except Exception:  # noqa: BLE001
        return ""


def mask_key(k: str) -> str:
    k = (k or "").strip()
    if not k:
        return ""
    if len(k) <= 10:
        return k[:2] + "…"
    return k[:6] + "…" + k[-4:]


def _cfg() -> dict:
    """生效配置：**环境变量优先**，其次界面保存的文件，最后默认值。"""
    f = load_config()
    key = os.environ.get("STOCKAI_AI_KEY", "").strip() or str(f.get("key") or "").strip()
    base = (os.environ.get("STOCKAI_AI_BASE")
            or f.get("base") or "https://api.deepseek.com")
    model = (os.environ.get("STOCKAI_AI_MODEL")
             or f.get("model") or "deepseek-chat")
    to = os.environ.get("STOCKAI_AI_TIMEOUT") or f.get("timeout") or 60
    return {"key": str(key).strip(), "base": str(base).strip(),
            "model": str(model).strip(),
            "timeout": float(to),
            "from_env": bool(os.environ.get("STOCKAI_AI_KEY", "").strip())}


def available() -> bool:
    """是否已配置 API Key。"""
    return bool(_cfg()["key"])


def status() -> dict:
    c = _cfg()
    src = ("环境变量" if c.get("from_env") else
           ("界面配置" if c["key"] else ""))
    return {"available": bool(c["key"]), "base": c["base"], "model": c["model"],
            "key_masked": mask_key(c["key"]), "source": src,
            "config_file": CONFIG_FILE,
            "detail": ("已配置（来源：%s）：%s ｜ 模型 %s"
                       % (src, c["base"], c["model"])
                       if c["key"] else
                       "未配置 API Key（可在侧边栏「AI 设置」里填写，"
                       "或设置环境变量 STOCKAI_AI_KEY）")}


def chat(messages: list, *, temperature: float = 0.3,
         max_tokens: int = 1200) -> dict:
    """调用 OpenAI 兼容的 chat/completions 接口。

    Returns
    -------
    dict(ok, text, error, model, elapsed_ms)
    """
    c = _cfg()
    if not c["key"]:
        return {"ok": False, "text": "", "error": "未配置 API Key", "model": "",
                "elapsed_ms": 0}
    import requests

    url = c["base"].rstrip("/") + "/v1/chat/completions"
    payload = {"model": c["model"], "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens,
               "stream": False}
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=c["timeout"],
                          headers={"Authorization": "Bearer " + c["key"],
                                   "Content-Type": "application/json"})
        ms = int((time.time() - t0) * 1000)
        if r.status_code != 200:
            return {"ok": False, "text": "", "model": c["model"],
                    "error": "HTTP %s: %s" % (r.status_code, r.text[:160]),
                    "elapsed_ms": ms}
        j = r.json()
        txt = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return {"ok": bool(txt), "text": txt.strip(), "error": "" if txt else "空回复",
                "model": j.get("model") or c["model"], "elapsed_ms": ms,
                "usage": j.get("usage") or {}}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": "", "model": c["model"],
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:140]),
                "elapsed_ms": int((time.time() - t0) * 1000)}


# --------------------------------------------------------------------------- #
# 一、把程序结论压成一份"给 AI 看的事实清单"
# --------------------------------------------------------------------------- #
def summarize_payload(payload: dict) -> str:
    """从分析 payload 里提取关键结论，压缩成紧凑文本（控制 token）。"""
    if not isinstance(payload, dict):
        return "（无有效分析结果）"
    mk = payload.get("market") or {}
    stt = mk.get("structure") or {}
    deep = stt.get("deep") or {}
    comp = deep.get("compose") or {}
    fine = deep.get("fine") or {}
    rot = mk.get("rotation") or {}
    env = mk.get("macro_env") or {}
    mf = mk.get("market_fundamental") or {}
    heat = mk.get("retail_heat") or {}
    drv = mk.get("drivers") or {}
    idx_fc = mk.get("index_forecasts") or {}

    L = []
    L.append("## 标的：%s（%s）" % (payload.get("name") or "—",
                                  payload.get("as_of") or "—"))
    if payload.get("price") is not None:
        L.append("现价：%s" % payload.get("price"))

    # 三层结构
    reg = stt.get("regime") or {}
    ph = stt.get("phase") or {}
    if reg:
        L.append("\n## 程序的三层判定")
        L.append("大周期：%s（距5年低点%s%%、距高点%s%%）"
                 % (reg.get("regime"), reg.get("from_low_pct"), reg.get("from_high_pct")))
    if ph:
        L.append("中级波段：%s（置信度%.0f%%）闸门=%s"
                 % (ph.get("label"), (ph.get("confidence") or 0) * 100,
                    "、".join(ph.get("gates") or []) or "无"))
    if deep:
        L.append("四维评分：总%.0f/100（%s）｜驱动%.0f 结构%.0f 位置%.0f 量价%.0f"
                 % ((comp.get("total") or 0) * 100, comp.get("lean"),
                    (comp.get("dims", {}).get("driver") or 0) * 100,
                    (comp.get("dims", {}).get("structure") or 0) * 100,
                    (comp.get("dims", {}).get("position") or 0) * 100,
                    (comp.get("dims", {}).get("volume") or 0) * 100))
        if fine.get("stage"):
            L.append("精细阶段：%s ｜ 浪型：%s" % (fine["stage"], fine.get("wave") or "未明"))
        if fine.get("rally_pct") is not None:
            L.append("最近摆动：拉升%.2f%%、回撤%s"
                     % (fine["rally_pct"],
                        ("%.0f%%" % (fine["pullback_ratio"] * 100))
                        if fine.get("pullback_ratio") is not None else "—"))

    # 概率预测
    if idx_fc:
        L.append("\n## 指数上涨概率（相对分析日现价）")
        for k, series in list(idx_fc.items())[:3]:
            cells = " ".join("%s=%.0f%%" % (x["horizon"], x["p_up"] * 100)
                             for x in series[:6])
            L.append("%s：%s（置信度%.0f%%）"
                     % (k, cells, (series[0].get("confidence") or 0) * 100))

    # 宏观与资金
    ind = env.get("indicators") or {}
    if ind:
        bits = []
        for k, lab in (("pmi", "PMI"), ("cpi", "CPI累计"), ("ppi", "PPI累计"),
                       ("m2", "M2"), ("lpr", "LPR")):
            d = ind.get(k) or {}
            if d.get("value") is not None:
                bits.append("%s=%s(%s)" % (lab, d.get("value"), d.get("date") or "—"))
        if bits:
            L.append("\n## 宏观（程序已校验新鲜度）")
            L.append("；".join(bits))
        if env.get("stale"):
            L.append("已排除过期指标：%s" % "、".join(env["stale"]))
    pe = mf.get("pe") or {}
    mg = mf.get("margin") or {}
    nt = mf.get("north") or {}
    if pe.get("percentile") is not None:
        L.append("指数估值：PE %.2f（历史分位%.1f%%）"
                 % (pe.get("pe") or 0, pe["percentile"]))
    if mg.get("change_20d_pct") is not None:
        L.append("两融：余额%.0f亿，近20日%+.2f%%"
                 % (mg.get("balance") or 0, mg["change_20d_pct"]))
    if nt.get("sum_20d") is not None:
        L.append("北向：近20日%+.1f亿（20日中%d天净买）"
                 % (nt["sum_20d"], nt.get("positive_days_20") or 0))
    if heat.get("score") is not None:
        L.append("散户入场热度：%.0f/100（%s）*代理指标，非开户数"
                 % (heat["score"], heat.get("level") or ""))

    # 板块与动能
    if rot.get("board_breadth_pct") is not None:
        L.append("\n## 板块")
        L.append("宽度%.1f%%（%d/%d 个板块），完整性%s"
                 % (rot.get("board_breadth_pct") or 0,
                    round((rot.get("board_breadth_pct") or 0) / 100
                          * (rot.get("board_count") or 0)),
                    rot.get("board_count") or 0,
                    rot.get("breadth_quality") or "—"))
    if drv.get("intent"):
        L.append("主要动能对象：%s → 意图「%s」（证据数%d）"
                 % (((drv.get("items") or [{}])[0].get("name") or "—"),
                    drv["intent"], drv.get("n_evidence") or 0))

    # 数据缺口
    gaps = []
    if env.get("stale"):
        gaps.append("宏观部分指标过期：%s" % "、".join(env["stale"]))
    # 两融变化：以 market_fundamental 为准（散户热度里没带上是正常的，
    # 之前用 heat.proxies 判断会误报"缺两融变化"，AI 据此质疑摘要不完整）
    if mg.get("change_20d_pct") is None:
        gaps.append("缺两融 20 日变化")
    if not idx_fc:
        gaps.append("无指数概率预测")
    if not rot.get("board_count"):
        gaps.append("无板块数据")
    L.append("\n## 已知数据缺口")
    L.append("；".join(gaps) if gaps else "（主要指标齐全）")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 一之二、板块与个股的核对素材
# --------------------------------------------------------------------------- #
def summarize_boards(payload: dict, top_n: int = 12) -> str:
    """板块核对素材：板块宽度、概率预测、动能对象、轮动提示。"""
    if not isinstance(payload, dict):
        return "（无有效分析结果）"
    mk = payload.get("market") or {}
    rot = mk.get("rotation") or {}
    bfs = mk.get("board_forecasts") or []
    drv = mk.get("drivers") or {}
    L = ["## 板块概览"]
    if rot.get("board_breadth_pct") is not None:
        L.append("宽度 %.1f%%（%d 个板块），完整性 %s"
                 % (rot["board_breadth_pct"], rot.get("board_count") or 0,
                    rot.get("breadth_quality") or "—"))
    if rot.get("board_avg_pct") is not None:
        L.append("平均涨幅 %+.2f%%、中位 %+.2f%%"
                 % (rot.get("board_avg_pct") or 0, rot.get("board_median_pct") or 0))
    hint = rot.get("rotation_hint") or {}
    if hint:
        L.append("轮动提示：%s" % (hint.get("detail") or hint))

    if bfs:
        L.append("\n## 板块上涨概率（相对当前价）")
        L.append("板块｜今日%｜净流入亿｜1日｜5日｜20日｜5日期望%｜置信度")
        shown = 0
        for b in bfs:
            fs = b.get("forecasts") or []
            if not fs:
                L.append("%s｜%+.2f%%｜—｜无预测：%s"
                         % (b.get("name"), b.get("pct") or 0,
                            (b.get("note") or "")[:30]))
                continue
            d = {x["horizon"]: x for x in fs}
            L.append("%s｜%+.2f%%｜%s｜%s｜%s｜%s｜%s｜%.0f%%" % (
                b.get("name"), b.get("pct") or 0,
                b.get("net_inflow") if b.get("net_inflow") is not None else "—",
                ("%.0f%%" % (d["d1"]["p_up"] * 100)) if "d1" in d else "—",
                ("%.0f%%" % (d["d5"]["p_up"] * 100)) if "d5" in d else "—",
                ("%.0f%%" % (d["d20"]["p_up"] * 100)) if "d20" in d else "—",
                ("%+.2f" % d["d5"]["exp_return"]) if "d5" in d else "—",
                (fs[0].get("confidence") or 0) * 100))
            shown += 1
            if shown >= top_n:
                break

    if drv.get("intent"):
        L.append("\n## 主要动能对象")
        for it in (drv.get("items") or [])[:5]:
            L.append("%s｜%+.2f%%｜成交占比%s%%｜贡献%s｜宽度%s%%"
                     % (it.get("name"), it.get("pct") or 0,
                        ("%.1f" % it["share"]) if it.get("share") is not None else "—",
                        ("%+.3f" % it["contrib"]) if it.get("contrib") is not None else "—",
                        ("%.0f" % it["breadth"]) if it.get("breadth") is not None else "—"))
        L.append("意图判定：%s（证据数 %d）"
                 % (drv["intent"], drv.get("n_evidence") or 0))
    L.append("\n## 注：板块缺少基本面维度，因子数少于个股，置信度天然偏低。")
    return "\n".join(L)


def summarize_subject(payload: dict, top_factors: int = 30) -> str:
    """个股/基金核对素材：八周期概率、阶段、关键因子、基金专属项。"""
    if not isinstance(payload, dict):
        return "（无有效分析结果）"
    L = ["## 标的：%s（%s）现价 %s"
         % (payload.get("name") or "—", payload.get("as_of") or "—",
            payload.get("price") if payload.get("price") is not None else "—")]
    fc = payload.get("forecasts") or []
    if fc:
        L.append("\n## 八周期概率（相对分析日现价）")
        L.append("周期｜P涨｜P跌｜期望%｜区间｜置信度｜大涨阈值%")
        for f in fc:
            g = f.get if isinstance(f, dict) else (lambda k, _f=f: getattr(_f, k, None))
            L.append("%s｜%.0f%%｜%.0f%%｜%+.2f｜[%+.1f,%+.1f]｜%.0f%%｜%.1f" % (
                g("label") or g("horizon"),
                (g("p_up") or 0) * 100, (g("p_down") or 0) * 100,
                g("exp_return") or 0, g("ci_low") or 0, g("ci_high") or 0,
                (g("confidence") or 0) * 100,
                g("big_move_threshold") or 0))
    # 因子（按 |强度×可信度| 排序取前 N）
    # 注意：payload 的因子可能是中文键（因子/方向/强度/可信度/适用周期），
    # 这里必须与 summarize_factors 用同一套 ALIAS 映射，否则全取不到值，
    # 发给 AI 的会是"方向0/强度0/可信度0"的空表（AI 核对图1 第2条）。
    fs = payload.get("factors_all") or []
    if fs:
        A = {"name": ("name", "因子", "名称"),
             "category": ("category", "类别", "大类"),
             "direction": ("direction", "方向"),
             "strength": ("strength", "强度"),
             "reliability": ("reliability", "可信度"),
             "horizon": ("horizon", "适用周期", "周期")}
        DCN = {"多": 1, "空": -1, "中性": 0, "偏多": 1, "偏空": -1}

        def _gp(o, fld):
            for k in A[fld]:
                if isinstance(o, dict):
                    if k in o and o[k] is not None:
                        return o[k]
                else:
                    v = getattr(o, k, None)
                    if v is not None:
                        return v
            return None

        def _key(f):
            return -abs((_gp(f, "strength") or 0) * (_gp(f, "reliability") or 0))

        top = sorted(fs, key=_key)[:top_factors]
        L.append("\n## 关键因子（前 %d 条，按强度×可信度排序）" % len(top))
        L.append("名称｜大类｜方向｜强度｜可信度｜周期")
        for f in top:
            dv = _gp(f, "direction")
            if isinstance(dv, str):
                dv = DCN.get(dv.strip(), 0)
            hz = _gp(f, "horizon")
            if isinstance(hz, (list, tuple)):
                hz = ",".join(str(x) for x in hz[:3])
            elif isinstance(hz, str):
                hz = hz.replace("/", ",")[:16]
            L.append("%s｜%s｜%s｜%s｜%s｜%s" % (
                str(_gp(f, "name") or "")[:20], _gp(f, "category") or "",
                {1: "多", -1: "空", 0: "中"}.get(dv, "—"),
                ("%.2f" % _gp(f, "strength")) if isinstance(_gp(f, "strength"), (int, float)) else "—",
                ("%.2f" % _gp(f, "reliability")) if isinstance(_gp(f, "reliability"), (int, float)) else "—",
                hz or "—"))
    # 阶段与一致性
    stg = payload.get("stage") or {}
    if stg:
        L.append("\n## 个股阶段/一致性")
        if stg.get("label"):
            L.append("阶段：%s（置信度 %.0f%%）"
                     % (stg["label"], (stg.get("confidence") or 0) * 100))
        if stg.get("consistency") is not None:
            L.append("八周期一致性：%.0f%%" % (stg["consistency"] * 100))
    # 基金专属
    fd = payload.get("fund") or {}
    if fd:
        L.append("\n## 基金专属")
        if fd.get("type"):
            L.append("类型：%s" % fd["type"])
        ha = fd.get("holdings_analysis") or {}
        if ha.get("detail"):
            L.append("成分股：%s" % ha["detail"][:120])
    return "\n".join(L)


BOARD_PROMPT = """你是A股板块轮动研究员。用户会给出一份**板块层面**的量化结论。

请重点审查：
1. 板块概率与"当日涨跌幅"是否矛盾（如今日大涨却给低概率，要说明是
   均值回归逻辑还是数据问题）
2. 宽度（涨跌家数比）与板块涨幅是否一致；宽度数据是否可能失真
3. 只对"最强与最弱两端"做了预测，这个抽样是否有偏——中间的板块被忽略会漏掉什么
4. 动能对象的"成交占比"能否支撑"贡献度"的说法（口径是否可靠）
5. 板块缺少基本面维度，哪些结论因此**不该**下

严格要求：不要编造清单外的板块或数字；不确定就说证据不足。中文，700 字内。
输出：### 一、矛盾之处 / ### 二、抽样偏差与数据问题 /
      ### 三、不该下的结论 / ### 四、建议补充的观察点"""

SUBJECT_PROMPT = """你是A股个股研究员。用户会给出一只个股（或基金）的量化预测结论。

请重点审查：
1. **八周期概率是否自洽**：短期偏空、长期回到50%，是"证据不足回到先验"
   还是真有转向逻辑？请指出哪些周期的概率属于"没有信息量"
2. **因子是否重复计入**：同一信息（如均线、量价）被多个因子反复计权
3. **方向与数值矛盾**：如"估值低位"给了偏空、"放量下跌"给了偏多
4. **置信度是否虚高**：因子少、样本短时可信度应更低
5. **期望幅度与区间是否匹配**：期望是否落在 CI 内、幅度是否合理
6. 基金标的：折溢价/跟踪误差/成分股集中度 是否被恰当使用

严格要求：不要编造清单外的因子或数字；不确定就说证据不足。中文，700 字内。
输出：### 一、概率自洽性 / ### 二、因子冗余与矛盾 /
      ### 三、置信度与区间问题 / ### 四、最应该补的信息"""


def review_target(payload: dict, target: str = "market", *, use_cache: bool = True,
                  extra_question: str = "") -> dict:
    """按核对对象选择素材与提示词。

    target: ``market`` 大盘 / ``board`` 板块 / ``subject`` 个股基金
    """
    if not available():
        return {"ok": False, "text": "", "error": "未配置 API Key",
                "model": "", "elapsed_ms": 0, "facts": ""}
    if target == "board":
        facts, prompt = summarize_boards(payload), BOARD_PROMPT
    elif target == "subject":
        facts, prompt = summarize_subject(payload), SUBJECT_PROMPT
    else:
        facts, prompt = summarize_payload(payload), SYSTEM_PROMPT
    key = hash((target, facts, extra_question))
    with _LOCK:
        got = _CACHE.get(key)
        if use_cache and got and (time.time() - got[0]) < _CACHE_TTL:
            return {**got[1], "cached": True}
    user = facts + ("\n\n额外问题：" + extra_question if extra_question else "")
    r = chat([{"role": "system", "content": prompt},
              {"role": "user", "content": user}])
    r["facts"], r["target"] = facts, target
    if r.get("ok"):
        with _LOCK:
            _CACHE[key] = (time.time(), r)
    return r


# --------------------------------------------------------------------------- #
# 二、AI 复核
# --------------------------------------------------------------------------- #
def review(payload: dict, *, extra_question: str = "",
           use_cache: bool = True) -> dict:
    """让 AI 复核程序结论，指出矛盾与缺口。

    Returns
    -------
    dict(ok, text, error, model, elapsed_ms, facts)
    """
    if not available():
        return {"ok": False, "text": "", "error": "未配置 API Key（STOCKAI_AI_KEY）",
                "model": "", "elapsed_ms": 0, "facts": ""}
    facts = summarize_payload(payload)
    key = hash((facts, extra_question))
    with _LOCK:
        got = _CACHE.get(key)
        if use_cache and got and (time.time() - got[0]) < _CACHE_TTL:
            return {**got[1], "cached": True}
    user = ("以下是量化程序刚算出的结论，请你复核。\n\n" + facts
            + ("\n\n额外问题：" + extra_question if extra_question else ""))
    r = chat([{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": user}])
    r["facts"] = facts
    if r.get("ok"):
        with _LOCK:
            _CACHE[key] = (time.time(), r)
    return r


def summarize_factors(payload: dict, top_n: int = 40) -> str:
    """把因子列表压缩成紧凑文本，供 AI 逐条评价。

    只保留判断"合理性"需要的字段：方向/强度/可信度/周期/理论依据/数值。
    长明细（detail）截断，避免 token 爆炸。
    """
    if not isinstance(payload, dict):
        return ""
    fs = payload.get("factors_all") or payload.get("factors") or []
    if not fs:
        return "（无因子数据）"
    # payload 里的因子既可能是英文键（name/category/…），也可能是
    # 已本地化的中文键（因子/类别/方向/强度/可信度/观测值/理论依据/适用周期）。
    # 之前只读英文键，导致发给 AI 的清单八项全空——AI 直接拒答。
    ALIAS = {
        "name": ("name", "因子", "名称"),
        "category": ("category", "类别", "大类"),
        "direction": ("direction", "方向"),
        "strength": ("strength", "强度"),
        "reliability": ("reliability", "可信度"),
        "horizon": ("horizon", "适用周期", "周期"),
        "value": ("value", "观测值", "数值"),
        "theory": ("theory", "理论依据"),
    }
    DIR_CN = {"多": 1, "空": -1, "中性": 0, "偏多": 1, "偏空": -1}

    def _pick(obj, field):
        for k in ALIAS[field]:
            if isinstance(obj, dict):
                if k in obj and obj[k] is not None:
                    return obj[k]
            else:
                v = getattr(obj, k, None)
                if v is not None:
                    return v
        return None

    rows = []
    for f in fs[:top_n]:
        name = str(_pick(f, "name") or "")[:22]
        cat = str(_pick(f, "category") or "")
        d = _pick(f, "direction")
        if isinstance(d, str):
            d = DIR_CN.get(d.strip(), 0)
        sgn = {1: "多", -1: "空", 0: "中"}.get(d, "—")
        st = _pick(f, "strength")
        rel = _pick(f, "reliability")
        hz = _pick(f, "horizon")
        if isinstance(hz, (list, tuple)):
            hz = ",".join(str(x) for x in hz[:3])
        elif isinstance(hz, str):
            hz = hz.replace("/", ",")[:16]
        _v = _pick(f, "value")
        if isinstance(_v, dict):
            # 值本身是 dict 时压成 "k=v" 形式，避免素材里出现原始字典
            _v = "、".join("%s=%s" % (k, v) for k, v in list(_v.items())[:3])
        val = str(_v or "")[:24]
        theory = str(_pick(f, "theory") or "")[:20]
        rows.append("%s｜%s｜%s｜强%s｜信%s｜%s｜%s｜%s" % (
            name, cat, sgn,
            ("%.2f" % st) if isinstance(st, (int, float)) else "—",
            ("%.2f" % rel) if isinstance(rel, (int, float)) else "—",
            hz or "—", val or "—", theory or "—"))
    # 若仍有大量空行，说明字段映射失败——此时明确告知，不把空表喂给 AI
    blank = sum(1 for r in rows if r.startswith("｜") or r.split("｜")[0] == "")
    if blank >= max(1, int(len(rows) * 0.8)):
        return ("（因子字段解析失败：共 %d 条，但未取到名称/方向等字段。"
                "请检查 payload 结构。）" % len(fs))
    head = ("因子共 %d 条，以下列出前 %d 条，格式：\n"
            "名称｜大类｜方向｜强度｜可信度｜周期｜数值｜理论依据"
            % (len(fs), len(rows)))
    return head + "\n" + "\n".join(rows)


FACTOR_EVAL_PROMPT = """你是A股量化因子审查员。用户会给出一份因子清单，
每条含：名称｜大类｜方向｜强度｜可信度｜周期｜数值｜理论依据。

请逐条评价**这条因子在此刻是否合理**，重点看：
1. 方向与数值是否自洽（例如"低位放量不跌"却给了强空，就是矛盾）
2. 强度/可信度是否与其理论可靠度相称（如"缺口理论"不该给 0.9 可信度）
3. 周期错配（如用单日量能给"未来1年"的方向）
4. 大类之间的重复计算（同一信息被多个因子反复计入）

输出格式（严格）：
### 一、明显不合理的因子（附理由，最多 8 条）
### 二、方向可疑但证据不足的（最多 8 条）
### 三、大类内部的重复计入
### 四、整体因子集的问题

严格要求：**不要编造清单里没有的因子或数字**；不确定就说"证据不足"。中文，700 字内。"""


ANALOG_PROMPT = """你是A股历史行情研究员。用户会给出当前市场状态的关键特征。
请找出历史上**形态相似**的时期，并说明后来怎么走。

要求：
1. 明确标注这是**基于你的训练知识的近似类比**，不是精确匹配
2. 每个类比给出：时期（年月）、相似点、关键差异、后续走势、当年结论
3. **必须指出不相似之处**——类比最容易犯的错就是说"这次一样"
4. 最多 3 个类比，不要凑数；确实想不出就说"没有足够相似的案例"
5. 结尾给出：这些类比**不足以支撑什么结论**

严禁编造具体点位与涨跌幅数字；用"约""量级上"等表述。
中文，700 字内。"""


def review_factors(payload: dict, *, top_n: int = 40,
                   use_cache: bool = True) -> dict:
    """让 AI 逐条评价因子是否合理（只输出文字，不修改任何因子）。"""
    if not available():
        return {"ok": False, "text": "", "error": "未配置 API Key",
                "model": "", "elapsed_ms": 0, "facts": ""}
    facts = summarize_factors(payload, top_n=top_n)
    if not facts or facts.startswith("（无因子"):
        return {"ok": False, "text": "", "error": "无因子数据",
                "model": "", "elapsed_ms": 0, "facts": facts}
    key = hash(("factors", facts))
    with _LOCK:
        got = _CACHE.get(key)
        if use_cache and got and (time.time() - got[0]) < _CACHE_TTL:
            return {**got[1], "cached": True}
    r = chat([{"role": "system", "content": FACTOR_EVAL_PROMPT},
              {"role": "user", "content": facts}])
    r["facts"] = facts
    if r.get("ok"):
        with _LOCK:
            _CACHE[key] = (time.time(), r)
    return r


def review_analog(payload: dict, *, use_cache: bool = True) -> dict:
    """让 AI 找历史相似案例（近似类比，不做精确匹配）。"""
    if not available():
        return {"ok": False, "text": "", "error": "未配置 API Key",
                "model": "", "elapsed_ms": 0, "facts": ""}
    facts = summarize_payload(payload)
    key = hash(("analog", facts))
    with _LOCK:
        got = _CACHE.get(key)
        if use_cache and got and (time.time() - got[0]) < _CACHE_TTL:
            return {**got[1], "cached": True}
    r = chat([{"role": "system", "content": ANALOG_PROMPT},
              {"role": "user", "content": facts}])
    r["facts"] = facts
    if r.get("ok"):
        with _LOCK:
            _CACHE[key] = (time.time(), r)
    return r


# --------------------------------------------------------------------------- #
# 三、缺失数据的"线索"（不写入模型）
# --------------------------------------------------------------------------- #
FILL_PROMPT = """你熟悉中国金融市场数据的公开披露渠道。用户想找一个指标的数值，
但程序拿不到。请给出：

1. **最新可得数值**（如果你确实知道）——必须注明数据期间（如"2026年9月"）
   和来源渠道。若不确定，直接说"不确定"，**不要编造**。
2. **权威查询路径**：具体到网站栏目名。
3. **口径提醒**：这个指标有几种口径，容易混淆的是什么。

用中文，300 字内，结构化输出。"""


def suggest(field: str, *, note: str = "") -> dict:
    """请 AI 给出某个缺失指标的线索（**仅供人工核对，不进模型**）。"""
    if not available():
        return {"ok": False, "text": "", "error": "未配置 API Key", "field": field}
    r = chat([{"role": "system", "content": FILL_PROMPT},
              {"role": "user", "content": "指标：%s\n背景：%s" % (field, note)}])
    r["field"] = field
    r["disclaimer"] = ("⚠️ AI 推测，**未经验证**。请到官方渠道核对后再手动录入，"
                       "不会自动参与打分。")
    return r


if __name__ == "__main__":  # pragma: no cover
    import tempfile
    import warnings

    warnings.filterwarnings("ignore")

    # ---- 自测隔离配置 ----
    # 用户可能已在界面配置了真实 Key，直接跑会真的发起网络请求、
    # 也会让"未配置时安全降级"的断言失效。这里切到临时路径 + 清环境变量。
    # 配置读写已改由 core/local_store 负责，隔离也要改到那一层：
    # 临时把 local_store.load 换成「永远空」、save 换成 no-op，
    # 自测既读不到用户真实 Key，也不会写坏真实配置。
    _ORIG_ENV = os.environ.pop("STOCKAI_AI_KEY", None)
    import core.local_store as _LS

    _ORIG_LOAD, _ORIG_SAVE = _LS.load, _LS.save
    _LS.load = lambda name: {}
    _LS.save = lambda name, data: True

    FAKE = {
        "name": "上证指数", "as_of": "2026-10-24", "price": 3690.47,
        "market": {
            "structure": {
                "regime": {"regime": "牛市", "from_low_pct": 36.6, "from_high_pct": -13.0},
                "phase": {"label": "熊初（出清段）", "confidence": 0.33,
                          "gates": ["未站上年线", "未站上季线(MA60)"]},
                "deep": {"compose": {"total": 0.36, "lean": "偏空",
                                     "dims": {"driver": 0.272, "structure": 0.38,
                                              "position": 0.121, "volume": 0.758}},
                         "fine": {"stage": "下跌趋势中的弱反弹（B 浪）",
                                  "wave": "B 浪弱反弹",
                                  "rally_pct": 0.9, "pullback_ratio": 0.09}},
            },
            "macro_env": {"indicators": {
                "pmi": {"value": 50.1, "date": "2026-09-01"},
                "cpi": {"value": 0.9, "date": "2026-09-01"},
                "ppi": {"value": 2.3, "date": "2026-09-01"},
                "m2": {"value": 7.2, "date": "2026-09-01"},
                "lpr": {"value": 3.0, "date": "2026-10-20"}}, "stale": []},
            "market_fundamental": {
                "pe": {"pe": 29.32, "percentile": 25.5},
                "margin": {"balance": 24026.49, "change_20d_pct": -7.94},
                "north": {"sum_20d": -279.94, "positive_days_20": 5}},
            "retail_heat": {"score": 32.9, "level": "偏冷",
                            "proxies": {"两融20日变化%": -7.94}},
            "rotation": {"board_breadth_pct": 91.8, "board_count": 90,
                         "breadth_quality": "完整"},
            "drivers": {"intent": "待定", "n_evidence": 1,
                        "items": [{"name": "电子信息"}]},
            "index_forecasts": {"sh": [
                {"horizon": "d1", "p_up": 0.40, "confidence": 0.33},
                {"horizon": "d5", "p_up": 0.41, "confidence": 0.33},
                {"horizon": "d20", "p_up": 0.42, "confidence": 0.33},
                {"horizon": "q1", "p_up": 0.49, "confidence": 0.30}]},
        },
    }

    print("=== 1. 未配置 Key 时安全降级 ===")
    st = status()
    print("  status:", st["detail"])
    assert st["available"] == available()
    r = review(FAKE)
    assert r["ok"] is False and "未配置" in r["error"]
    assert suggest("新增开户数")["ok"] is False
    print("  未配置时返回错误而不抛异常 ✓")

    print("\n=== 2. 结论摘要（给 AI 的事实清单）===")
    facts = summarize_payload(FAKE)
    print(facts[:700])
    for kw in ("四维评分", "熊初", "指数上涨概率", "宏观", "散户入场热度",
               "主要动能对象", "数据缺口"):
        assert kw in facts, "摘要缺少 %s" % kw
    print("\n  摘要含全部关键区块 ✓  字符数:", len(facts))

    print("\n=== 3. 摘要必须标注代理指标，不能冒充开户数 ===")
    assert "代理指标" in facts and "非开户数" in facts
    print("  已标注「代理指标，非开户数」 ✓")

    print("\n=== 4. 空 payload 安全 ===")
    assert summarize_payload(None)
    assert summarize_payload({})
    print("  空输入安全 ✓")

    print("\n=== 5. 因子清单（供逐条评价）===")
    FAKE2 = dict(FAKE)
    FAKE2["factors_all"] = [
        {"name": "低位放量不跌", "category": "technical", "direction": 1,
         "strength": 0.6, "reliability": 0.7, "horizon": ["d1", "d5"],
         "value": "量比1.5", "theory": "量价关系"},
        {"name": "均线空头排列", "category": "technical", "direction": -1,
         "strength": 0.8, "reliability": 0.72, "horizon": ["d5", "d20"],
         "value": "MA20<MA60<MA250", "theory": "均线系统"},
        {"name": "缺口理论强多", "category": "technical", "direction": 1,
         "strength": 0.9, "reliability": 0.9, "horizon": ["y1"],
         "value": "向上跳空", "theory": "缺口理论"},
    ]
    ff = summarize_factors(FAKE2)
    print(ff)
    assert "低位放量不跌" in ff and "均线空头排列" in ff
    assert "名称｜大类｜方向" in ff, "表头格式不符"
    print("  因子清单含表头与前 3 条 ✓")

    print("\n=== 6. 逐条评因子 / 历史类比：未配置时安全降级 ===")
    assert review_factors(FAKE2)["ok"] is False
    assert review_analog(FAKE)["ok"] is False
    # 未配置 Key 时先返回"未配置"（检查顺序），配置后才轮到"无因子数据"
    _rf = review_factors({})
    assert _rf["ok"] is False and _rf.get("error")
    print("  三个复核入口均安全降级 ✓")

    print("\n=== 7. 缓存键稳定 ===")
    a = summarize_payload(FAKE)
    b = summarize_payload(FAKE)
    assert a == b
    print("  同一 payload 摘要一致 ✓")

    # ---- 恢复真实配置（自测会临时替换 local_store 的读写）----
    _LS.load, _LS.save = _ORIG_LOAD, _ORIG_SAVE
    if _ORIG_ENV is not None:
        os.environ["STOCKAI_AI_KEY"] = _ORIG_ENV

    print("\nai_analyst self-test OK")

# --------------------------------------------------------------------------- #
# 四、AI 检索权威最新「事件/政策」信息（按当前分析范围，仅供人工参考）
# --------------------------------------------------------------------------- #
#: 各核对对象的检索焦点
_EVENT_FOCUS = {
    "market": ("大盘层面的宏观与政策事件：中央/政治局会议、两会与重要文件、"
               "货币政策（降准降息/MLF/LPR）、财政与产业政策、"
               "监管取向（IPO/减持/量化）、地缘与外部冲击（美联储/汇率/关税）"),
    "board": ("行业与板块层面的政策与事件：产业扶持或限制政策、"
              "补贴与税收优惠、产能治理与价格调控、"
              "行业数据与景气拐点、重大订单或技术突破"),
    "subject": ("个股与所属行业层面的事件：公司公告（业绩预告/重大合同/回购增持/"
                "股权变动/诉讼）、行业政策、监管处罚、"
                "上下游价格与供需变化"),
}

EVENT_PROMPT = """你是A股信息核查助手。用户会告诉你他正在分析的**范围与标的**，
请检索你知识范围内**与该范围直接相关的、权威渠道发布的最新主要事件/政策信息**。

硬性要求：
1. 只给**事件**，不要给行情预测、不要给买卖建议、不要编造数值。
2. 每条必须包含四项：**事件** ｜ **时间（年月，尽量精确）** ｜
   **来源（机构或媒体名称，如 国务院/央行/证监会/统计局/新华社）** ｜
   **对该范围的可能影响方向（利多/利空/中性，不量化）**。
3. 与范围无关的宏观事件不要凑数；**宁少勿滥，最多 6 条**。
4. 你不确定的事必须写「不确定」；无法确认时间就写「时间待核实」。
5. 如果你的知识截止时间早于用户当前时间，必须在结尾明确写出
   「我的知识可能已过期，请以官方网站最新发布为准」。
6. 结尾单独一段「需要人工核实的地方」，列出你认为需要用户去官网
   确认的条目。

输出格式：
### 一、与当前范围相关的最新事件（最多 6 条）
| 事件 | 时间 | 来源 | 影响方向 |
### 二、需要人工核实的地方

中文，600 字内。"""


def latest_events(scope: str = "market", *, target: str = "",
                  boards: list | None = None, use_cache: bool = True) -> dict:
    """检索与当前分析范围相关的权威最新事件（**只展示，不入模型**）。

    Parameters
    ----------
    scope : ``market`` / ``board`` / ``subject``
    target : 个股/基金名称（scope=subject 时用）
    boards : 当前领涨板块名（scope=board 时用，作为行业线索）
    """
    if not available():
        return {"ok": False, "text": "", "error": "未配置 API Key", "model": ""}
    focus = _EVENT_FOCUS.get(scope) or _EVENT_FOCUS["market"]
    ctx = ["分析范围：%s" % {"market": "大盘", "board": "板块",
                            "subject": "个股/基金"}.get(scope, scope)]
    if scope == "subject" and target:
        ctx.append("标的：%s" % target)
    if scope == "board" and boards:
        ctx.append("当前领涨/关注板块：%s" % "、".join(str(b) for b in boards[:8]))
    # 附带程序当前的状态判断，让 AI 知道"在什么背景下找事件"
    try:
        from . import market_source as _MS  # 不存在则跳过
    except Exception:  # noqa: BLE001
        pass
    user = ("请围绕下面的范围与关注点检索最新事件：\n"
            + "\n".join(ctx)
            + "\n\n关注点：%s\n" % focus
            + "\n再次强调：只给事件与来源时间，不要预测价格，不要编造。")
    key = hash(("events", scope, target, tuple(boards or ()), focus))
    with _LOCK:
        got = _CACHE.get(key)
        if use_cache and got and (time.time() - got[0]) < _CACHE_TTL:
            return {**got[1], "cached": True}
    r = chat([{"role": "system", "content": EVENT_PROMPT},
              {"role": "user", "content": user}], max_tokens=1000)
    r["disclaimer"] = ("AI 检索结果仅供人工参考，**不写入任何模型数据**，"
                       "也不参与概率计算。事件时间与来源可能记错，"
                       "请以官方网站最新发布为准。")
    r["scope"], r["target"] = scope, target
    if r.get("ok"):
        with _LOCK:
            _CACHE[key] = (time.time(), r)
    return r


def macro_news(*, use_cache: bool = True) -> dict:
    """兼容旧接口：等价于检索大盘层面的最新政策/宏观事件。"""
    return latest_events("market", use_cache=use_cache)

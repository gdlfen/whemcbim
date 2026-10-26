# -*- coding: utf-8 -*-
"""
云端「每人一份、刷新不失效」的私有数据持久化。

可行性说明（为什么用 URL）
--------------------------
浏览器端持久化有三种手段，Streamlit 只支持其中一种：
  · localStorage  ——需要前端 JS 组件回传，纯 Streamlit 做不到（要加依赖）
  · cookie        ——Streamlit 只能读（st.context.cookies），不能写
  · **URL 查询参数** ——可读可写（st.query_params），刷新后仍在 ✅

所以云端模式改为把私有数据编码进 URL：
    ?pd_ai_config=<base64>  &pd_macro_manual=<base64>

这样得到的效果：
  ✅ **每人一份**：数据在你自己的浏览器地址栏里，服务器不保存
  ✅ **互不可见**：不同访客的 URL 不同，天然隔离
  ✅ **刷新不失效**：刷新页面 URL 不变，数据仍能读回
  ✅ **可修改、可清空**：保存即改写 URL；清空即删除参数

必须告知的风险（界面会提示）
----------------------------
API Key 会出现在浏览器地址栏与历史记录里。因此：
  · **不要把这个带参数的链接分享给别人**（等于分享 Key）
  · 公用电脑上用完请点「清除」
  · 若不想让 Key 进 URL，可在本地运行（本地落盘、不进 URL）

线程安全
--------
后台分析线程里不能访问 st.query_params，所以这里维护一份进程内缓存，
每次脚本运行开头调用 :func:`sync_from_query` 把 URL 刷进缓存；
非 UI 线程只读缓存。
"""
from __future__ import annotations

import base64
import json
import os
import threading

from .utils import get_logger

log = get_logger()

_LOCK = threading.RLock()

#: 进程内缓存：{name: dict}（UI 线程从 URL 同步，其它线程只读）
_MEM: dict = {}

#: URL 参数前缀
_PREFIX = "pd_"

#: 云端模式下**是否允许把 API Key 写进 URL**（可由界面开关控制）
_ALLOW_KEY_IN_URL = True


def is_local() -> bool:
    """当前是否本地部署（可以安全落盘）。"""
    if os.environ.get("STREAMLIT_CLOUD") or os.environ.get("IS_STREAMLIT_CLOUD"):
        return False
    try:
        p = os.path.abspath(__file__).replace("\\", "/")
        if p.startswith("/mount/src"):
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def set_allow_key_in_url(flag: bool) -> None:
    global _ALLOW_KEY_IN_URL
    _ALLOW_KEY_IN_URL = bool(flag)


def allow_key_in_url() -> bool:
    return _ALLOW_KEY_IN_URL


def _base_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")


def path_of(name: str) -> str:
    return os.path.join(_base_dir(), name)


def _enc(data: dict) -> str:
    raw = json.dumps(data or {}, ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _dec(s: str) -> dict:
    try:
        pad = "=" * (-len(s) % 4)
        raw = base64.urlsafe_b64decode(s + pad)
        d = json.loads(raw.decode("utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _qp():
    """拿到 st.query_params；非 UI 线程返回 None。"""
    try:
        import streamlit as st

        return st.query_params
    except Exception:  # noqa: BLE001
        return None


def sync_from_query() -> None:
    """把 URL 参数同步到进程内缓存（每次脚本运行开头调用一次）。"""
    if is_local():
        return
    qp = _qp()
    if qp is None:
        return
    try:
        items = dict(qp)
    except Exception:  # noqa: BLE001
        return
    for k, v in items.items():
        if not str(k).startswith(_PREFIX):
            continue
        name = str(k)[len(_PREFIX):]
        val = v[0] if isinstance(v, (list, tuple)) and v else v
        if val:
            with _LOCK:
                _MEM[name] = _dec(str(val))


def load(name: str) -> dict:
    """读取私有数据（云端只从 URL/内存读，服务器不保存）。"""
    if is_local():
        p = path_of(name)
        try:
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    d = json.load(fh)
                return d if isinstance(d, dict) else {}
        except Exception as exc:  # noqa: BLE001
            log.debug("读私有数据失败 %s: %s", name, exc)
        return {}
    with _LOCK:
        got = _MEM.get(name)
    if got is not None:
        return dict(got)
    # 缓存未命中时尝试直接从 URL 读一次（UI 线程）
    qp = _qp()
    if qp is not None:
        try:
            val = qp.get(_PREFIX + name)
            if val:
                d = _dec(str(val))
                with _LOCK:
                    _MEM[name] = d
                return d
        except Exception:  # noqa: BLE001
            pass
    return {}


def save(name: str, data: dict) -> bool:
    """保存私有数据：本地落盘；云端写进 URL（刷新不失效、每人一份）。"""
    data = dict(data or {})
    if is_local():
        try:
            os.makedirs(_base_dir(), exist_ok=True)
            p = path_of(name)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            try:
                os.chmod(p, 0o600)
            except Exception:  # noqa: BLE001
                pass
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("写私有数据失败 %s: %s", name, exc)
            return False
    # ---- 云端：写 URL ----
    with _LOCK:
        _MEM[name] = data
    # Key 是否允许进 URL
    if name == "ai_config.json" and not _ALLOW_KEY_IN_URL:
        data = {k: v for k, v in data.items() if k != "key"}
    qp = _qp()
    if qp is None:
        return False
    try:
        if data:
            qp[_PREFIX + name] = _enc(data)
        else:
            try:
                del qp[_PREFIX + name]
            except Exception:  # noqa: BLE001
                pass
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("写 URL 参数失败 %s: %s", name, exc)
        return False


def clear(name: str) -> bool:
    """清空某项私有数据（云端同时从 URL 移除参数）。"""
    with _LOCK:
        _MEM.pop(name, None)
    if is_local():
        try:
            p = path_of(name)
            if os.path.exists(p):
                os.remove(p)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("清除私有数据失败 %s: %s", name, exc)
            return False
    qp = _qp()
    if qp is None:
        return False
    try:
        del qp[_PREFIX + name]
        return True
    except Exception:  # noqa: BLE001
        return True


def describe(name: str) -> str:
    """给界面用的存储位置说明。"""
    if is_local():
        return "仅保存在本机：%s（不会上传到任何地方）" % path_of(name)
    if name == "ai_config.json" and not _ALLOW_KEY_IN_URL:
        return ("**仅保存在当前会话内存**（未启用 URL 保存）：刷新后需重填。"
                "如需跨刷新保存，请在下方打开「把 Key 存进浏览器地址栏」。")
    return ("保存在**你的浏览器地址栏**（`%s%s`）：服务器不保存、只属于你、"
            "刷新不失效。⚠️ **请勿把带参数的链接分享给他人**。"
            % (_PREFIX, name))


def ensure_gitignored() -> None:
    try:
        gi = os.path.join(os.path.dirname(_base_dir()), ".gitignore")
        if not os.path.exists(gi):
            return
        txt = open(gi, encoding="utf-8").read()
        need = [n for n in ("data/ai_config.json", "data/macro_manual.json")
                if n not in txt]
        if need:
            with open(gi, "a", encoding="utf-8") as fh:
                fh.write("\n# 设备私有数据，禁止提交\n")
                for n in need:
                    fh.write(n + "\n")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")

    print("=== 1. 编解码往返 ===")
    d = {"value": 1.5, "中文": "测试", "note": "含 emoji 与特殊字符 &?=/"}
    enc = _enc(d)
    dec = _dec(enc)
    print("  URL 片段长度:", len(enc))
    assert dec == d, "编解码不一致"
    print("  往返一致 ✓")

    print("\n=== 2. 本地模式落盘 ===")
    print("  is_local:", is_local())
    save("__selftest__.json", {"a": 1})
    assert load("__selftest__.json").get("a") == 1
    print("  读写正常 ✓")

    print("\n=== 3. 清空 ===")
    clear("__selftest__.json")
    assert not os.path.exists(path_of("__selftest__.json"))
    print("  已清除且文件删除 ✓")

    print("\n=== 4. 云端模式（无 st 上下文）===")
    _orig = os.environ.get("STREAMLIT_CLOUD")
    os.environ["STREAMLIT_CLOUD"] = "1"
    try:
        assert is_local() is False
        ok = save("__cloud__.json", {"b": 2})
        print("  save 返回:", ok, "（无 UI 上下文时预期 False）")
        assert load("__cloud__.json").get("b") == 2, "内存缓存应可读回"
        print("  内存缓存读回 ✓；未生成文件:", not os.path.exists(path_of("__cloud__.json")))
    finally:
        if _orig is None:
            os.environ.pop("STREAMLIT_CLOUD", None)
        else:
            os.environ["STREAMLIT_CLOUD"] = _orig

    print("\n=== 5. describe ===")
    print("  本地:", describe("ai_config.json")[:60])
    os.environ["STREAMLIT_CLOUD"] = "1"
    try:
        print("  云端:", describe("ai_config.json")[:80])
    finally:
        os.environ.pop("STREAMLIT_CLOUD", None)

    print("\nlocal_store self-test OK")

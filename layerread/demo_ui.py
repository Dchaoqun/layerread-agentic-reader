"""Streamlit-only helpers for the hosted LayerRead demo."""

from __future__ import annotations

import uuid

import streamlit as st

from layerread.demo import DemoLimiter, DemoPolicy


DEMO_BUILD_ID = "2026-08-20.3"


_VISITOR_JS = """
export default function(component) {
  const { data, setStateValue } = component
  const storageKey = "layerread_demo_visitor_id"
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
  let visitorId = ""
  try {
    const stored = globalThis.localStorage?.getItem(storageKey) || ""
    visitorId = uuidPattern.test(stored) ? stored : (data?.fallback_id || "")
    if (uuidPattern.test(visitorId)) {
      globalThis.localStorage?.setItem(storageKey, visitorId)
      setStateValue("visitor_id", visitorId)
    }
  } catch (_error) {
    visitorId = data?.fallback_id || ""
    if (uuidPattern.test(visitorId)) setStateValue("visitor_id", visitorId)
  }
}
"""


_VISITOR_COMPONENT = st.components.v2.component(
    "layerread_demo_visitor_identity",
    html="<span aria-hidden=\"true\"></span>",
    js=_VISITOR_JS,
)


def ensure_demo_visitor_id() -> str:
    """Return a stable anonymous browser id with a per-session fallback."""

    fallback = st.session_state.get("demo_visitor_id")
    if not isinstance(fallback, str) or not _is_uuid4(fallback):
        fallback = str(uuid.uuid4())
        st.session_state.demo_visitor_id = fallback
    result = _VISITOR_COMPONENT(
        key="layerread_demo_visitor_component",
        data={"fallback_id": fallback},
        on_visitor_id_change=_noop,
        height=0,
    )
    stored = getattr(result, "visitor_id", "")
    if isinstance(stored, str) and _is_uuid4(stored):
        st.session_state.demo_visitor_id = stored
        return stored
    return fallback


def build_demo_limiter() -> DemoLimiter:
    return DemoLimiter(DemoPolicy.from_env(), ensure_demo_visitor_id())


def render_demo_sidebar(limiter: DemoLimiter) -> None:
    with st.sidebar:
        st.header("在线体验")
        analysis = limiter.status("analysis")
        chat = limiter.status("chat")
        st.caption(
            f"今日剩余：文章分析 {analysis.remaining} 次 · Chat {chat.remaining} 轮"
        )
        st.caption("正文、图片、分析和对话只保留在当前页面会话，不进入文章库。")
        st.caption(f"在线构建：{DEMO_BUILD_ID}")
        with st.expander("遇到问题？"):
            st.caption(
                "向维护者反馈配额或模型错误时，请附上体验支持码。"
                "它只来自随机匿名访客编号，不包含文章或身份信息。"
            )
            st.code(limiter.visitor_hash[:12], language=None)


def _is_uuid4(value: str) -> bool:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return parsed.version == 4 and str(parsed) == value.lower()


def _noop() -> None:
    return None

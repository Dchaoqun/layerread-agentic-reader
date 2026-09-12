"""Streamlit Custom Component v2 bridge for the LayerRead Chrome connector."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st


_CONNECTOR_HTML = """
<div class="connector-card">
  <span class="status-dot" aria-hidden="true"></span>
  <div class="connector-copy">
    <strong id="connector-title">正在检测 LayerRead Connector…</strong>
    <span id="connector-detail">如果刚从公众号文章跳转，请稍候。</span>
    <div class="connector-progress" aria-hidden="true"><span></span></div>
  </div>
  <button id="connector-retry" type="button">重新检测</button>
</div>
"""

_CONNECTOR_CSS = """
.connector-card {
  align-items: center;
  background: var(--st-secondary-background-color);
  border: 1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent);
  border-radius: var(--st-border-radius, 0.5rem);
  display: flex;
  gap: 0.75rem;
  padding: 0.85rem 1rem;
}
.status-dot {
  background: #d97706;
  border-radius: 999px;
  flex: 0 0 auto;
  height: 0.65rem;
  width: 0.65rem;
}
.status-dot.success { background: #16a34a; }
.status-dot.error { background: #dc2626; }
.connector-copy {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 0.15rem;
  min-width: 0;
}
.connector-copy strong { color: var(--st-text-color); }
.connector-copy span {
  color: color-mix(in srgb, var(--st-text-color) 70%, transparent);
  font-size: 0.88rem;
}
.connector-progress {
  background: color-mix(in srgb, var(--st-text-color) 12%, transparent);
  border-radius: 999px;
  height: 0.3rem;
  margin-top: 0.35rem;
  overflow: hidden;
  position: relative;
  width: 100%;
}
.connector-progress span {
  animation: connector-progress 1.15s ease-in-out infinite;
  background: var(--st-primary-color);
  border-radius: inherit;
  height: 100%;
  left: -35%;
  position: absolute;
  width: 35%;
}
.connector-progress.success span {
  animation: none;
  background: #16a34a;
  left: 0;
  width: 100%;
}
.connector-progress.error span {
  animation: none;
  background: #dc2626;
  left: 0;
  width: 100%;
}
@keyframes connector-progress {
  from { left: -35%; }
  to { left: 100%; }
}
button {
  background: transparent;
  border: 1px solid var(--st-primary-color);
  border-radius: var(--st-border-radius, 0.5rem);
  color: var(--st-primary-color);
  cursor: pointer;
  font: inherit;
  padding: 0.4rem 0.7rem;
}
button:hover { background: color-mix(in srgb, var(--st-primary-color) 10%, transparent); }
"""

_CONNECTOR_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component
  const title = parentElement.querySelector("#connector-title")
  const detail = parentElement.querySelector("#connector-detail")
  const dot = parentElement.querySelector(".status-dot")
  const progress = parentElement.querySelector(".connector-progress")
  const retry = parentElement.querySelector("#connector-retry")
  if (!title || !detail || !dot || !progress || !retry) return

  const tokenPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
  const token = typeof data?.import_token === "string" ? data.import_token : ""
  const ackToken = typeof data?.ack_import_token === "string" ? data.ack_import_token : ""
  const requestId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`
  let timeoutId = null
  const missingConnectorVersions = new Set()

  function setStatus(kind, heading, message) {
    dot.classList.remove("success", "error")
    progress.classList.remove("success", "error")
    if (kind) dot.classList.add(kind)
    if (kind) progress.classList.add(kind)
    title.textContent = heading
    detail.textContent = message
  }

  function removeImportToken(targetWindow) {
    try {
      const cleanUrl = new URL(targetWindow.location.href)
      if (!cleanUrl.searchParams.has("layerread_import")) return
      cleanUrl.searchParams.delete("layerread_import")
      targetWindow.history.replaceState({}, "", cleanUrl.toString())
    } catch (_error) {
      // The app still works if an embedding page prevents history access.
    }
  }

  function clearImportToken() {
    removeImportToken(window)
    if (window.top && window.top !== window) removeImportToken(window.top)
  }

  function requestConnector() {
    const type = tokenPattern.test(token) ? "LAYERREAD_REQUEST_IMPORT" : "LAYERREAD_CONNECTOR_PING"
    const isImport = type === "LAYERREAD_REQUEST_IMPORT"
    retry.textContent = isImport ? "重新接收" : "重新检测"
    setStatus(
      "",
      isImport ? "正在领取文章包…" : "正在检测 LayerRead Connector…",
      isImport ? "正文和高清图片正在安全传入 LayerRead。" : "请保持这个页面打开。",
    )
    window.postMessage(
      { source: "layerread-app", type, request_id: requestId, import_id: token },
      window.location.origin,
    )
    clearTimeout(timeoutId)
    timeoutId = window.setTimeout(() => {
      setStatus(
        "error",
        isImport ? "文章接收超时" : "未检测到 LayerRead Connector",
        isImport
          ? "文章包可能较大，或扩展连接已中断。请点击“重新接收”。"
          : "请先安装并启用项目中的 Chrome 扩展，然后重新检测。",
      )
    }, isImport ? 15000 : 2500)
  }

  function acknowledgeImport() {
    if (!tokenPattern.test(ackToken)) return
    window.postMessage(
      {
        source: "layerread-app",
        type: "LAYERREAD_ACK_IMPORT",
        request_id: requestId,
        import_id: ackToken,
      },
      window.location.origin,
    )
  }

  function handleMessage(event) {
    if (event.source !== window || event.origin !== window.location.origin) return
    const message = event.data
    if (!message || message.source !== "layerread-connector") return
    if (message.request_id !== requestId) return
    if (message.type === "LAYERREAD_CONNECTOR_AVAILABLE") {
      clearTimeout(timeoutId)
      const version = typeof message.connector_version === "string" ? message.connector_version : ""
      setStatus(
        "success",
        "LayerRead Connector 已连接",
        version
          ? `已检测到版本 ${version}。打开微信公众号文章并点击扩展图标，即可一键发送到这里。`
          : "打开微信公众号文章并点击扩展图标，即可一键发送到这里。",
      )
      return
    }
    if (
      message.type === "LAYERREAD_IMPORT_ACKNOWLEDGED"
      && message.ok
      && message.import_id === ackToken
    ) {
      setTriggerValue("acknowledged", ackToken)
      return
    }
    if (message.type !== "LAYERREAD_IMPORT_RESULT") return
    if (!message.ok) {
      const version = typeof message.connector_version === "string" ? message.connector_version : "未知版本"
      missingConnectorVersions.add(version)
      setStatus(
        "",
        "仍在等待正确的 Connector 实例…",
        `版本 ${[...missingConnectorVersions].join("、")} 没有这个文章包。若安装过多个版本，请只保留当前版本。`,
      )
      return
    }

    clearTimeout(timeoutId)
    clearImportToken()
    setStatus("", "文章包已找到", "正在交给 LayerRead 校验并生成预览…")
    setTriggerValue("received", message.payload)
  }

  window.addEventListener("message", handleMessage)
  retry.onclick = requestConnector
  acknowledgeImport()
  requestConnector()

  return () => {
    clearTimeout(timeoutId)
    window.removeEventListener("message", handleMessage)
    retry.onclick = null
  }
}
"""


_CONNECTOR_COMPONENT = st.components.v2.component(
    "layerread_connector_bridge",
    html=_CONNECTOR_HTML,
    css=_CONNECTOR_CSS,
    js=_CONNECTOR_JS,
)


def render_connector_bridge(
    *,
    key: str,
    import_token: str = "",
    ack_import_token: str = "",
    on_received_change: Callable[[], None] | None = None,
    on_unavailable_change: Callable[[], None] | None = None,
    on_acknowledged_change: Callable[[], None] | None = None,
):
    """Mount the connector bridge and return its component result."""

    return _CONNECTOR_COMPONENT(
        key=key,
        data={
            "import_token": import_token,
            "ack_import_token": ack_import_token,
        },
        on_received_change=on_received_change or (lambda: None),
        on_unavailable_change=on_unavailable_change or (lambda: None),
        on_acknowledged_change=on_acknowledged_change or (lambda: None),
    )

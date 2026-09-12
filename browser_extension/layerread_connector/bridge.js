const LAYERREAD_HOSTS = new Set(["localhost", "127.0.0.1"])
const ONLINE_LAYERREAD_ORIGIN = globalThis.LAYERREAD_ONLINE_ORIGIN || ""
const TOKEN_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
const inFlight = new Set()
const CONNECTOR_VERSION = chrome.runtime.getManifest().version

function isAllowedPage() {
  const isLocal = location.protocol === "http:" && LAYERREAD_HOSTS.has(location.hostname) && location.port === "8501"
  const isOnline = Boolean(ONLINE_LAYERREAD_ORIGIN) && location.origin === ONLINE_LAYERREAD_ORIGIN
  return isLocal || isOnline
}

if (isAllowedPage()) {
  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== location.origin) return
    const message = event.data
    if (!message || message.source !== "layerread-app") return
    if (typeof message.request_id !== "string" || message.request_id.length > 100) return

    if (message.type === "LAYERREAD_CONNECTOR_PING") {
      chrome.runtime.sendMessage({ type: "LAYERREAD_CONNECTOR_PING" }, (response) => {
        if (chrome.runtime.lastError || !response?.ok) return
        window.postMessage({
          source: "layerread-connector",
          type: "LAYERREAD_CONNECTOR_AVAILABLE",
          request_id: message.request_id,
          connector_version: CONNECTOR_VERSION,
        }, location.origin)
      })
      return
    }

    if (message.type === "LAYERREAD_ACK_IMPORT") {
      const importId = message.import_id
      if (typeof importId !== "string" || !TOKEN_PATTERN.test(importId)) return
      chrome.runtime.sendMessage({
        type: "LAYERREAD_ACK_IMPORT",
        import_id: importId,
      }, (response) => {
        const error = chrome.runtime.lastError?.message
        window.postMessage({
          source: "layerread-connector",
          type: "LAYERREAD_IMPORT_ACKNOWLEDGED",
          request_id: message.request_id,
          import_id: importId,
          ok: Boolean(response?.ok) && !error,
          connector_version: CONNECTOR_VERSION,
        }, location.origin)
      })
      return
    }

    if (message.type !== "LAYERREAD_REQUEST_IMPORT") return
    const importId = message.import_id
    if (typeof importId !== "string" || !TOKEN_PATTERN.test(importId) || inFlight.has(importId)) return
    inFlight.add(importId)
    chrome.runtime.sendMessage({
      type: "LAYERREAD_CLAIM_IMPORT",
      import_id: importId,
    }, (response) => {
      inFlight.delete(importId)
      const error = chrome.runtime.lastError?.message
      window.postMessage({
        source: "layerread-connector",
        type: "LAYERREAD_IMPORT_RESULT",
        request_id: message.request_id,
        ok: Boolean(response?.ok) && !error,
        payload: response?.payload,
        error: error || response?.error || "LayerRead Connector 没有返回文章。",
        connector_version: CONNECTOR_VERSION,
      }, location.origin)
    })
  })
}

importScripts("config.js")

const LOCAL_LAYERREAD_ORIGIN = "http://localhost:8501"
const ONLINE_LAYERREAD_ORIGIN = globalThis.LAYERREAD_ONLINE_ORIGIN || ""
const IMPORT_PREFIX = "layerread_import:"
const IMPORT_TTL_MS = 10 * 60 * 1000
const TOKEN_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

const MAX_SOURCE_IMAGES = 12
const MAX_OUTPUT_ITEMS = 24
const MAX_IMAGE_BYTES = 1_450_000
const MAX_TOTAL_IMAGE_BYTES = 6 * 1024 * 1024
const MAX_SESSION_ENVELOPE_BYTES = 9 * 1024 * 1024
const MAX_IMAGE_PIXELS = 40_000_000
const MAX_NORMAL_EDGE = 3200
const LONG_IMAGE_RATIO = 3
const TILE_TARGET_WIDTH = 2400
const TILE_TARGET_HEIGHT = 2400
const TILE_OVERLAP = 180
function storageKey(importId) {
  return `${IMPORT_PREFIX}${importId}`
}

function isLayerReadUrl(candidate) {
  try {
    const url = new URL(candidate)
    const isLocal = (
      url.protocol === "http:" &&
      (url.hostname === "localhost" || url.hostname === "127.0.0.1") &&
      url.port === "8501"
    )
    const isOnline = Boolean(ONLINE_LAYERREAD_ORIGIN) && url.origin === ONLINE_LAYERREAD_ORIGIN
    return isLocal || isOnline
  } catch (_error) {
    return false
  }
}

async function selectedLayerReadOrigin() {
  const { layerread_target: target } = await chrome.storage.local.get("layerread_target")
  if (target === "local" || !ONLINE_LAYERREAD_ORIGIN) return LOCAL_LAYERREAD_ORIGIN
  return ONLINE_LAYERREAD_ORIGIN
}

function isWeChatArticleUrl(candidate) {
  try {
    const url = new URL(candidate)
    return url.protocol === "https:" && url.hostname === "mp.weixin.qq.com"
  } catch (_error) {
    return false
  }
}

function isMissingContentScriptError(error) {
  const message = error instanceof Error ? error.message : String(error || "")
  return (
    message.includes("Could not establish connection") ||
    message.includes("Receiving end does not exist")
  )
}

async function extractArticleFromTab(tab) {
  const request = { type: "LAYERREAD_EXTRACT_ARTICLE" }
  try {
    return await chrome.tabs.sendMessage(tab.id, request)
  } catch (error) {
    if (!isMissingContentScriptError(error)) throw error
  }

  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"],
    })
    return await chrome.tabs.sendMessage(tab.id, request)
  } catch (_error) {
    throw new Error("扩展刚更新，当前公众号页面尚未连接。请刷新文章页面后再点击一次扩展图标。")
  }
}

function isLayerReadSender(sender) {
  return isLayerReadUrl(sender?.tab?.url || sender?.url || "")
}

async function openOrReuseLayerRead(importId) {
  const targetOrigin = await selectedLayerReadOrigin()
  const targetUrl = `${targetOrigin}/?layerread_import=${encodeURIComponent(importId)}`
  const patterns = targetOrigin === LOCAL_LAYERREAD_ORIGIN
    ? ["http://localhost/*", "http://127.0.0.1/*"]
    : [`${targetOrigin}/*`]
  const layerReadTabs = await chrome.tabs.query({ url: patterns })
  const candidates = layerReadTabs.filter((tab) => tab.id && isLayerReadUrl(tab.url || ""))
  const existing = candidates.find((tab) => tab.active) || candidates[0]
  if (!existing?.id) {
    await chrome.tabs.create({ url: targetUrl })
    return
  }

  await chrome.tabs.update(existing.id, { url: targetUrl, active: true })
  if (Number.isInteger(existing.windowId)) {
    await chrome.windows.update(existing.windowId, { focused: true }).catch(() => {})
  }
}

async function showBadge(tabId, text, color, title) {
  await chrome.action.setBadgeBackgroundColor({ tabId, color })
  await chrome.action.setBadgeText({ tabId, text })
  await chrome.action.setTitle({ tabId, title })
  setTimeout(() => {
    chrome.action.setBadgeText({ tabId, text: "" }).catch(() => {})
    chrome.action.setTitle({ tabId, title: "发送当前文章到 LayerRead" }).catch(() => {})
  }, 5000)
}

async function cleanExpiredImports() {
  const values = await chrome.storage.session.get(null)
  const now = Date.now()
  const expired = Object.entries(values)
    .filter(([key, value]) => key.startsWith(IMPORT_PREFIX) && (!value?.expires_at || value.expires_at <= now))
    .map(([key]) => key)
  if (expired.length) await chrome.storage.session.remove(expired)
}

async function removePendingImports() {
  const values = await chrome.storage.session.get(null)
  const pending = Object.keys(values).filter((key) => key.startsWith(IMPORT_PREFIX))
  if (pending.length) await chrome.storage.session.remove(pending)
}

function isSessionStorageQuotaError(error) {
  const message = error instanceof Error ? error.message : String(error || "")
  return /quota|QUOTA_BYTES|storage.*exceed/i.test(message)
}

async function storeImportEnvelope(importId, envelope) {
  // Only one unclaimed LayerRead import is useful. Removing older LayerRead entries
  // prevents a failed first hand-off from consuming the quota needed by a retry.
  await removePendingImports()
  let omittedImageItems = 0
  while (true) {
    try {
      await chrome.storage.session.set({ [storageKey(importId)]: envelope })
      return omittedImageItems
    } catch (error) {
      if (!isSessionStorageQuotaError(error)) throw error
      if (!envelope.article.images.length) {
        throw new Error(
          "文章正文超过 Chrome 临时传输容量；请改用正文粘贴，或选择内容较短的文章。",
        )
      }
      envelope.article.images.pop()
      omittedImageItems += 1
    }
  }
}

function bytesToDataUrl(bytes, mimeType) {
  let binary = ""
  const chunkSize = 0x8000
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize))
  }
  return `data:${mimeType};base64,${btoa(binary)}`
}

async function blobToDataUrl(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer())
  return bytesToDataUrl(bytes, blob.type)
}

function createCanvas(width, height) {
  const canvas = new OffscreenCanvas(Math.max(1, Math.round(width)), Math.max(1, Math.round(height)))
  const context = canvas.getContext("2d", { alpha: false })
  if (!context) throw new Error("浏览器无法创建图片处理画布。")
  context.fillStyle = "#ffffff"
  context.fillRect(0, 0, canvas.width, canvas.height)
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = "high"
  return { canvas, context }
}

async function encodeCanvas(canvas, preferPng) {
  if (preferPng) {
    const png = await canvas.convertToBlob({ type: "image/png" })
    if (png.size <= MAX_IMAGE_BYTES) return png
  }
  for (const quality of [0.94, 0.91, 0.88]) {
    const webp = await canvas.convertToBlob({ type: "image/webp", quality })
    if (webp.size <= MAX_IMAGE_BYTES) return webp
  }
  throw new Error("图片在保持文字清晰度的前提下仍超过安全上限。")
}

async function renderRegion(bitmap, region, outputWidth, outputHeight, preferPng) {
  const { canvas, context } = createCanvas(outputWidth, outputHeight)
  context.drawImage(
    bitmap,
    region.x,
    region.y,
    region.width,
    region.height,
    0,
    0,
    canvas.width,
    canvas.height,
  )
  const blob = await encodeCanvas(canvas, preferPng)
  return {
    blob,
    width: canvas.width,
    height: canvas.height,
  }
}

function tileRegions(width, height) {
  const scale = Math.min(1, TILE_TARGET_WIDTH / width)
  const sourceTileHeight = Math.max(1, Math.floor(TILE_TARGET_HEIGHT / scale))
  const sourceOverlap = Math.max(1, Math.floor(TILE_OVERLAP / scale))
  const step = Math.max(1, sourceTileHeight - sourceOverlap)
  const regions = []
  for (let top = 0; top < height; top += step) {
    const regionHeight = Math.min(sourceTileHeight, height - top)
    regions.push({ x: 0, y: top, width, height: regionHeight })
    if (top + regionHeight >= height) break
  }
  return { regions, scale }
}

function outputItem(candidate, encoded, itemId, tileIndex, tileCount, bitmap) {
  return {
    image_id: itemId,
    data_url: null,
    mime_type: encoded.blob.type,
    width: encoded.width,
    height: encoded.height,
    alt_text: candidate.alt_text || "",
    caption: candidate.caption || "",
    context_before: candidate.context_before || "",
    context_after: candidate.context_after || "",
    source_url: candidate.source_url.startsWith("https://") ? candidate.source_url : "",
    tile_index: tileIndex,
    tile_count: tileCount,
    original_width: bitmap.width,
    original_height: bitmap.height,
    byte_length: encoded.blob.size,
  }
}

async function prepareCandidateSafe(candidate) {
  try {
    // Keep the Blob only long enough to create its Data URL, then return a serializable item.
    const response = await fetch(candidate.source_url, { credentials: "omit", cache: "force-cache" })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const sourceBlob = await response.blob()
    const bitmap = await createImageBitmap(sourceBlob)
    try {
      if (!bitmap.width || !bitmap.height || bitmap.width * bitmap.height > MAX_IMAGE_PIXELS) return []
      const rendered = []
      const isLong = bitmap.height / bitmap.width >= LONG_IMAGE_RATIO && bitmap.height > MAX_NORMAL_EDGE
      if (isLong) {
        const { regions, scale } = tileRegions(bitmap.width, bitmap.height)
        const tileCount = Math.min(regions.length, MAX_OUTPUT_ITEMS)
        for (let index = 0; index < tileCount; index += 1) {
          const region = regions[index]
          const encoded = await renderRegion(
            bitmap,
            region,
            Math.round(region.width * scale),
            Math.round(region.height * scale),
            true,
          )
          const item = outputItem(
            candidate,
            encoded,
            `${candidate.image_id}-T${String(index + 1).padStart(2, "0")}`,
            index + 1,
            tileCount,
            bitmap,
          )
          item.data_url = await blobToDataUrl(encoded.blob)
          const { byte_length: byteLength, ...serializable } = item
          rendered.push({ serializable, byteLength })
        }
      } else {
        const scale = Math.min(1, MAX_NORMAL_EDGE / Math.max(bitmap.width, bitmap.height))
        const encoded = await renderRegion(
          bitmap,
          { x: 0, y: 0, width: bitmap.width, height: bitmap.height },
          Math.round(bitmap.width * scale),
          Math.round(bitmap.height * scale),
          sourceBlob.type === "image/png",
        )
        const item = outputItem(candidate, encoded, candidate.image_id, 1, 1, bitmap)
        item.data_url = await blobToDataUrl(encoded.blob)
        const { byte_length: byteLength, ...serializable } = item
        rendered.push({ serializable, byteLength })
      }
      return rendered
    } finally {
      bitmap.close()
    }
  } catch (_error) {
    return []
  }
}

async function collectPreparedImages(candidates) {
  const images = []
  let totalBytes = 0
  let sourceCount = 0
  for (const candidate of candidates || []) {
    if (sourceCount >= MAX_SOURCE_IMAGES || images.length >= MAX_OUTPUT_ITEMS) break
    sourceCount += 1
    const prepared = await prepareCandidateSafe(candidate)
    const candidateBytes = prepared.reduce((sum, item) => sum + item.byteLength, 0)
    if (!prepared.length || totalBytes + candidateBytes > MAX_TOTAL_IMAGE_BYTES) continue
    const remaining = MAX_OUTPUT_ITEMS - images.length
    const accepted = prepared.slice(0, remaining)
    images.push(...accepted.map((item) => item.serializable))
    totalBytes += accepted.reduce((sum, item) => sum + item.byteLength, 0)
  }
  return images
}

function envelopeByteLength(value) {
  return new TextEncoder().encode(JSON.stringify(value)).length
}

async function sendCurrentArticle(tab) {
  try {
    await cleanExpiredImports()
    if (!tab?.id) throw new Error("没有找到当前浏览器标签页。")
    if (!isWeChatArticleUrl(tab.url || "")) {
      throw new Error("请先打开一篇微信公众号文章，再点击 LayerRead Connector。")
    }
    const response = await extractArticleFromTab(tab)
    if (!response?.ok || !response.article) {
      throw new Error(response?.error || "当前页面不是可导入的微信公众号文章。")
    }

    await chrome.tabs.sendMessage(tab.id, {
      type: "LAYERREAD_SHOW_MESSAGE",
      level: "success",
      message: "正文已提取，正在处理高清图片…",
    }).catch(() => {})

    const article = { ...response.article }
    const candidates = article.image_candidates || []
    delete article.image_candidates
    article.images = await collectPreparedImages(candidates)

    const importId = crypto.randomUUID()
    const envelope = {
      protocol_version: 2,
      import_id: importId,
      created_at: new Date().toISOString(),
      expires_at: Date.now() + IMPORT_TTL_MS,
      article,
    }
    let omittedImageItems = 0
    while (article.images.length && envelopeByteLength(envelope) > MAX_SESSION_ENVELOPE_BYTES) {
      article.images.pop()
      omittedImageItems += 1
    }
    if (envelopeByteLength(envelope) > MAX_SESSION_ENVELOPE_BYTES) {
      throw new Error("文章正文超过浏览器安全传输上限；请改用正文粘贴，或选择内容较短的文章。")
    }

    omittedImageItems += await storeImportEnvelope(importId, envelope)
    await showBadge(tab.id, "✓", "#16a34a", "文章已发送到 LayerRead")
    await chrome.tabs.sendMessage(tab.id, {
      type: "LAYERREAD_SHOW_MESSAGE",
      level: "success",
      message: article.images.length
        ? `已保留 ${article.images.length} 个高清图片或长图分片${
            omittedImageItems ? `；为适配浏览器临时容量，跳过 ${omittedImageItems} 个图片或分片` : ""
          }，正在打开 LayerRead…`
        : omittedImageItems
          ? `正文已提取；图片因浏览器临时容量不足而跳过 ${omittedImageItems} 个，正在打开 LayerRead…`
          : "正文已提取；未找到可安全传输的正文图片，正在打开 LayerRead…",
    }).catch(() => {})
    await openOrReuseLayerRead(importId)
  } catch (error) {
    const message = error instanceof Error ? error.message : "文章发送失败。"
    if (tab?.id) {
      await showBadge(tab.id, "!", "#dc2626", message)
      chrome.tabs.sendMessage(tab.id, {
        type: "LAYERREAD_SHOW_MESSAGE",
        level: "error",
        message,
      }).catch(() => {})
    }
  }
}

chrome.action.onClicked.addListener((tab) => {
  sendCurrentArticle(tab)
})

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (!isLayerReadSender(sender)) return false

  if (request?.type === "LAYERREAD_CONNECTOR_PING") {
    sendResponse({ ok: true })
    return false
  }
  if (request?.type === "LAYERREAD_ACK_IMPORT") {
    const importId = request.import_id
    if (typeof importId !== "string" || !TOKEN_PATTERN.test(importId)) {
      sendResponse({ ok: false })
      return false
    }
    chrome.storage.session.remove(storageKey(importId)).then(
      () => sendResponse({ ok: true }),
      () => sendResponse({ ok: false }),
    )
    return true
  }
  if (request?.type !== "LAYERREAD_CLAIM_IMPORT") return false

  const importId = request.import_id
  if (typeof importId !== "string" || !TOKEN_PATTERN.test(importId)) {
    sendResponse({ ok: false, error: "一次性导入标识无效。" })
    return false
  }

  ;(async () => {
    await cleanExpiredImports()
    const key = storageKey(importId)
    const values = await chrome.storage.session.get(key)
    const payload = values[key]
    if (!payload || payload.expires_at <= Date.now()) {
      await chrome.storage.session.remove(key)
      sendResponse({ ok: false, error: "一次性导入已失效，请回到文章页面重新发送。" })
      return
    }
    sendResponse({ ok: true, payload })
  })().catch(() => {
    sendResponse({ ok: false, error: "扩展读取文章失败，请重新发送。" })
  })
  return true
})

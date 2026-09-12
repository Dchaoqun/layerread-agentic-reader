const BLOCK_TAGS = new Set([
  "ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "DIV", "DL", "FIELDSET",
  "FIGCAPTION", "FIGURE", "FOOTER", "H1", "H2", "H3", "H4", "H5", "H6",
  "HEADER", "HR", "LI", "MAIN", "NAV", "OL", "P", "PRE", "SECTION", "TABLE",
  "TBODY", "TD", "TFOOT", "TH", "THEAD", "TR", "UL",
])
const SKIP_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "SVG", "CANVAS"])
const MAX_BODY_BYTES = 5 * 1024 * 1024
const MAX_IMAGE_CANDIDATES = 18
const OBVIOUS_JUNK_PATTERN = /(?:二维码|头像|赞赏|广告|logo|表情|emoji|avatar|qrcode|reward)/i

function firstText(selectors) {
  for (const selector of selectors) {
    const element = document.querySelector(selector)
    const value = element?.textContent?.replace(/\s+/g, " ").trim()
    if (value) return value
  }
  return ""
}

function firstMeta(selectors) {
  for (const selector of selectors) {
    const value = document.querySelector(selector)?.getAttribute("content")?.trim()
    if (value) return value
  }
  return ""
}

function compactText(value, maxLength = 500) {
  const cleaned = (value || "").replace(/\s+/g, " ").trim()
  return cleaned.length <= maxLength ? cleaned : `${cleaned.slice(0, maxLength - 1)}…`
}

function imageSource(element) {
  const raw = element.getAttribute("data-src") ||
    element.getAttribute("data-original") ||
    element.currentSrc || element.getAttribute("src") || ""
  if (!raw) return ""
  try {
    const url = new URL(raw, location.href)
    if (url.protocol === "http:" && /(?:^|\.)mmbiz\.qpic\.cn$/i.test(url.hostname)) {
      url.protocol = "https:"
    }
    return url.protocol === "https:" || url.protocol === "data:" ? url.toString() : ""
  } catch (_error) {
    return ""
  }
}

function nearbyBlockText(element, direction) {
  let node = element.closest("figure, p, section, div") || element
  for (let depth = 0; depth < 4 && node; depth += 1) {
    let sibling = direction === "before" ? node.previousElementSibling : node.nextElementSibling
    for (let attempt = 0; attempt < 3 && sibling; attempt += 1) {
      const text = compactText(sibling.textContent, 800)
      if (text) return text
      sibling = direction === "before" ? sibling.previousElementSibling : sibling.nextElementSibling
    }
    node = node.parentElement
  }
  return ""
}

function imageCaption(element) {
  const figureCaption = element.closest("figure")?.querySelector("figcaption")?.textContent
  if (compactText(figureCaption)) return compactText(figureCaption)
  const parent = element.parentElement
  const siblingCaption = parent?.nextElementSibling?.matches("figcaption, .caption, [class*=caption]")
    ? parent.nextElementSibling.textContent
    : ""
  return compactText(siblingCaption)
}

function isObviousJunk(element, source, width, height, alt, caption) {
  if (!source) return true
  if (width > 0 && height > 0 && width < 48 && height < 48) return true
  const signals = `${alt} ${caption} ${element.className || ""} ${element.id || ""} ${source}`
  return OBVIOUS_JUNK_PATTERN.test(signals)
}

function imagePriority(candidate) {
  const area = Math.max(1, candidate.width * candidate.height)
  let score = Math.min(12, Math.log2(area) - 12)
  if (candidate.caption) score += 4
  if (candidate.alt_text) score += 2
  if (candidate.context_before || candidate.context_after) score += 2
  const ratio = candidate.width && candidate.height
    ? Math.max(candidate.width / candidate.height, candidate.height / candidate.width)
    : 1
  if (ratio >= 2.5) score += 2
  if (candidate.width >= 900 || candidate.height >= 900) score += 3
  return score
}

function collectArticleImages(root) {
  const candidates = []
  const byElement = new Map()
  for (const element of root.querySelectorAll("img")) {
    let style
    try {
      style = getComputedStyle(element)
    } catch (_error) {
      style = null
    }
    if (style?.display === "none" || style?.visibility === "hidden") continue

    const source = imageSource(element)
    const width = Math.round(element.naturalWidth || Number(element.getAttribute("data-w")) || element.width || 0)
    const height = Math.round(element.naturalHeight || Number(element.getAttribute("data-h")) || element.height || 0)
    const alt = compactText(element.getAttribute("alt"))
    const caption = imageCaption(element)
    if (isObviousJunk(element, source, width, height, alt, caption)) continue

    const imageId = `IMG${String(candidates.length + 1).padStart(2, "0")}`
    const candidate = {
      image_id: imageId,
      source_url: source,
      width,
      height,
      alt_text: alt,
      caption,
      context_before: nearbyBlockText(element, "before"),
      context_after: nearbyBlockText(element, "after"),
      order: candidates.length + 1,
    }
    candidate.priority = imagePriority(candidate)
    candidates.push(candidate)
    byElement.set(element, candidate)
  }

  const selected = [...candidates]
    .sort((left, right) => right.priority - left.priority || left.order - right.order)
    .slice(0, MAX_IMAGE_CANDIDATES)
    .map(({ priority, ...candidate }) => candidate)
  return { candidates, selected, byElement }
}

function visibleArticleText(root, imageMap) {
  const chunks = []

  function walk(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      chunks.push(node.nodeValue || "")
      return
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return
    const element = node
    if (SKIP_TAGS.has(element.tagName) || element.getAttribute("aria-hidden") === "true") return
    if (element.tagName === "IMG") {
      const image = imageMap.get(element)
      if (image) {
        const description = image.caption || image.alt_text
        chunks.push(`\n\n[图片 ${image.image_id}${description ? `：${description}` : ""}]\n\n`)
      }
      return
    }
    if (element.tagName === "BR") {
      chunks.push("\n")
      return
    }

    const block = BLOCK_TAGS.has(element.tagName)
    if (block) chunks.push("\n\n")
    for (const child of element.childNodes) walk(child)
    if (block) chunks.push("\n\n")
  }

  walk(root)
  return chunks.join("")
    .replace(/\r/g, "\n")
    .replace(/[\t\f\v\u00a0 ]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

function extractArticle() {
  if (location.protocol !== "https:" || location.hostname !== "mp.weixin.qq.com") {
    throw new Error("请先打开一篇微信公众号文章。")
  }
  const root = document.querySelector("#js_content")
  if (!root) throw new Error("当前页面没有找到微信公众号正文。")

  const { candidates, selected, byElement } = collectArticleImages(root)
  const body = visibleArticleText(root, byElement)
  if (body.replace(/\s/g, "").length < 80) {
    throw new Error("提取到的正文过短，请确认文章已经完整加载。")
  }
  if (new TextEncoder().encode(body).length > MAX_BODY_BYTES) {
    throw new Error("文章正文超过 5 MB 安全上限。")
  }

  return {
    title: firstText(["#activity-name", "h1.rich_media_title"]) ||
      firstMeta(['meta[property="og:title"]']) || document.title,
    account: firstText(["#js_name", ".rich_media_meta_nickname"]),
    author: firstMeta(['meta[name="author"]', 'meta[property="article:author"]']) ||
      firstText(["#author", ".rich_media_meta_text"]),
    published_at: firstText(["#publish_time", "#meta_content_hide_info em"]) ||
      firstMeta(['meta[property="article:published_time"]']),
    source_url: `${location.origin}${location.pathname}${location.search}`,
    body,
    image_count: candidates.length,
    image_candidates: selected,
  }
}

function showToast(message, level) {
  const existing = document.getElementById("layerread-connector-toast")
  if (existing) existing.remove()
  const toast = document.createElement("div")
  toast.id = "layerread-connector-toast"
  toast.textContent = message
  Object.assign(toast.style, {
    background: level === "error" ? "#b91c1c" : "#166534",
    borderRadius: "8px",
    boxShadow: "0 8px 30px rgba(0,0,0,.22)",
    color: "white",
    fontSize: "14px",
    fontWeight: "600",
    maxWidth: "380px",
    padding: "12px 16px",
    position: "fixed",
    right: "20px",
    top: "20px",
    zIndex: "2147483647",
  })
  document.documentElement.appendChild(toast)
  window.setTimeout(() => toast.remove(), 5000)
}

chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
  if (request?.type === "LAYERREAD_SHOW_MESSAGE") {
    showToast(request.message || "", request.level || "success")
    return false
  }
  if (request?.type !== "LAYERREAD_EXTRACT_ARTICLE") return false
  try {
    sendResponse({ ok: true, article: extractArticle() })
  } catch (error) {
    sendResponse({
      ok: false,
      error: error instanceof Error ? error.message : "微信公众号文章提取失败。",
    })
  }
  return false
})

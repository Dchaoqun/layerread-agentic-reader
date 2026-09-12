const onlineOrigin = globalThis.LAYERREAD_ONLINE_ORIGIN || ""
const demoInput = document.querySelector("#target-demo")
const originCopy = document.querySelector("#demo-origin")
const status = document.querySelector("#status")

originCopy.textContent = onlineOrigin || "当前扩展包尚未配置在线 Demo 地址。"
demoInput.disabled = !onlineOrigin

chrome.storage.local.get("layerread_target", ({ layerread_target: stored }) => {
  const target = stored === "local" || !onlineOrigin ? "local" : "demo"
  const input = document.querySelector(`input[name="target"][value="${target}"]`)
  if (input) input.checked = true
})

document.querySelector("#save").addEventListener("click", async () => {
  const selected = document.querySelector('input[name="target"]:checked')?.value
  if (!selected || (selected === "demo" && !onlineOrigin)) return
  await chrome.storage.local.set({ layerread_target: selected })
  status.textContent = selected === "demo" ? "已切换到在线 Demo。" : "已切换到本地应用。"
})

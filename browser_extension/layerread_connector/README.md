# LayerRead Connector

这是 LayerRead 的 Chrome Manifest V3 扩展。它在用户已经打开的微信公众号文章页面中读取正文和正文图片，并通过一次性令牌发送给获准的 LayerRead 页面。源码包默认只连接本机 `http://localhost:8501/`；在线包通过构建脚本加入一个精确 HTTPS Origin。

## 安装

1. 在 Chrome 打开 `chrome://extensions/`。
2. 打开右上角“开发者模式”。
3. 点击“加载已解压的扩展程序”。
4. 选择本目录：`browser_extension/layerread_connector`。
5. 建议把 LayerRead Connector 固定到浏览器工具栏。
6. 如果安装过 Deep Reader、Personal 或其他 LayerRead Connector，只保留当前版本；多个版本会同时响应 localhost 页面。

## 使用

1. 先启动 LayerRead，确认 `http://localhost:8501/` 可以打开。
2. 在 Chrome 打开一篇 `https://mp.weixin.qq.com/` 文章。
3. 等待页面加载完成，从顶部滚动到底部一次，确认正文中的图片、PPT、图表和长图都已显示。
4. 点击工具栏中的 LayerRead Connector 图标。
5. 扩展会提取正文、处理重要图片并自动打开 LayerRead；对照原文检查图片和内容后再确认导入。

微信公众号图片可能懒加载；尚未滚动到、尚未显示的图片无法被 Connector 读取。发现图片缺失时，请刷新原文、完成滚动后重新导入。

更新扩展代码后，请在 `chrome://extensions/` 找到 LayerRead Connector 并点击一次“重新加载”。0.4.4 会在新导入开始前清除 LayerRead 自己尚未完成的旧导入包，避免重试时累计占满 Chrome 会话存储；如果单篇文章仍接近容量上限，会从最低优先级的图片或分片开始自动减少并明确提示，正文会优先传输。它也会在扩展更新导致旧文章标签页失去连接时自动重新注入内容脚本并重试，在获准的 LayerRead Origin 的同域 iframe 中注入桥接脚本，并等 LayerRead 确认收到文章后才删除浏览器中的临时文章包；若 Chrome 仍阻止注入，刷新该文章页面后再点击扩展即可。打开扩展详情中的“扩展程序选项”，可以在已配置的在线 Demo 和本地应用之间切换。

## 构建在线 Demo 包

获得最终 Demo URL 后，在项目根目录运行：

```powershell
uv run python scripts/build_demo_connector.py `
  --origin https://你的最终-demo-域名 `
  --output C:\path\to\layerread-demo-connector
```

输出目录必须尚不存在。脚本只接受不含路径、账号、查询参数和片段的 HTTPS Origin，并把该精确 Origin 同时写入扩展运行配置、Host Permission 和桥接脚本匹配规则。

## 安全边界

- 只在 `mp.weixin.qq.com` 文章页运行正文提取脚本。
- 文章仅暂存在 `chrome.storage.session`，十分钟后失效；LayerRead 完成校验并确认接收后立即删除。开始新的导入时，只会清理扩展自身尚未完成的旧导入包，不会清除其他扩展或网页数据。
- 源码桥接脚本只响应 `localhost:8501` 和 `127.0.0.1:8501`；在线构建只额外响应指定的精确 HTTPS Origin。
- 只申请微信公众号图片 CDN、本机 LayerRead 和最终 Demo 精确域名的读取权限，不使用 `<all_urls>`。
- 默认保留正文内容图片；只过滤头像、二维码、广告、Logo、表情和极小图片等明显噪声。
- 普通图片最长边最多 3200 像素；长图会以带重叠的高清分片传输，优先保证 PPT 和图片文字可读。
- 单篇最多处理 12 张来源图片、24 个图片/分片，图片总量不超过 6 MB，完整一次性包不超过 9 MB。
- 本地/Portfolio 模式会把导入结果保存到用户自己的 SQLite；在线 Demo 只保留在当前页面会话，不写入共享文章库。
- 导入图片不会自动调用模型。只有用户确认模型支持图片输入、在侧边栏明确授权，并主动点击分析后，图片才会发送给所选模型供应商。
- 在线 Demo 中，正文和图片会经过 LayerRead 服务器，并可能发送给部署者配置的模型供应商；图片请求可能产生额外费用并受限额和供应商数据政策约束。
- 不绕过付费、私密或权限受限内容。

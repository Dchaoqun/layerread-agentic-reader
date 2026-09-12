# v0.7 真实文章测试材料

此目录只保存可以安全提交到仓库的测试说明、开放许可材料或团队自有材料。

真实文章全文默认放在本目录下的 `local/`，该目录已被 `.gitignore` 排除。不要提交未经许可的公众号全文、付费文章、私人材料、API Key、Notion Token 或其他凭据。

每份本地材料建议使用 UTF-8 纯文本或 Markdown，并在对应的 v0.7 单篇验收记录中登记：

- 来源链接、标题、作者和获取日期。
- 使用授权或仅限本地测试的说明。
- 文件 SHA-256。
- 字符数、段落数和图片依赖情况。
- 与公开网页存在差异时的必要说明。

PowerShell 计算文件哈希：

```powershell
Get-FileHash -Algorithm SHA256 "tests\fixtures\v07_articles\local\article.md"
```

测试记录不得复制完整文章。仅在定位清洗、引用或解析问题确有必要时保留最短充分摘述，并附对应 `[Pxx]` 编号。

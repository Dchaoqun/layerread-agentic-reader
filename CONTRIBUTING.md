# Contributing to LayerRead

感谢你关注叠读 LayerRead。

当前仓库首先用于展示和验证 Agentic 阅读工作流。Bug 报告、复现步骤、文档反馈和安全建议都很欢迎；但在贡献者协议和版权授权流程确定之前，项目暂不接受外部代码 Pull Request。请不要提交你无权授权的代码、文章、图片、Prompt 或测试数据。

报告普通问题时，请提供：

- 操作系统、Python、uv、Streamlit 和 Chrome 版本
- 使用的导入方式与模型类型；不要粘贴 API Key
- 最小复现步骤、预期结果和实际结果
- 已脱敏的错误信息；不要上传私人文章、SQLite 数据库或 Connector 导出包

## 本地验证

修改代码或准备最小复现后，请先运行完整自动化测试：

```powershell
uv sync
uv run pytest
```

提交问题时请说明测试是否通过；如果失败，请提供首个失败用例和已脱敏的错误摘要。

安全问题不要作为公开 Issue 提交。正式公开仓库前，维护者会启用 GitHub Private Vulnerability Reporting 并在 `SECURITY.md` 中公布流程。

所有公开代码使用 `AGPL-3.0-only`。未来如果开放代码贡献，将先公布明确的贡献者许可或 CLA 规则。

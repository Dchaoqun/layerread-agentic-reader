from __future__ import annotations

import socket

import pytest

from layerread.url_import import (
    URLImportError,
    extract_article_from_url,
    validate_public_url,
)


GENERIC_HTML = """
<html>
  <head>
    <meta charset="utf-8">
    <title>一篇测试文章</title>
    <meta name="author" content="测试作者">
    <meta property="article:published_time" content="2026-07-19">
  </head>
  <body>
    <nav>不应进入正文的导航</nav>
    <article>
      <h1>一篇测试文章</h1>
      <p>第一段解释为什么深度阅读需要先判断信息质量，再决定投入多少注意力。</p>
      <p>第二段说明有效学习还需要主动回忆、即时反馈和能够回到原文的可靠引用。</p>
      <p>第三段补充，结构化导出让结论、个人回答与待研究问题能够长期沉淀和复用。</p>
    </article>
  </body>
</html>
"""


WECHAT_HTML = """
<html><head><meta property="og:title" content="公众号测试文章"></head><body>
<h1 id="activity-name">公众号测试文章</h1><a id="js_name">深读测试号</a>
<em id="publish_time">2026-07-19</em>
<div id="js_content">
<p>第一段公众号正文用于验证专用适配器能够识别 js_content 容器并保留主要文字。</p>
<p>第二段继续补充足够内容，确保抽取结果超过文章分析允许进入处理流程的最低长度。</p>
<p>第三段说明失败时还应继续尝试通用抽取，并最终允许用户回到手动粘贴正文。</p>
</div></body></html>
"""


def test_generic_adapter_extracts_body_and_metadata() -> None:
    result = extract_article_from_url(
        "https://example.com/article",
        fetcher=lambda _url: ("https://example.com/article", GENERIC_HTML),
    )

    assert result.is_success
    assert result.extraction_method == "generic_html"
    assert result.title == "一篇测试文章"
    assert result.author == "测试作者"
    assert "主动回忆" in result.body
    assert "不应进入正文的导航" not in result.body


def test_wechat_adapter_has_priority_and_extracts_account() -> None:
    result = extract_article_from_url(
        "https://mp.weixin.qq.com/s/test",
        fetcher=lambda _url: ("https://mp.weixin.qq.com/s/test", WECHAT_HTML),
    )

    assert result.is_success
    assert result.extraction_method == "wechat_official_account"
    assert result.account == "深读测试号"
    assert result.published_at == "2026-07-19"


def test_extraction_failure_keeps_actionable_attempts() -> None:
    result = extract_article_from_url(
        "https://example.com/empty",
        fetcher=lambda _url: ("https://example.com/empty", "<html><body>太短</body></html>"),
    )

    assert not result.is_success
    assert result.attempts[-1].adapter == "generic_html"
    assert "正文过短" in result.attempts[-1].message


def test_meta_without_name_or_property_never_crashes_parser() -> None:
    html = GENERIC_HTML.replace(
        '<meta charset="utf-8">',
        '<meta charset="utf-8"><meta http-equiv="X-UA-Compatible" content="IE=edge">',
    )

    result = extract_article_from_url(
        "https://example.com/article",
        fetcher=lambda _url: ("https://example.com/article", html),
    )

    assert result.is_success
    assert "主动回忆" in result.body


def test_unexpected_adapter_parse_error_becomes_failed_attempt() -> None:
    class BrokenAdapter:
        name = "broken"

        def can_handle(self, _url: str) -> bool:
            return True

        def extract(self, _url: str, _html: str):
            raise AttributeError("simulated malformed HTML")

    result = extract_article_from_url(
        "https://example.com/article",
        fetcher=lambda _url: ("https://example.com/article", GENERIC_HTML),
        adapters=(BrokenAdapter(),),
    )

    assert not result.is_success
    assert result.attempts[0].adapter == "broken"
    assert "网页结构异常" in result.attempts[0].message


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://127.0.0.1/private",
        "https://user:secret@example.com/article",
    ],
)
def test_private_or_credentialed_urls_are_rejected(url: str) -> None:
    with pytest.raises(URLImportError):
        validate_public_url(url)


def test_public_hostname_is_normalized(monkeypatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
    )

    assert validate_public_url("https://Example.com/article#fragment") == "https://example.com/article"

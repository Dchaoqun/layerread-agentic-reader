from __future__ import annotations

import base64

import httpx
import pytest

import layerread.notion as notion_module
from layerread.article import ArticleImage
from layerread.export import ExportBlock, ExportDocument, ExportSection
from layerread.notion import (
    MAX_BLOCKS_PER_REQUEST,
    MAX_RICH_TEXT_CHARS,
    NotionClient,
    NotionError,
    document_to_notion_blocks,
    normalize_notion_id,
    split_rich_text,
)


def make_document(block_count: int = 1) -> ExportDocument:
    return ExportDocument(
        title="LayerRead 测试导出",
        source_url="https://example.com/article",
        sections=[
            ExportSection(
                section_id="metadata",
                title="基本信息",
                blocks=[
                    ExportBlock(kind="paragraph", text=f"内容 {index}")
                    for index in range(block_count)
                ],
            )
        ],
    )


def test_notion_id_accepts_raw_id_and_page_url() -> None:
    raw = "123456781234123412341234567890ab"
    expected = "12345678-1234-1234-1234-1234567890ab"

    assert normalize_notion_id(raw) == expected
    assert normalize_notion_id(f"https://www.notion.so/Title-{raw}?v=1") == expected


def test_long_rich_text_is_split_under_notion_limit() -> None:
    chunks = split_rich_text("段" * (MAX_RICH_TEXT_CHARS * 2 + 17))

    assert "".join(chunks) == "段" * (MAX_RICH_TEXT_CHARS * 2 + 17)
    assert all(len(chunk) <= MAX_RICH_TEXT_CHARS for chunk in chunks)


def test_split_does_not_include_separator_just_beyond_limit() -> None:
    text = "段" * MAX_RICH_TEXT_CHARS + "。尾"

    chunks = split_rich_text(text)

    assert "".join(chunks) == text
    assert [len(chunk) for chunk in chunks] == [MAX_RICH_TEXT_CHARS, 2]


def test_quote_blocks_never_exceed_notion_rich_text_limit() -> None:
    quote = "引" * MAX_RICH_TEXT_CHARS + "。用"
    document = ExportDocument(
        title="引用边界测试",
        source_url="",
        sections=[
            ExportSection(
                section_id="metadata",
                title="引用",
                blocks=[ExportBlock(kind="quote", text=quote)],
            )
        ],
    )

    blocks = document_to_notion_blocks(document)
    quote_blocks = [block for block in blocks if block["type"] == "quote"]
    contents = [
        item["text"]["content"]
        for block in quote_blocks
        for item in block["quote"]["rich_text"]
    ]

    assert "".join(contents) == quote
    assert all(len(content) <= MAX_RICH_TEXT_CHARS for content in contents)


def test_document_conversion_preserves_sections_and_source_link() -> None:
    blocks = document_to_notion_blocks(make_document())

    assert blocks[0]["type"] == "paragraph"
    assert blocks[0]["paragraph"]["rich_text"][0]["text"]["link"]["url"] == "https://example.com/article"
    assert blocks[1]["type"] == "heading_1"
    assert blocks[2]["type"] == "paragraph"


class FakeNotionClient(NotionClient):
    def __init__(self) -> None:
        super().__init__("test-token")
        self.calls = []
        self.multipart_calls = []

    def _request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "GET" and path.startswith("/pages/"):
            return {
                "properties": {
                    "title": {
                        "type": "title",
                        "title": [{"plain_text": "学习成果"}],
                    }
                }
            }
        if method == "POST" and path == "/pages":
            return {
                "id": "new-page-id",
                "url": "https://www.notion.so/new-page-id",
            }
        if method == "POST" and path == "/file_uploads":
            return {"id": "uploaded-image-id", "status": "pending"}
        if method == "PATCH":
            return {"results": []}
        raise AssertionError((method, path, payload))

    def _request_multipart(self, path, *, filename, content, content_type):
        self.multipart_calls.append((path, filename, content, content_type))
        return {"id": "uploaded-image-id", "status": "uploaded"}


def test_export_creates_page_then_appends_blocks_in_batches() -> None:
    client = FakeNotionClient()
    document = make_document(MAX_BLOCKS_PER_REQUEST * 2 + 5)

    result = client.export_document(
        document,
        "123456781234123412341234567890ab",
        "page",
    )

    patches = [call for call in client.calls if call[0] == "PATCH"]
    assert result.page_url == "https://www.notion.so/new-page-id"
    assert len(patches) == 3
    assert all(len(call[2]["children"]) <= MAX_BLOCKS_PER_REQUEST for call in patches)


def test_export_uploads_article_images_and_attaches_image_blocks() -> None:
    client = FakeNotionClient()
    image_bytes = b"notion-image-payload"
    image = ArticleImage(
        image_id="IMG03-T2",
        data_url="data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
        mime_type="image/png",
        width=1600,
        height=2400,
        caption="关键路线图",
        tile_index=2,
        tile_count=3,
    )

    result = client.export_document(
        make_document(),
        "123456781234123412341234567890ab",
        "page",
        images=(image,),
    )

    assert result.uploaded_images == 1
    assert client.multipart_calls == [
        (
            "/file_uploads/uploaded-image-id/send",
            "layerread-img03-t2.png",
            image_bytes,
            "image/png",
        )
    ]
    children = [
        block
        for method, path, payload in client.calls
        if method == "PATCH"
        for block in payload["children"]
    ]
    image_block = next(block for block in children if block["type"] == "image")
    assert any(
        block["type"] == "heading_1"
        and block["heading_1"]["rich_text"][0]["text"]["content"] == "原文重要图片"
        for block in children
    )
    assert image_block["image"]["file_upload"]["id"] == "uploaded-image-id"
    assert (
        image_block["image"]["caption"][0]["text"]["content"]
        == "关键路线图（长图分片 2/3）"
    )


def test_multipart_upload_lets_httpx_set_boundary(monkeypatch) -> None:
    observed_headers: list[str] = []

    def create_client(route, timeout_seconds):
        def handler(request: httpx.Request) -> httpx.Response:
            observed_headers.append(request.headers["content-type"])
            assert b"notion-image-payload" in request.content
            return httpx.Response(
                200,
                request=request,
                json={"id": "upload-id", "status": "uploaded"},
            )

        return _mock_client(handler)

    monkeypatch.setattr(notion_module, "_environment_proxy_url", lambda: "")
    monkeypatch.setattr(notion_module, "_create_http_client", create_client)
    client = NotionClient("ntn_test", proxy_url="")

    result = client._request_multipart(
        "/file_uploads/upload-id/send",
        filename="layerread-img01.webp",
        content=b"notion-image-payload",
        content_type="image/webp",
    )

    assert result["status"] == "uploaded"
    assert observed_headers[0].startswith("multipart/form-data; boundary=")


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_connection_falls_back_from_proxy_to_direct(monkeypatch) -> None:
    route_calls: list[str] = []
    monkeypatch.setattr(notion_module, "_environment_proxy_url", lambda: "")

    def create_client(route, timeout_seconds):
        route_calls.append(route.proxy_url)

        def handler(request: httpx.Request) -> httpx.Response:
            if route.proxy_url:
                raise httpx.ConnectError("proxy unavailable", request=request)
            return httpx.Response(
                200,
                request=request,
                json={
                    "properties": {
                        "title": {
                            "type": "title",
                            "title": [{"plain_text": "Agent Harness"}],
                        }
                    }
                },
            )

        return _mock_client(handler)

    monkeypatch.setattr(notion_module, "_create_http_client", create_client)
    client = NotionClient("ntn_test", proxy_url="http://127.0.0.1:7890")

    target = client.test_connection(
        "3b0a2b92ce2380658f9ae502e2b19764",
        "page",
    )

    assert target.display_name == "Agent Harness"
    assert route_calls == ["http://127.0.0.1:7890", ""]


def test_successful_route_is_reused_for_writes(monkeypatch) -> None:
    route_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(notion_module, "_environment_proxy_url", lambda: "")

    def create_client(route, timeout_seconds):
        def handler(request: httpx.Request) -> httpx.Response:
            route_calls.append((request.method, route.proxy_url))
            if route.proxy_url:
                raise httpx.ConnectError("proxy unavailable", request=request)
            if request.method == "GET":
                return httpx.Response(
                    200,
                    request=request,
                    json={
                        "properties": {
                            "title": {
                                "type": "title",
                                "title": [{"plain_text": "Agent Harness"}],
                            }
                        }
                    },
                )
            if request.method == "POST":
                return httpx.Response(
                    200,
                    request=request,
                    json={"id": "new-page-id", "url": "https://notion.so/new-page-id"},
                )
            return httpx.Response(200, request=request, json={"results": []})

        return _mock_client(handler)

    monkeypatch.setattr(notion_module, "_create_http_client", create_client)
    client = NotionClient("ntn_test", proxy_url="http://127.0.0.1:7890")

    client.export_document(
        make_document(),
        "3b0a2b92ce2380658f9ae502e2b19764",
        "page",
    )

    assert route_calls == [
        ("GET", "http://127.0.0.1:7890"),
        ("GET", ""),
        ("POST", ""),
        ("PATCH", ""),
    ]


def test_write_read_timeout_is_not_retried(monkeypatch) -> None:
    route_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(notion_module, "_environment_proxy_url", lambda: "")

    def create_client(route, timeout_seconds):
        def handler(request: httpx.Request) -> httpx.Response:
            route_calls.append((request.method, route.proxy_url))
            if request.method == "GET":
                return httpx.Response(
                    200,
                    request=request,
                    json={
                        "properties": {
                            "title": {
                                "type": "title",
                                "title": [{"plain_text": "Agent Harness"}],
                            }
                        }
                    },
                )
            raise httpx.ReadTimeout("response timeout", request=request)

        return _mock_client(handler)

    monkeypatch.setattr(notion_module, "_create_http_client", create_client)
    client = NotionClient("ntn_test", proxy_url="http://127.0.0.1:7890")

    with pytest.raises(NotionError, match="可能已经到达 Notion"):
        client.export_document(
            make_document(),
            "3b0a2b92ce2380658f9ae502e2b19764",
            "page",
        )

    assert route_calls == [
        ("GET", "http://127.0.0.1:7890"),
        ("POST", "http://127.0.0.1:7890"),
    ]


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, "Integration Token 无效或已失效"),
        (404, "尚未共享给 Integration"),
    ],
)
def test_http_errors_are_specific_and_secret_safe(
    monkeypatch,
    status_code: int,
    expected: str,
) -> None:
    token = "ntn_super_secret_value"

    def create_client(route, timeout_seconds):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code,
                request=request,
                json={
                    "code": "object_not_found" if status_code == 404 else "unauthorized",
                    "message": f"request rejected for {token}",
                },
            )

        return _mock_client(handler)

    monkeypatch.setattr(notion_module, "_create_http_client", create_client)
    client = NotionClient(token, proxy_url="")

    with pytest.raises(NotionError) as raised:
        client.test_connection(
            "3b0a2b92ce2380658f9ae502e2b19764",
            "page",
        )

    message = str(raised.value)
    assert expected in message
    assert token not in message

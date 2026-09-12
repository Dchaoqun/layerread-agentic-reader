from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image
import pytest

from layerread.connector import (
    ConnectorImportError,
    is_connector_extraction_method,
    parse_connector_payload,
)


def _payload(**article_overrides: object) -> dict[str, object]:
    article: dict[str, object] = {
        "title": "Chrome 一键导入测试",
        "account": "深读测试号",
        "author": "测试作者",
        "published_at": "2026-07-19",
        "source_url": "https://mp.weixin.qq.com/s/layerread-test",
        "body": (
            "第一段说明扩展会读取用户已经打开的公众号页面，因此不需要服务器重新请求微信。\n\n"
            "第二段说明文章通过一次性令牌进入本地 LayerRead，并在用户确认后才进入正式分析。\n\n"
            "第三段补充扩展会筛选重要图片，并把清晰图片与正文一起交给支持视觉的模型。"
        ),
        "image_count": 2,
    }
    article.update(article_overrides)
    return {
        "protocol_version": 1,
        "import_id": "9dc5dc7f-a1e2-4bc3-8d45-112233445566",
        "created_at": "2026-07-19T12:00:00.000Z",
        "article": article,
    }


def _image_data_url(*, format: str = "PNG") -> str:
    payload = BytesIO()
    Image.new("RGB", (320, 180), color="white").save(payload, format=format)
    mime = "image/png" if format == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(payload.getvalue()).decode('ascii')}"


def test_connector_records_accept_new_and_legacy_extraction_methods() -> None:
    assert is_connector_extraction_method("layerread_chrome_connector")
    assert is_connector_extraction_method("deepread_chrome_connector")
    assert not is_connector_extraction_method("generic_html")


def _v2_payload(**image_overrides: object) -> dict[str, object]:
    payload = _payload(image_count=1)
    payload["protocol_version"] = 2
    article = payload["article"]
    assert isinstance(article, dict)
    image: dict[str, object] = {
        "image_id": "IMG01",
        "data_url": _image_data_url(),
        "mime_type": "image/png",
        "width": 320,
        "height": 180,
        "alt_text": "AI 生成的产品路线 PPT",
        "caption": "路线图",
        "context_before": "下面是一页路线图",
        "context_after": "作者随后解释实施节奏",
        "source_url": "https://mmbiz.qpic.cn/example",
        "tile_index": 1,
        "tile_count": 1,
        "original_width": 320,
        "original_height": 180,
    }
    image.update(image_overrides)
    article["images"] = [image]
    return payload


def test_connector_payload_becomes_confirmable_article() -> None:
    result = parse_connector_payload(_payload())

    assert result.is_success
    assert result.title == "Chrome 一键导入测试"
    assert result.account == "深读测试号"
    assert result.source_url == "https://mp.weixin.qq.com/s/layerread-test"
    assert result.extraction_method == "layerread_chrome_connector"
    assert result.media_detected
    assert result.image_reference_count == 2


def test_v2_connector_payload_keeps_validated_image_for_model() -> None:
    result = parse_connector_payload(_v2_payload())

    assert len(result.images) == 1
    assert result.images[0].image_id == "IMG01"
    assert result.images[0].width == 320
    assert result.images[0].caption == "路线图"
    assert "高清图片" in result.notes[0]


@pytest.mark.parametrize(
    ("image_overrides", "message"),
    [
        ({"image_id": "../../bad"}, "图片标识无效"),
        ({"width": 319}, "声明的尺寸与实际内容不一致"),
        ({"mime_type": "image/jpeg"}, "声明的类型与实际内容不一致"),
        ({"source_url": "https://example.com/image.png"}, "不受信任"),
        ({"data_url": "data:image/png;base64,aaaa==="}, "Base64"),
    ],
)
def test_v2_connector_rejects_unsafe_images(
    image_overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConnectorImportError, match=message):
        parse_connector_payload(_v2_payload(**image_overrides))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "协议版本不兼容"),
        (
            {
                **_payload(),
                "import_id": "not-a-token",
            },
            "导入标识无效",
        ),
        (
            _payload(source_url="https://example.com/article"),
            "只接受 mp.weixin.qq.com",
        ),
        (
            _payload(body="正文太短"),
            "正文过短",
        ),
        (
            _payload(image_count=-1),
            "图片数量超出安全范围",
        ),
    ],
)
def test_invalid_connector_payloads_are_rejected(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ConnectorImportError, match=message):
        parse_connector_payload(payload)


def test_connector_body_size_is_limited(monkeypatch) -> None:
    monkeypatch.setattr("layerread.connector.MAX_CONNECTOR_BODY_BYTES", 20)

    with pytest.raises(ConnectorImportError, match="超过 5 MB"):
        parse_connector_payload(_payload(body="这是一段超过测试上限的正文内容" * 10))

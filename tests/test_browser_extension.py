from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.build_demo_connector import build_demo_connector, normalize_https_origin
from layerread.connector_package import (
    build_online_connector_archive,
    connector_version,
)


EXTENSION_DIR = (
    Path(__file__).resolve().parents[1]
    / "browser_extension"
    / "layerread_connector"
)


def test_connector_manifest_uses_narrow_mv3_permissions() -> None:
    manifest = json.loads((EXTENSION_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == 3
    assert manifest["name"] == "LayerRead Connector"
    assert manifest["version"] == "0.4.4"
    assert manifest["permissions"] == ["activeTab", "scripting", "storage"]
    assert "<all_urls>" not in json.dumps(manifest)
    assert manifest["host_permissions"] == [
        "https://mmbiz.qpic.cn/*",
        "https://mmbiz.qlogo.cn/*",
        "http://localhost/*",
        "http://127.0.0.1/*",
    ]
    matches = {
        value
        for script in manifest["content_scripts"]
        for value in script["matches"]
    }
    assert "https://mp.weixin.qq.com/*" in matches
    assert "http://localhost/*" in matches
    assert "http://127.0.0.1/*" in matches
    assert manifest["options_page"] == "options.html"
    bridge = next(
        script for script in manifest["content_scripts"] if "bridge.js" in script["js"]
    )
    assert bridge["js"] == ["config.js", "bridge.js"]
    assert bridge["all_frames"] is True


def test_connector_uses_one_time_session_transfer() -> None:
    background = (EXTENSION_DIR / "background.js").read_text(encoding="utf-8")
    bridge = (EXTENSION_DIR / "bridge.js").read_text(encoding="utf-8")

    assert "chrome.storage.session" in background
    assert "IMPORT_TTL_MS" in background
    assert "chrome.storage.session.remove(key)" in background
    assert "layerread_import=" in background
    assert 'url.port === "8501"' in background
    assert 'location.port === "8501"' in bridge
    assert "ONLINE_LAYERREAD_ORIGIN" in background
    assert "ONLINE_LAYERREAD_ORIGIN" in bridge
    assert "LAYERREAD_CLAIM_IMPORT" in bridge
    assert "LAYERREAD_ACK_IMPORT" in background
    assert "LAYERREAD_ACK_IMPORT" in bridge
    assert "LAYERREAD_IMPORT_ACKNOWLEDGED" in bridge
    assert "chrome.runtime.getManifest().version" in bridge
    assert "connector_version: CONNECTOR_VERSION" in bridge


def test_connector_retry_replaces_stale_import_and_degrades_images_on_quota() -> None:
    background = (EXTENSION_DIR / "background.js").read_text(encoding="utf-8")

    assert "async function removePendingImports()" in background
    assert "key.startsWith(IMPORT_PREFIX)" in background
    assert "await removePendingImports()" in background
    assert "isSessionStorageQuotaError" in background
    assert "envelope.article.images.pop()" in background
    assert "文章正文超过 Chrome 临时传输容量" in background


def test_connector_reuses_an_open_layerread_tab() -> None:
    background = (EXTENSION_DIR / "background.js").read_text(encoding="utf-8")

    assert "chrome.tabs.query" in background
    assert "chrome.tabs.update(existing.id" in background
    assert "chrome.windows.update(existing.windowId" in background
    assert "await openOrReuseLayerRead(importId)" in background


def test_connector_recovers_after_extension_reload() -> None:
    background = (EXTENSION_DIR / "background.js").read_text(encoding="utf-8")

    assert "isMissingContentScriptError" in background
    assert 'message.includes("Could not establish connection")' in background
    assert "chrome.scripting.executeScript" in background
    assert 'files: ["content.js"]' in background
    assert "return await chrome.tabs.sendMessage(tab.id, request)" in background


def test_wechat_content_script_returns_plain_text_not_html() -> None:
    content = (EXTENSION_DIR / "content.js").read_text(encoding="utf-8")

    assert 'document.querySelector("#js_content")' in content
    assert "Node.TEXT_NODE" in content
    assert "textContent" in content
    assert "innerHTML" not in content
    assert "MAX_BODY_BYTES" in content
    assert "[图片 ${image.image_id}" in content
    assert "image_candidates" in content


def test_connector_preserves_readable_images_and_tiles_long_images() -> None:
    background = (EXTENSION_DIR / "background.js").read_text(encoding="utf-8")

    assert "protocol_version: 2" in background
    assert "MAX_NORMAL_EDGE = 3200" in background
    assert "TILE_TARGET_WIDTH = 2400" in background
    assert "TILE_OVERLAP = 180" in background
    assert "MAX_TOTAL_IMAGE_BYTES = 6 * 1024 * 1024" in background
    assert "quality of [0.94, 0.91, 0.88]" in background
    assert "createImageBitmap" in background
    assert "OffscreenCanvas" in background


def test_demo_connector_build_adds_only_the_exact_https_origin(tmp_path) -> None:
    output = tmp_path / "connector"

    build_demo_connector("https://layerread-demo.example/", output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "https://layerread-demo.example/*" in manifest["host_permissions"]
    assert "https://*/*" not in json.dumps(manifest)
    bridge = next(
        script for script in manifest["content_scripts"] if "bridge.js" in script["js"]
    )
    assert "https://layerread-demo.example/*" in bridge["matches"]
    config = (output / "config.js").read_text(encoding="utf-8")
    assert config == 'globalThis.LAYERREAD_ONLINE_ORIGIN = "https://layerread-demo.example"\n'


def test_demo_connector_archive_is_ready_for_unpack_and_load() -> None:
    archive_bytes = build_online_connector_archive(
        "https://layerread-agentic-reader-demo.streamlit.app"
    )

    with ZipFile(BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        root = "layerread-online-connector/"
        manifest = json.loads(archive.read(root + "manifest.json"))
        config = archive.read(root + "config.js").decode("utf-8")

    assert root + "background.js" in names
    assert root + "bridge.js" in names
    assert (
        "https://layerread-agentic-reader-demo.streamlit.app/*"
        in manifest["host_permissions"]
    )
    online_bridge = next(
        script
        for script in manifest["content_scripts"]
        if "bridge.js" in script["js"]
    )
    assert online_bridge["all_frames"] is True
    assert "https://*/*" not in json.dumps(manifest)
    assert config == (
        'globalThis.LAYERREAD_ONLINE_ORIGIN = '
        '"https://layerread-agentic-reader-demo.streamlit.app"\n'
    )


def test_connector_package_reports_manifest_version() -> None:
    assert connector_version() == "0.4.4"


def test_connector_ui_waits_for_the_instance_that_owns_the_import() -> None:
    connector_ui = (
        Path(__file__).resolve().parents[1] / "layerread" / "connector_ui.py"
    ).read_text(encoding="utf-8")

    assert "removeImportToken(window.top)" in connector_ui
    assert '"import_token": import_token' in connector_ui
    assert '"ack_import_token": ack_import_token' in connector_ui
    assert 'setTriggerValue("acknowledged", ackToken)' in connector_ui
    assert "仍在等待正确的 Connector 实例" in connector_ui
    assert "missingConnectorVersions.add(version)" in connector_ui
    failed_branch = connector_ui.split("if (!message.ok) {", 1)[1].split(
        "clearTimeout(timeoutId)", 1
    )[0]
    assert "clearImportToken()" not in failed_branch
    assert 'setTriggerValue("unavailable", true)' not in failed_branch


@pytest.mark.parametrize(
    "origin",
    (
        "http://layerread-demo.example",
        "https://layerread-demo.example/path",
        "https://user@example.com",
        "https://layerread-demo.example?target=other",
    ),
)
def test_demo_connector_build_rejects_non_origin_targets(origin) -> None:
    with pytest.raises(ValueError):
        normalize_https_origin(origin)


def test_demo_connector_build_rejects_output_inside_source_extension() -> None:
    with pytest.raises(ValueError, match="outside the source extension"):
        build_demo_connector(
            "https://layerread-demo.example",
            EXTENSION_DIR / "generated",
        )

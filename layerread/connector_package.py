"""Build a narrowly scoped Chrome Connector archive for one hosted origin."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXTENSION = PROJECT_ROOT / "browser_extension" / "layerread_connector"
ARCHIVE_ROOT = "layerread-online-connector"


def connector_version(source_extension: Path = SOURCE_EXTENSION) -> str:
    """Return the packaged Chrome extension version."""

    manifest = json.loads(
        (source_extension / "manifest.json").read_text(encoding="utf-8")
    )
    return str(manifest["version"])


def normalize_https_origin(value: str) -> str:
    """Return one exact HTTPS origin and reject broader URL inputs."""

    parsed = urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Demo origin must be one exact HTTPS origin without credentials, "
            "path, query, or fragment."
        )
    return f"https://{parsed.netloc.lower()}"


def online_connector_files(
    origin: str,
    source_extension: Path = SOURCE_EXTENSION,
) -> dict[str, bytes]:
    """Return extension files with only the requested online origin added."""

    normalized = normalize_https_origin(origin)
    files = {
        path.relative_to(source_extension).as_posix(): path.read_bytes()
        for path in source_extension.rglob("*")
        if path.is_file()
    }
    files["config.js"] = (
        f"globalThis.LAYERREAD_ONLINE_ORIGIN = {json.dumps(normalized)}\n"
    ).encode("utf-8")

    manifest = json.loads(files["manifest.json"].decode("utf-8"))
    online_pattern = f"{normalized}/*"
    manifest["host_permissions"].append(online_pattern)
    bridge_script = next(
        script
        for script in manifest["content_scripts"]
        if "bridge.js" in script.get("js", [])
    )
    bridge_script["matches"].append(online_pattern)
    files["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return files


def build_online_connector_archive(origin: str) -> bytes:
    """Create an unpack-and-load ZIP for the hosted Connector."""

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for relative_path, content in sorted(online_connector_files(origin).items()):
            archive.writestr(f"{ARCHIVE_ROOT}/{relative_path}", content)
    return buffer.getvalue()

"""Build a Chrome Connector package for one exact hosted LayerRead origin."""

from __future__ import annotations

import argparse
from pathlib import Path

from layerread.connector_package import (
    SOURCE_EXTENSION,
    normalize_https_origin,
    online_connector_files,
)


def build_demo_connector(origin: str, output_dir: Path) -> Path:
    normalize_https_origin(origin)
    output_dir = output_dir.resolve()
    if output_dir == SOURCE_EXTENSION or SOURCE_EXTENSION in output_dir.parents:
        raise ValueError("Output directory must be outside the source extension.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    for relative_path, content in online_connector_files(origin).items():
        target = output_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", required=True, help="Exact HTTPS Demo origin")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    built = build_demo_connector(args.origin, args.output)
    print(f"Built exact-origin Connector at {built}")


if __name__ == "__main__":
    main()

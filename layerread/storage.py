"""SQLite persistence for LayerRead article learning snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

from layerread.analysis_schema import Analysis
from layerread.article import ArticleDraft, ArticleImage
from layerread.chat_schema import ChatSession
from layerread.learning_schema import LearningSession
from layerread.visualization import VisualizationBundle


SCHEMA_VERSION = 2
DEFAULT_DATABASE_PATH = Path("data") / "layerread.sqlite3"
LEGACY_DATABASE_PATH = Path("data") / "deepread.sqlite3"


class StorageError(RuntimeError):
    """Safe, user-facing local-storage failure."""


@dataclass(frozen=True, slots=True)
class LocalArticleSummary:
    record_id: str
    title: str
    paragraph_count: int
    has_analysis: bool
    has_learning: bool
    has_chat: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class LocalSnapshot:
    record_id: str
    title: str
    article: ArticleDraft
    analysis: Analysis | None
    learning_session: LearningSession | None
    chat_session: ChatSession | None
    visualizations: VisualizationBundle | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class NotionExportRecord:
    export_id: int
    article_id: str
    target_id: str
    exported_sections: tuple[str, ...]
    notion_page_id: str
    notion_page_url: str
    status: str
    error_message: str
    exported_at: datetime


def database_path() -> Path:
    configured = os.environ.get("LAYERREAD_DB_PATH", "").strip()
    if configured:
        return Path(configured)

    legacy_configured = os.environ.get("DEEPREAD_DB_PATH", "").strip()
    if legacy_configured:
        return Path(legacy_configured)

    if not DEFAULT_DATABASE_PATH.exists() and LEGACY_DATABASE_PATH.is_file():
        _migrate_legacy_database(LEGACY_DATABASE_PATH, DEFAULT_DATABASE_PATH)
    return DEFAULT_DATABASE_PATH


def _migrate_legacy_database(source: Path, destination: Path) -> None:
    """Copy a legacy database atomically while preserving the original file."""

    if destination.exists() or not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.migrating"
    )
    try:
        legacy_connection = sqlite3.connect(source)
        new_connection = sqlite3.connect(temporary)
        try:
            legacy_connection.backup(new_connection)
        finally:
            new_connection.close()
            legacy_connection.close()
        temporary.replace(destination)
    except (OSError, sqlite3.Error) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise StorageError(
            "旧版本地数据库迁移失败；原文件保持不变，请检查 data 目录权限。"
        ) from exc


def article_record_id(article: ArticleDraft) -> str:
    source = "\n".join(
        [
            article.numbered_text,
            article.reading_goal,
            article.familiarity,
            article.focus_area,
            str(article.media_dependency),
            *(hashlib.sha256(image.data_url.encode("ascii")).hexdigest() for image in article.images),
        ]
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def article_title(article: ArticleDraft, max_length: int = 56) -> str:
    if article.source_title.strip():
        source_title = " ".join(article.source_title.split())
        if len(source_title) <= max_length:
            return source_title
        return source_title[: max_length - 1].rstrip() + "…"
    if not article.numbered_paragraphs:
        return "未命名文章"
    _, _, first_paragraph = article.numbered_paragraphs[0].partition(" ")
    compact = " ".join(first_paragraph.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "…"


def _article_to_json(article: ArticleDraft) -> str:
    payload = {
        "raw_text": article.raw_text,
        "clean_text": article.clean_text,
        "numbered_paragraphs": list(article.numbered_paragraphs),
        "removed_paragraphs": list(article.removed_paragraphs),
        "reading_goal": article.reading_goal,
        "familiarity": article.familiarity,
        "focus_area": article.focus_area,
        "media_dependency": article.media_dependency,
        "media_completeness": article.media_completeness,
        "raw_char_count": article.raw_char_count,
        "clean_char_count": article.clean_char_count,
        "created_at": article.created_at.isoformat(),
        "source_title": article.source_title,
        "source_url": article.source_url,
        "account": article.account,
        "author": article.author,
        "published_at": article.published_at,
        "extraction_method": article.extraction_method,
        "extraction_status": article.extraction_status,
        "images": [
            {
                "image_id": image.image_id,
                "data_url": image.data_url,
                "mime_type": image.mime_type,
                "width": image.width,
                "height": image.height,
                "alt_text": image.alt_text,
                "caption": image.caption,
                "context_before": image.context_before,
                "context_after": image.context_after,
                "source_url": image.source_url,
                "tile_index": image.tile_index,
                "tile_count": image.tile_count,
                "original_width": image.original_width,
                "original_height": image.original_height,
            }
            for image in article.images
        ],
        "image_count_detected": article.image_count_detected,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _article_from_json(raw: str) -> ArticleDraft:
    payload = json.loads(raw)
    return ArticleDraft(
        raw_text=payload["raw_text"],
        clean_text=payload["clean_text"],
        numbered_paragraphs=tuple(payload["numbered_paragraphs"]),
        removed_paragraphs=tuple(payload["removed_paragraphs"]),
        reading_goal=payload["reading_goal"],
        familiarity=payload["familiarity"],
        focus_area=payload["focus_area"],
        media_dependency=payload["media_dependency"],
        media_completeness=payload["media_completeness"],
        raw_char_count=payload["raw_char_count"],
        clean_char_count=payload["clean_char_count"],
        created_at=datetime.fromisoformat(payload["created_at"]),
        source_title=payload.get("source_title", ""),
        source_url=payload.get("source_url", ""),
        account=payload.get("account", ""),
        author=payload.get("author", ""),
        published_at=payload.get("published_at", ""),
        extraction_method=payload.get("extraction_method", "manual_paste"),
        extraction_status=payload.get("extraction_status", "confirmed"),
        images=tuple(
            ArticleImage(
                image_id=image["image_id"],
                data_url=image["data_url"],
                mime_type=image["mime_type"],
                width=image["width"],
                height=image["height"],
                alt_text=image.get("alt_text", ""),
                caption=image.get("caption", ""),
                context_before=image.get("context_before", ""),
                context_after=image.get("context_after", ""),
                source_url=image.get("source_url", ""),
                tile_index=image.get("tile_index", 1),
                tile_count=image.get("tile_count", 1),
                original_width=image.get("original_width", image["width"]),
                original_height=image.get("original_height", image["height"]),
            )
            for image in payload.get("images", [])
        ),
        image_count_detected=payload.get("image_count_detected", 0),
    )


def _model_json(model: Analysis | LearningSession | ChatSession | VisualizationBundle | None) -> str | None:
    return model.model_dump_json() if model is not None else None


def _content_hash(parts: list[str | None]) -> str:
    payload = "\x1e".join(part if part is not None else "<none>" for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or database_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(path: Path | None = None) -> None:
    try:
        with _connect(path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            current = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()[0]
            if current > SCHEMA_VERSION:
                raise StorageError("本地数据库版本高于当前应用支持版本。")
            if current < 1:
                connection.executescript(
                    """
                    CREATE TABLE articles (
                        record_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        paragraph_count INTEGER NOT NULL,
                        article_json TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE analyses (
                        record_id TEXT PRIMARY KEY REFERENCES articles(record_id) ON DELETE CASCADE,
                        payload_json TEXT NOT NULL
                    );
                    CREATE TABLE learning_sessions (
                        record_id TEXT PRIMARY KEY REFERENCES articles(record_id) ON DELETE CASCADE,
                        payload_json TEXT NOT NULL
                    );
                    CREATE TABLE chat_sessions (
                        record_id TEXT PRIMARY KEY REFERENCES articles(record_id) ON DELETE CASCADE,
                        payload_json TEXT NOT NULL
                    );
                    CREATE TABLE visualizations (
                        record_id TEXT PRIMARY KEY REFERENCES articles(record_id) ON DELETE CASCADE,
                        payload_json TEXT NOT NULL
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, datetime.now(UTC).isoformat()),
                )
                current = 1
            if current < 2:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS notion_exports (
                        export_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id TEXT NOT NULL REFERENCES articles(record_id) ON DELETE CASCADE,
                        target_id TEXT NOT NULL,
                        exported_sections TEXT NOT NULL,
                        notion_page_id TEXT NOT NULL DEFAULT '',
                        notion_page_url TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL,
                        error_message TEXT NOT NULL DEFAULT '',
                        exported_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (2, datetime.now(UTC).isoformat()),
                )
    except StorageError:
        raise
    except (OSError, sqlite3.Error) as exc:
        raise StorageError(f"无法初始化本地数据库：{exc}") from exc


def _replace_optional_payload(
    connection: sqlite3.Connection,
    table: str,
    record_id: str,
    payload: str | None,
) -> None:
    if payload is None:
        connection.execute(f"DELETE FROM {table} WHERE record_id = ?", (record_id,))
        return
    connection.execute(
        f"""
        INSERT INTO {table}(record_id, payload_json) VALUES (?, ?)
        ON CONFLICT(record_id) DO UPDATE SET payload_json = excluded.payload_json
        """,
        (record_id, payload),
    )


def save_snapshot(
    article: ArticleDraft,
    analysis: Analysis | None,
    learning_session: LearningSession | None,
    chat_session: ChatSession | None,
    visualizations: VisualizationBundle | None,
    path: Path | None = None,
) -> LocalArticleSummary:
    """Atomically save the current article and every related asset."""

    if analysis is None and (learning_session is not None or chat_session is not None):
        raise StorageError("学习或 Chat 数据缺少对应的文章分析，无法保存。")
    if analysis is not None:
        for session in (learning_session, chat_session):
            if session is not None and session.article_id != analysis.article_id:
                raise StorageError("待保存的学习资产与当前文章分析不一致。")

    initialize_database(path)
    record_id = article_record_id(article)
    title = article_title(article)
    article_json = _article_to_json(article)
    analysis_json = _model_json(analysis)
    learning_json = _model_json(learning_session)
    chat_json = _model_json(chat_session)
    visualization_json = _model_json(visualizations)
    content_hash = _content_hash(
        [article_json, analysis_json, learning_json, chat_json, visualization_json]
    )
    now = datetime.now(UTC)

    try:
        with _connect(path) as connection:
            existing = connection.execute(
                "SELECT created_at, updated_at, content_hash FROM articles WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if existing is not None and existing["content_hash"] == content_hash:
                return LocalArticleSummary(
                    record_id=record_id,
                    title=title,
                    paragraph_count=article.paragraph_count,
                    has_analysis=analysis is not None,
                    has_learning=learning_session is not None,
                    has_chat=chat_session is not None,
                    created_at=datetime.fromisoformat(existing["created_at"]),
                    updated_at=datetime.fromisoformat(existing["updated_at"]),
                )

            created_at = (
                datetime.fromisoformat(existing["created_at"])
                if existing is not None
                else now
            )
            connection.execute(
                """
                INSERT INTO articles(
                    record_id, title, paragraph_count, article_json,
                    content_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    title = excluded.title,
                    paragraph_count = excluded.paragraph_count,
                    article_json = excluded.article_json,
                    content_hash = excluded.content_hash,
                    updated_at = excluded.updated_at
                """,
                (
                    record_id,
                    title,
                    article.paragraph_count,
                    article_json,
                    content_hash,
                    created_at.isoformat(),
                    now.isoformat(),
                ),
            )
            _replace_optional_payload(connection, "analyses", record_id, analysis_json)
            _replace_optional_payload(
                connection,
                "learning_sessions",
                record_id,
                learning_json,
            )
            _replace_optional_payload(connection, "chat_sessions", record_id, chat_json)
            _replace_optional_payload(
                connection,
                "visualizations",
                record_id,
                visualization_json,
            )
    except (OSError, sqlite3.Error) as exc:
        raise StorageError(f"本地保存失败，当前会话内容仍已保留：{exc}") from exc

    return LocalArticleSummary(
        record_id=record_id,
        title=title,
        paragraph_count=article.paragraph_count,
        has_analysis=analysis is not None,
        has_learning=learning_session is not None,
        has_chat=chat_session is not None,
        created_at=created_at,
        updated_at=now,
    )


def _optional_payload(
    connection: sqlite3.Connection,
    table: str,
    record_id: str,
) -> str | None:
    row = connection.execute(
        f"SELECT payload_json FROM {table} WHERE record_id = ?",
        (record_id,),
    ).fetchone()
    return row["payload_json"] if row is not None else None


def load_snapshot(record_id: str, path: Path | None = None) -> LocalSnapshot:
    initialize_database(path)
    try:
        with _connect(path) as connection:
            row = connection.execute(
                "SELECT * FROM articles WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise StorageError("找不到指定的本地文章记录。")
            analysis_json = _optional_payload(connection, "analyses", record_id)
            learning_json = _optional_payload(
                connection,
                "learning_sessions",
                record_id,
            )
            chat_json = _optional_payload(connection, "chat_sessions", record_id)
            visualization_json = _optional_payload(
                connection,
                "visualizations",
                record_id,
            )
        return LocalSnapshot(
            record_id=record_id,
            title=row["title"],
            article=_article_from_json(row["article_json"]),
            analysis=(Analysis.model_validate_json(analysis_json) if analysis_json else None),
            learning_session=(
                LearningSession.model_validate_json(learning_json)
                if learning_json
                else None
            ),
            chat_session=(
                ChatSession.model_validate_json(chat_json) if chat_json else None
            ),
            visualizations=(
                VisualizationBundle.model_validate_json(visualization_json)
                if visualization_json
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
    except StorageError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
        raise StorageError(f"本地文章记录损坏或无法读取：{exc}") from exc


def list_articles(path: Path | None = None) -> list[LocalArticleSummary]:
    initialize_database(path)
    try:
        with _connect(path) as connection:
            rows = connection.execute(
                """
                SELECT
                    a.record_id,
                    a.title,
                    a.paragraph_count,
                    a.created_at,
                    a.updated_at,
                    EXISTS(SELECT 1 FROM analyses x WHERE x.record_id = a.record_id) AS has_analysis,
                    EXISTS(SELECT 1 FROM learning_sessions x WHERE x.record_id = a.record_id) AS has_learning,
                    EXISTS(SELECT 1 FROM chat_sessions x WHERE x.record_id = a.record_id) AS has_chat
                FROM articles a
                ORDER BY a.updated_at DESC, a.rowid DESC
                """
            ).fetchall()
        return [
            LocalArticleSummary(
                record_id=row["record_id"],
                title=row["title"],
                paragraph_count=row["paragraph_count"],
                has_analysis=bool(row["has_analysis"]),
                has_learning=bool(row["has_learning"]),
                has_chat=bool(row["has_chat"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]
    except sqlite3.Error as exc:
        raise StorageError(f"无法读取本地文章列表：{exc}") from exc


def load_latest_snapshot(path: Path | None = None) -> LocalSnapshot | None:
    articles = list_articles(path)
    return load_snapshot(articles[0].record_id, path) if articles else None


def record_notion_export(
    article_id: str,
    target_id: str,
    exported_sections: list[str],
    *,
    status: str,
    notion_page_id: str = "",
    notion_page_url: str = "",
    error_message: str = "",
    path: Path | None = None,
) -> NotionExportRecord:
    """Record an export outcome without ever accepting or storing a token."""

    if status not in {"success", "failed"}:
        raise StorageError("Notion 导出状态无效。")
    initialize_database(path)
    exported_at = datetime.now(UTC)
    sections_json = json.dumps(exported_sections, ensure_ascii=False)
    try:
        with _connect(path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO notion_exports(
                    article_id, target_id, exported_sections, notion_page_id,
                    notion_page_url, status, error_message, exported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_id,
                    target_id,
                    sections_json,
                    notion_page_id,
                    notion_page_url,
                    status,
                    error_message,
                    exported_at.isoformat(),
                ),
            )
            export_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise StorageError("当前文章尚未保存，暂时无法记录 Notion 导出结果。") from exc
    except sqlite3.Error as exc:
        raise StorageError(f"无法记录 Notion 导出结果：{exc}") from exc
    return NotionExportRecord(
        export_id=export_id,
        article_id=article_id,
        target_id=target_id,
        exported_sections=tuple(exported_sections),
        notion_page_id=notion_page_id,
        notion_page_url=notion_page_url,
        status=status,
        error_message=error_message,
        exported_at=exported_at,
    )


def list_notion_exports(
    article_id: str,
    path: Path | None = None,
) -> list[NotionExportRecord]:
    initialize_database(path)
    try:
        with _connect(path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM notion_exports
                WHERE article_id = ?
                ORDER BY exported_at DESC, export_id DESC
                """,
                (article_id,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"无法读取 Notion 导出记录：{exc}") from exc
    return [
        NotionExportRecord(
            export_id=row["export_id"],
            article_id=row["article_id"],
            target_id=row["target_id"],
            exported_sections=tuple(json.loads(row["exported_sections"])),
            notion_page_id=row["notion_page_id"],
            notion_page_url=row["notion_page_url"],
            status=row["status"],
            error_message=row["error_message"],
            exported_at=datetime.fromisoformat(row["exported_at"]),
        )
        for row in rows
    ]

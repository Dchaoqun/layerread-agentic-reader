"""Hosted-demo policy, content limits, and metadata-only quota accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Literal, Protocol, TypeVar
import uuid

import psycopg


logger = logging.getLogger(__name__)


DemoAction = Literal[
    "analysis",
    "chat",
    "learning_questions",
    "learning_feedback",
    "chat_digest",
]

_ACTION_DEFAULT_LIMITS: dict[DemoAction, int] = {
    "analysis": 2,
    "chat": 10,
    "learning_questions": 1,
    "learning_feedback": 1,
    "chat_digest": 1,
}
_ACTION_DEFAULT_UNITS: dict[DemoAction, int] = {
    "analysis": 5,
    "chat": 1,
    "learning_questions": 2,
    "learning_feedback": 2,
    "chat_digest": 2,
}
_ACTION_ENV_NAMES: dict[DemoAction, str] = {
    "analysis": "LAYERREAD_DEMO_DAILY_ANALYSIS_LIMIT",
    "chat": "LAYERREAD_DEMO_DAILY_CHAT_LIMIT",
    "learning_questions": "LAYERREAD_DEMO_DAILY_LEARNING_LIMIT",
    "learning_feedback": "LAYERREAD_DEMO_DAILY_FEEDBACK_LIMIT",
    "chat_digest": "LAYERREAD_DEMO_DAILY_DIGEST_LIMIT",
}


def _visitor_attempt_limit(action: DemoAction, success_limit: int) -> int:
    """Bound compensated failures without reducing successful-use allowance."""

    if action == "analysis":
        return success_limit * 2
    return success_limit + 1


def _attempts_from_cost_units(cost_units: int, units_per_attempt: int) -> int:
    return max(0, cost_units) // units_per_attempt


def _attempt_limit_message(action: DemoAction) -> str:
    if action == "analysis":
        return (
            "今天的文章分析尝试次数已达到安全上限。失败不会占用成功分析次数，"
            "但为保护在线 Demo 预算，失败尝试不能无限重试。"
        )
    return "今天的这项模型功能尝试次数已达到安全上限，请明天再试。"


def _positive_int(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _enabled(source: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = source.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class DemoPolicy:
    """Configurable limits for a deployment-owned online demo."""

    quota_db_path: Path
    quota_database_url: str | None
    require_persistent_quota: bool
    max_article_chars: int
    max_images: int
    max_total_image_bytes: int
    global_daily_calls: int
    global_daily_cost_units: int
    metadata_retention_days: int
    action_limits: Mapping[DemoAction, int]
    action_units: Mapping[DemoAction, int]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DemoPolicy":
        source = os.environ if env is None else env
        return cls(
            quota_db_path=Path(
                source.get(
                    "LAYERREAD_DEMO_QUOTA_DB_PATH",
                    "data/layerread-demo-usage.sqlite3",
                )
            ),
            quota_database_url=(
                source.get("LAYERREAD_DEMO_QUOTA_DATABASE_URL", "").strip() or None
            ),
            require_persistent_quota=_enabled(
                source,
                "LAYERREAD_DEMO_REQUIRE_PERSISTENT_QUOTA",
            ),
            max_article_chars=_positive_int(
                source, "LAYERREAD_DEMO_MAX_ARTICLE_CHARS", 50_000
            ),
            max_images=_positive_int(source, "LAYERREAD_DEMO_MAX_IMAGES", 8),
            max_total_image_bytes=_positive_int(
                source, "LAYERREAD_DEMO_MAX_TOTAL_IMAGE_BYTES", 4 * 1024 * 1024
            ),
            global_daily_calls=_positive_int(
                source, "LAYERREAD_DEMO_GLOBAL_DAILY_CALLS", 100
            ),
            global_daily_cost_units=_positive_int(
                source, "LAYERREAD_DEMO_GLOBAL_DAILY_COST_UNITS", 200
            ),
            metadata_retention_days=_positive_int(
                source, "LAYERREAD_DEMO_METADATA_RETENTION_DAYS", 31
            ),
            action_limits={
                action: _positive_int(source, env_name, _ACTION_DEFAULT_LIMITS[action])
                for action, env_name in _ACTION_ENV_NAMES.items()
            },
            action_units=dict(_ACTION_DEFAULT_UNITS),
        )


@dataclass(frozen=True, slots=True)
class QuotaDecision:
    allowed: bool
    action: DemoAction
    remaining: int
    message: str = ""
    reservation_id: str = ""


class _ImageLike(Protocol):
    data_url: str


ImageT = TypeVar("ImageT", bound=_ImageLike)


@dataclass(frozen=True, slots=True)
class ImageLimitResult:
    images: tuple[ImageT, ...]
    dropped_count: int
    kept_bytes: int


def _data_url_bytes(data_url: str) -> int:
    """Estimate decoded bytes without materializing another image copy."""

    _, separator, payload = data_url.partition(",")
    if not separator:
        return len(data_url.encode("utf-8"))
    padding = len(payload) - len(payload.rstrip("="))
    return max(0, (len(payload) * 3) // 4 - padding)


def limit_images(
    images: Sequence[ImageT],
    policy: DemoPolicy,
) -> ImageLimitResult:
    """Keep the leading images that fit both the count and byte budgets."""

    kept: list[ImageT] = []
    kept_bytes = 0
    for image in images:
        image_bytes = _data_url_bytes(image.data_url)
        if len(kept) >= policy.max_images:
            break
        if kept_bytes + image_bytes > policy.max_total_image_bytes:
            break
        kept.append(image)
        kept_bytes += image_bytes
    return ImageLimitResult(
        images=tuple(kept),
        dropped_count=max(0, len(images) - len(kept)),
        kept_bytes=kept_bytes,
    )


class _SQLiteDemoQuotaStore:
    """Atomic quota accounting that never stores article or conversation data."""

    def __init__(self, policy: DemoPolicy):
        self.policy = policy

    def _connect(self) -> sqlite3.Connection:
        self.policy.quota_db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.policy.quota_db_path, timeout=10)
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS demo_visitor_usage (
                usage_day TEXT NOT NULL,
                visitor_hash TEXT NOT NULL,
                action TEXT NOT NULL,
                action_count INTEGER NOT NULL,
                cost_units INTEGER NOT NULL,
                PRIMARY KEY (usage_day, visitor_hash, action)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS demo_global_usage (
                usage_day TEXT PRIMARY KEY,
                call_count INTEGER NOT NULL,
                cost_units INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS demo_analysis_reservations (
                reservation_id TEXT PRIMARY KEY,
                usage_day TEXT NOT NULL,
                visitor_hash TEXT NOT NULL,
                action TEXT NOT NULL,
                released INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cutoff = (
            date.today() - timedelta(days=self.policy.metadata_retention_days)
        ).isoformat()
        connection.execute(
            "DELETE FROM demo_visitor_usage WHERE usage_day < ?",
            (cutoff,),
        )
        connection.execute(
            "DELETE FROM demo_global_usage WHERE usage_day < ?",
            (cutoff,),
        )
        connection.execute(
            "DELETE FROM demo_analysis_reservations WHERE usage_day < ?",
            (cutoff,),
        )
        connection.commit()
        return connection

    def status(
        self,
        visitor_hash: str,
        action: DemoAction,
        *,
        usage_day: str | None = None,
    ) -> QuotaDecision:
        day = usage_day or date.today().isoformat()
        limit = self.policy.action_limits[action]
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT action_count, cost_units
                FROM demo_visitor_usage
                WHERE usage_day = ? AND visitor_hash = ? AND action = ?
                """,
                (day, visitor_hash, action),
            ).fetchone()
            global_row = connection.execute(
                """
                SELECT call_count, cost_units
                FROM demo_global_usage
                WHERE usage_day = ?
                """,
                (day,),
            ).fetchone()
        used, visitor_units = tuple(map(int, row)) if row else (0, 0)
        calls, units = (map(int, global_row) if global_row else (0, 0))
        remaining = max(0, limit - used)
        if remaining == 0:
            return QuotaDecision(
                False,
                action,
                0,
                "今天的这项在线体验额度已经用完，请明天再试。",
            )
        attempts = _attempts_from_cost_units(
            visitor_units,
            self.policy.action_units[action],
        )
        if attempts >= _visitor_attempt_limit(action, limit):
            return QuotaDecision(
                False,
                action,
                remaining,
                _attempt_limit_message(action),
            )
        if calls >= self.policy.global_daily_calls or (
            units + self.policy.action_units[action]
            > self.policy.global_daily_cost_units
        ):
            return QuotaDecision(
                False,
                action,
                remaining,
                "在线 Demo 今日预算已用完，现有结果仍可查看和导出。",
            )
        return QuotaDecision(True, action, remaining)

    def claim(
        self,
        visitor_hash: str,
        action: DemoAction,
        *,
        usage_day: str | None = None,
    ) -> QuotaDecision:
        """Atomically reserve a success slot and record one protected attempt."""

        day = usage_day or date.today().isoformat()
        limit = self.policy.action_limits[action]
        units = self.policy.action_units[action]
        reservation_id = str(uuid.uuid4()) if action == "analysis" else ""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT action_count, cost_units
                FROM demo_visitor_usage
                WHERE usage_day = ? AND visitor_hash = ? AND action = ?
                """,
                (day, visitor_hash, action),
            ).fetchone()
            global_row = connection.execute(
                """
                SELECT call_count, cost_units
                FROM demo_global_usage
                WHERE usage_day = ?
                """,
                (day,),
            ).fetchone()
            used, visitor_units = tuple(map(int, row)) if row else (0, 0)
            global_calls, global_units = (
                tuple(map(int, global_row)) if global_row else (0, 0)
            )
            if used >= limit:
                connection.rollback()
                return QuotaDecision(
                    False,
                    action,
                    0,
                    "今天的这项在线体验额度已经用完，请明天再试。",
                )
            attempts = _attempts_from_cost_units(visitor_units, units)
            if attempts >= _visitor_attempt_limit(action, limit):
                connection.rollback()
                return QuotaDecision(
                    False,
                    action,
                    max(0, limit - used),
                    _attempt_limit_message(action),
                )
            if global_calls >= self.policy.global_daily_calls or (
                global_units + units > self.policy.global_daily_cost_units
            ):
                connection.rollback()
                return QuotaDecision(
                    False,
                    action,
                    max(0, limit - used),
                    "在线 Demo 今日预算已用完，现有结果仍可查看和导出。",
                )
            connection.execute(
                """
                INSERT INTO demo_visitor_usage (
                    usage_day, visitor_hash, action, action_count, cost_units
                ) VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(usage_day, visitor_hash, action) DO UPDATE SET
                    action_count = action_count + 1,
                    cost_units = cost_units + excluded.cost_units
                """,
                (day, visitor_hash, action, units),
            )
            connection.execute(
                """
                INSERT INTO demo_global_usage (usage_day, call_count, cost_units)
                VALUES (?, 1, ?)
                ON CONFLICT(usage_day) DO UPDATE SET
                    call_count = call_count + 1,
                    cost_units = cost_units + excluded.cost_units
                """,
                (day, units),
            )
            if reservation_id:
                connection.execute(
                    """
                    INSERT INTO demo_analysis_reservations (
                        reservation_id, usage_day, visitor_hash, action, released
                    ) VALUES (?, ?, ?, ?, 0)
                    """,
                    (reservation_id, day, visitor_hash, action),
                )
            connection.commit()
            return QuotaDecision(
                True,
                action,
                max(0, limit - used - 1),
                reservation_id=reservation_id,
            )
        finally:
            connection.close()

    def release_failed(
        self,
        visitor_hash: str,
        action: DemoAction,
        *,
        usage_day: str | None = None,
        reservation_id: str = "",
    ) -> bool:
        """Return a successful-use slot while retaining attempt and global cost."""

        day = usage_day or date.today().isoformat()
        with self._connect() as connection:
            if reservation_id:
                reservation = connection.execute(
                    """
                    UPDATE demo_analysis_reservations
                    SET released = 1
                    WHERE reservation_id = ? AND usage_day = ?
                      AND visitor_hash = ? AND action = ? AND released = 0
                    """,
                    (reservation_id, day, visitor_hash, action),
                )
                if reservation.rowcount == 0:
                    row = connection.execute(
                        """
                        SELECT released FROM demo_analysis_reservations
                        WHERE reservation_id = ? AND usage_day = ?
                          AND visitor_hash = ? AND action = ?
                        """,
                        (reservation_id, day, visitor_hash, action),
                    ).fetchone()
                    return bool(row and row[0] == 1)
            cursor = connection.execute(
                """
                UPDATE demo_visitor_usage
                SET action_count = action_count - 1
                WHERE usage_day = ? AND visitor_hash = ? AND action = ?
                  AND action_count > 0
                """,
                (day, visitor_hash, action),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return False
            connection.commit()
            return True


class QuotaStoreUnavailable(RuntimeError):
    """Raised when a deployment requires a durable quota store but has none."""


_POSTGRES_SCHEMA_LOCK = threading.Lock()
_POSTGRES_MAINTAINED_ON: dict[str, str] = {}


class _PostgresDemoQuotaStore:
    """Durable quota accounting for hosted deployments using PostgreSQL."""

    def __init__(self, policy: DemoPolicy):
        assert policy.quota_database_url is not None
        self.policy = policy
        self.database_url = policy.quota_database_url
        self.database_fingerprint = hashlib.sha256(
            self.database_url.encode("utf-8")
        ).hexdigest()

    def _connect(self):
        return psycopg.connect(self.database_url, connect_timeout=10)

    def _ensure_schema(self, connection) -> None:
        maintenance_day = date.today().isoformat()
        if _POSTGRES_MAINTAINED_ON.get(self.database_fingerprint) == maintenance_day:
            return
        with _POSTGRES_SCHEMA_LOCK:
            if (
                _POSTGRES_MAINTAINED_ON.get(self.database_fingerprint)
                == maintenance_day
            ):
                return
            cutoff = (
                date.today() - timedelta(days=self.policy.metadata_retention_days)
            ).isoformat()
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS demo_visitor_usage (
                        usage_day TEXT NOT NULL,
                        visitor_hash TEXT NOT NULL,
                        action TEXT NOT NULL,
                        action_count INTEGER NOT NULL CHECK (action_count >= 0),
                        cost_units INTEGER NOT NULL CHECK (cost_units >= 0),
                        PRIMARY KEY (usage_day, visitor_hash, action)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS demo_global_usage (
                        usage_day TEXT PRIMARY KEY,
                        call_count INTEGER NOT NULL CHECK (call_count >= 0),
                        cost_units INTEGER NOT NULL CHECK (cost_units >= 0)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS demo_analysis_reservations (
                        reservation_id TEXT PRIMARY KEY,
                        usage_day TEXT NOT NULL,
                        visitor_hash TEXT NOT NULL,
                        action TEXT NOT NULL,
                        released BOOLEAN NOT NULL DEFAULT FALSE
                    )
                    """
                )
                cursor.execute(
                    "DELETE FROM demo_visitor_usage WHERE usage_day < %s",
                    (cutoff,),
                )
                cursor.execute(
                    "DELETE FROM demo_global_usage WHERE usage_day < %s",
                    (cutoff,),
                )
                cursor.execute(
                    "DELETE FROM demo_analysis_reservations WHERE usage_day < %s",
                    (cutoff,),
                )
            connection.commit()
            _POSTGRES_MAINTAINED_ON[self.database_fingerprint] = maintenance_day

    def status(
        self,
        visitor_hash: str,
        action: DemoAction,
        *,
        usage_day: str | None = None,
    ) -> QuotaDecision:
        day = usage_day or date.today().isoformat()
        limit = self.policy.action_limits[action]
        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT action_count, cost_units
                    FROM demo_visitor_usage
                    WHERE usage_day = %s AND visitor_hash = %s AND action = %s
                    """,
                    (day, visitor_hash, action),
                )
                row = cursor.fetchone()
                cursor.execute(
                    """
                    SELECT call_count, cost_units
                    FROM demo_global_usage
                    WHERE usage_day = %s
                    """,
                    (day,),
                )
                global_row = cursor.fetchone()
        used, visitor_units = tuple(map(int, row)) if row else (0, 0)
        calls, units = (map(int, global_row) if global_row else (0, 0))
        remaining = max(0, limit - used)
        if remaining == 0:
            return QuotaDecision(
                False,
                action,
                0,
                "今天的这项在线体验额度已经用完，请明天再试。",
            )
        attempts = _attempts_from_cost_units(
            visitor_units,
            self.policy.action_units[action],
        )
        if attempts >= _visitor_attempt_limit(action, limit):
            return QuotaDecision(
                False,
                action,
                remaining,
                _attempt_limit_message(action),
            )
        if calls >= self.policy.global_daily_calls or (
            units + self.policy.action_units[action]
            > self.policy.global_daily_cost_units
        ):
            return QuotaDecision(
                False,
                action,
                remaining,
                "在线 Demo 今日预算已用完，现有结果仍可查看和导出。",
            )
        return QuotaDecision(True, action, remaining)

    def claim(
        self,
        visitor_hash: str,
        action: DemoAction,
        *,
        usage_day: str | None = None,
    ) -> QuotaDecision:
        """Reserve one action while locking the daily global and visitor rows."""

        day = usage_day or date.today().isoformat()
        limit = self.policy.action_limits[action]
        units = self.policy.action_units[action]
        reservation_id = str(uuid.uuid4()) if action == "analysis" else ""
        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO demo_global_usage (usage_day, call_count, cost_units)
                    VALUES (%s, 0, 0)
                    ON CONFLICT (usage_day) DO NOTHING
                    """,
                    (day,),
                )
                cursor.execute(
                    """
                    INSERT INTO demo_visitor_usage (
                        usage_day, visitor_hash, action, action_count, cost_units
                    ) VALUES (%s, %s, %s, 0, 0)
                    ON CONFLICT (usage_day, visitor_hash, action) DO NOTHING
                    """,
                    (day, visitor_hash, action),
                )
                cursor.execute(
                    """
                    SELECT call_count, cost_units
                    FROM demo_global_usage
                    WHERE usage_day = %s
                    FOR UPDATE
                    """,
                    (day,),
                )
                global_row = cursor.fetchone()
                cursor.execute(
                    """
                    SELECT action_count, cost_units
                    FROM demo_visitor_usage
                    WHERE usage_day = %s AND visitor_hash = %s AND action = %s
                    FOR UPDATE
                    """,
                    (day, visitor_hash, action),
                )
                row = cursor.fetchone()
                if global_row is None or row is None:
                    raise QuotaStoreUnavailable("quota rows could not be initialized")
                global_calls, global_units = map(int, global_row)
                used, visitor_units = map(int, row)
                if used >= limit:
                    return QuotaDecision(
                        False,
                        action,
                        0,
                        "今天的这项在线体验额度已经用完，请明天再试。",
                    )
                attempts = _attempts_from_cost_units(visitor_units, units)
                if attempts >= _visitor_attempt_limit(action, limit):
                    return QuotaDecision(
                        False,
                        action,
                        max(0, limit - used),
                        _attempt_limit_message(action),
                    )
                if global_calls >= self.policy.global_daily_calls or (
                    global_units + units > self.policy.global_daily_cost_units
                ):
                    return QuotaDecision(
                        False,
                        action,
                        max(0, limit - used),
                        "在线 Demo 今日预算已用完，现有结果仍可查看和导出。",
                    )
                cursor.execute(
                    """
                    UPDATE demo_visitor_usage
                    SET action_count = action_count + 1,
                        cost_units = cost_units + %s
                    WHERE usage_day = %s AND visitor_hash = %s AND action = %s
                    """,
                    (units, day, visitor_hash, action),
                )
                cursor.execute(
                    """
                    UPDATE demo_global_usage
                    SET call_count = call_count + 1,
                        cost_units = cost_units + %s
                    WHERE usage_day = %s
                    """,
                    (units, day),
                )
                if reservation_id:
                    cursor.execute(
                        """
                        INSERT INTO demo_analysis_reservations (
                            reservation_id, usage_day, visitor_hash, action, released
                        ) VALUES (%s, %s, %s, %s, FALSE)
                        """,
                        (reservation_id, day, visitor_hash, action),
                    )
        return QuotaDecision(
            True,
            action,
            max(0, limit - used - 1),
            reservation_id=reservation_id,
        )

    def release_failed(
        self,
        visitor_hash: str,
        action: DemoAction,
        *,
        usage_day: str | None = None,
        reservation_id: str = "",
    ) -> bool:
        """Return a successful-use slot while retaining attempt and global cost."""

        day = usage_day or date.today().isoformat()
        with self._connect() as connection:
            self._ensure_schema(connection)
            with connection.cursor() as cursor:
                if reservation_id:
                    cursor.execute(
                        """
                        UPDATE demo_analysis_reservations
                        SET released = TRUE
                        WHERE reservation_id = %s AND usage_day = %s
                          AND visitor_hash = %s AND action = %s
                          AND released = FALSE
                        """,
                        (reservation_id, day, visitor_hash, action),
                    )
                    if cursor.rowcount == 0:
                        cursor.execute(
                            """
                            SELECT released FROM demo_analysis_reservations
                            WHERE reservation_id = %s AND usage_day = %s
                              AND visitor_hash = %s AND action = %s
                            """,
                            (reservation_id, day, visitor_hash, action),
                        )
                        row = cursor.fetchone()
                        return bool(row and row[0])
                cursor.execute(
                    """
                    UPDATE demo_visitor_usage
                    SET action_count = action_count - 1
                    WHERE usage_day = %s AND visitor_hash = %s AND action = %s
                      AND action_count > 0
                    """,
                    (day, visitor_hash, action),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    return False
            connection.commit()
            return True


class DemoQuotaStore:
    """Select a local or durable quota backend without storing user content."""

    def __init__(self, policy: DemoPolicy):
        self.policy = policy
        if policy.quota_database_url:
            self.backend = _PostgresDemoQuotaStore(policy)
        elif policy.require_persistent_quota:
            self.backend = None
        else:
            self.backend = _SQLiteDemoQuotaStore(policy)

    def _require_backend(self):
        if self.backend is None:
            raise QuotaStoreUnavailable(
                "persistent quota storage is required but not configured"
            )
        return self.backend

    def status(
        self,
        visitor_hash: str,
        action: DemoAction,
        *,
        usage_day: str | None = None,
    ) -> QuotaDecision:
        return self._require_backend().status(
            visitor_hash,
            action,
            usage_day=usage_day,
        )

    def claim(
        self,
        visitor_hash: str,
        action: DemoAction,
        *,
        usage_day: str | None = None,
    ) -> QuotaDecision:
        return self._require_backend().claim(
            visitor_hash,
            action,
            usage_day=usage_day,
        )

    def release_failed(
        self,
        visitor_hash: str,
        action: DemoAction,
        *,
        usage_day: str | None = None,
        reservation_id: str = "",
    ) -> bool:
        return self._require_backend().release_failed(
            visitor_hash,
            action,
            usage_day=usage_day,
            reservation_id=reservation_id,
        )


class DemoLimiter:
    """Visitor-scoped facade over the metadata-only quota store."""

    def __init__(self, policy: DemoPolicy, visitor_id: str):
        visitor_hash = hashlib.sha256(visitor_id.encode("utf-8")).hexdigest()
        self.policy = policy
        self.visitor_hash = visitor_hash
        self.store = DemoQuotaStore(policy)

    def status(self, action: DemoAction) -> QuotaDecision:
        try:
            return self.store.status(self.visitor_hash, action)
        except (OSError, sqlite3.Error, psycopg.Error, QuotaStoreUnavailable):
            return QuotaDecision(
                False,
                action,
                0,
                "在线体验限制服务暂不可用，为避免产生未受控费用，模型调用已暂停。",
            )

    def claim(self, action: DemoAction) -> QuotaDecision:
        try:
            return self.store.claim(self.visitor_hash, action)
        except (OSError, sqlite3.Error, psycopg.Error, QuotaStoreUnavailable):
            return QuotaDecision(
                False,
                action,
                0,
                "在线体验限制服务暂不可用，为避免产生未受控费用，模型调用已暂停。",
            )

    def release_failed(self, action: DemoAction, reservation_id: str = "") -> bool:
        attempts = 3 if reservation_id else 1
        for attempt in range(attempts):
            try:
                return self.store.release_failed(
                    self.visitor_hash,
                    action,
                    reservation_id=reservation_id,
                )
            except (OSError, sqlite3.Error, psycopg.Error, QuotaStoreUnavailable) as exc:
                logger.warning(
                    "Demo quota release failed for support code %s (%s, attempt %d/%d)",
                    self.visitor_hash[:12],
                    type(exc).__name__,
                    attempt + 1,
                    attempts,
                )
                if attempt + 1 < attempts:
                    time.sleep(0.2 * (attempt + 1))
        return False

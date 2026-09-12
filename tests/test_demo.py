from __future__ import annotations

from dataclasses import dataclass
import sqlite3

import psycopg

import layerread.demo as demo_module
from layerread.demo import DemoLimiter, DemoPolicy, limit_images


def _policy(tmp_path, **overrides) -> DemoPolicy:
    values = {
        "LAYERREAD_DEMO_QUOTA_DB_PATH": str(tmp_path / "usage.sqlite3"),
        "LAYERREAD_DEMO_DAILY_ANALYSIS_LIMIT": "1",
        "LAYERREAD_DEMO_DAILY_CHAT_LIMIT": "2",
        "LAYERREAD_DEMO_GLOBAL_DAILY_CALLS": "20",
        "LAYERREAD_DEMO_GLOBAL_DAILY_COST_UNITS": "20",
        "LAYERREAD_DEMO_MAX_IMAGES": "2",
        "LAYERREAD_DEMO_MAX_TOTAL_IMAGE_BYTES": "6",
    }
    values.update(overrides)
    return DemoPolicy.from_env(values)


def test_demo_policy_defaults_to_two_analyses_and_ten_chat_turns() -> None:
    policy = DemoPolicy.from_env({})

    assert policy.action_limits["analysis"] == 2
    assert policy.action_limits["chat"] == 10


def test_demo_quota_is_atomic_for_reserved_actions(tmp_path) -> None:
    limiter = DemoLimiter(_policy(tmp_path), "11111111-1111-4111-8111-111111111111")

    assert limiter.claim("chat").allowed
    second = limiter.claim("chat")
    denied = limiter.claim("chat")

    assert second.allowed
    assert second.remaining == 0
    assert not denied.allowed
    assert denied.remaining == 0


def test_failed_analysis_restores_success_slot_but_keeps_attempt_guard(tmp_path) -> None:
    policy = _policy(
        tmp_path,
        LAYERREAD_DEMO_DAILY_ANALYSIS_LIMIT="2",
        LAYERREAD_DEMO_GLOBAL_DAILY_CALLS="100",
        LAYERREAD_DEMO_GLOBAL_DAILY_COST_UNITS="100",
    )
    limiter = DemoLimiter(policy, "12121212-1212-4121-8121-121212121212")

    for _ in range(4):
        assert limiter.claim("analysis").allowed
        assert limiter.release_failed("analysis")

    denied = limiter.status("analysis")
    assert not denied.allowed
    assert denied.remaining == 2
    assert "安全上限" in denied.message

    with sqlite3.connect(policy.quota_db_path) as connection:
        visitor_row = connection.execute(
            """
            SELECT action_count, cost_units
            FROM demo_visitor_usage
            WHERE action = 'analysis'
            """
        ).fetchone()
        global_row = connection.execute(
            "SELECT call_count, cost_units FROM demo_global_usage"
        ).fetchone()

    assert visitor_row == (0, 20)
    assert global_row == (4, 20)
    assert not limiter.release_failed("analysis")


def test_analysis_reservation_release_is_idempotent(tmp_path) -> None:
    policy = _policy(
        tmp_path,
        LAYERREAD_DEMO_DAILY_ANALYSIS_LIMIT="2",
        LAYERREAD_DEMO_GLOBAL_DAILY_CALLS="100",
        LAYERREAD_DEMO_GLOBAL_DAILY_COST_UNITS="100",
    )
    limiter = DemoLimiter(policy, "13131313-1313-4131-8131-131313131313")

    claim = limiter.claim("analysis")

    assert claim.allowed
    assert claim.reservation_id
    assert limiter.release_failed("analysis", claim.reservation_id)
    assert limiter.release_failed("analysis", claim.reservation_id)
    assert limiter.status("analysis").remaining == 2

    with sqlite3.connect(policy.quota_db_path) as connection:
        visitor_row = connection.execute(
            """
            SELECT action_count, cost_units
            FROM demo_visitor_usage
            WHERE action = 'analysis'
            """
        ).fetchone()

    assert visitor_row == (0, 5)


def test_analysis_reservation_release_retries_transient_store_error(
    monkeypatch,
    tmp_path,
) -> None:
    policy = _policy(
        tmp_path,
        LAYERREAD_DEMO_DAILY_ANALYSIS_LIMIT="2",
        LAYERREAD_DEMO_GLOBAL_DAILY_CALLS="100",
        LAYERREAD_DEMO_GLOBAL_DAILY_COST_UNITS="100",
    )
    limiter = DemoLimiter(policy, "14141414-1414-4141-8141-141414141414")
    claim = limiter.claim("analysis")
    real_release = limiter.store.release_failed
    attempts = 0

    def transient_release(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("temporary failure")
        return real_release(*args, **kwargs)

    monkeypatch.setattr(limiter.store, "release_failed", transient_release)
    monkeypatch.setattr(demo_module.time, "sleep", lambda _: None)

    assert limiter.release_failed("analysis", claim.reservation_id)
    assert attempts == 2
    assert limiter.status("analysis").remaining == 2


def test_demo_quota_stores_only_hashed_identity_and_counters(tmp_path) -> None:
    raw_id = "22222222-2222-4222-8222-222222222222"
    policy = _policy(tmp_path)
    limiter = DemoLimiter(policy, raw_id)
    assert limiter.claim("analysis").allowed

    with sqlite3.connect(policy.quota_db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(demo_visitor_usage)")
        }
        stored_hash = connection.execute(
            "SELECT visitor_hash FROM demo_visitor_usage"
        ).fetchone()[0]

    assert columns == {
        "usage_day",
        "visitor_hash",
        "action",
        "action_count",
        "cost_units",
    }
    assert raw_id not in stored_hash
    assert len(stored_hash) == 64


def test_demo_global_circuit_breaker_applies_across_visitors(tmp_path) -> None:
    policy = _policy(
        tmp_path,
        LAYERREAD_DEMO_GLOBAL_DAILY_CALLS="1",
        LAYERREAD_DEMO_GLOBAL_DAILY_COST_UNITS="100",
    )
    first = DemoLimiter(policy, "33333333-3333-4333-8333-333333333333")
    second = DemoLimiter(policy, "44444444-4444-4444-8444-444444444444")

    assert first.claim("chat").allowed
    denied = second.claim("analysis")

    assert not denied.allowed
    assert "预算" in denied.message


def test_demo_policy_selects_durable_postgres_quota_store(tmp_path) -> None:
    policy = _policy(
        tmp_path,
        LAYERREAD_DEMO_QUOTA_DATABASE_URL=(
            "postgresql://layerread:secret@example.test/layerread?sslmode=require"
        ),
        LAYERREAD_DEMO_REQUIRE_PERSISTENT_QUOTA="true",
    )

    assert policy.quota_database_url is not None
    assert policy.quota_database_url.startswith("postgresql://")
    assert policy.require_persistent_quota


def test_required_persistent_quota_fails_closed_without_database_url(tmp_path) -> None:
    policy = _policy(
        tmp_path,
        LAYERREAD_DEMO_REQUIRE_PERSISTENT_QUOTA="true",
    )
    limiter = DemoLimiter(policy, "55555555-5555-4555-8555-555555555555")

    status = limiter.status("analysis")
    claim = limiter.claim("analysis")
    released = limiter.release_failed("analysis")

    assert not status.allowed
    assert not claim.allowed
    assert not released
    assert "限制服务暂不可用" in status.message
    assert "限制服务暂不可用" in claim.message
    assert not policy.quota_db_path.exists()


def test_unreachable_postgres_quota_store_fails_closed(monkeypatch, tmp_path) -> None:
    policy = _policy(
        tmp_path,
        LAYERREAD_DEMO_QUOTA_DATABASE_URL=(
            "postgresql://layerread:do-not-leak@example.test/layerread"
        ),
        LAYERREAD_DEMO_REQUIRE_PERSISTENT_QUOTA="true",
    )

    def fail_to_connect(*args, **kwargs):
        raise psycopg.OperationalError("database unavailable")

    monkeypatch.setattr(demo_module.psycopg, "connect", fail_to_connect)
    decision = DemoLimiter(
        policy,
        "66666666-6666-4666-8666-666666666666",
    ).claim("analysis")

    assert not decision.allowed
    assert "限制服务暂不可用" in decision.message
    assert "do-not-leak" not in decision.message


@dataclass(frozen=True)
class _Image:
    data_url: str


def test_demo_image_limits_apply_count_and_decoded_byte_budget(tmp_path) -> None:
    policy = _policy(tmp_path)
    images = (
        _Image("data:image/png;base64,YWJj"),
        _Image("data:image/png;base64,ZGVm"),
        _Image("data:image/png;base64,Z2hp"),
    )

    result = limit_images(images, policy)

    assert result.images == images[:2]
    assert result.kept_bytes == 6
    assert result.dropped_count == 1

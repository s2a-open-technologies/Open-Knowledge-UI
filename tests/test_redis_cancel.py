"""Chat-Stopp ueber Redis (app/core/redis_bus.py): Kanal, Backstop-Schluessel, Betrieb ohne Redis."""

import asyncio
import json

import fakeredis
import pytest

from app.chat.streams import StreamRegistry
from app.core import redis_bus


@pytest.fixture
def bus(monkeypatch):
    registry = StreamRegistry()
    monkeypatch.setattr(redis_bus, "registry", registry)
    yield registry
    redis_bus.set_client(None)


async def _sleeping_stream(registry, stream_id, user_id):
    task = asyncio.create_task(asyncio.sleep(30))
    assert registry.add(stream_id, task, user_id)
    return task


def test_a_stop_from_another_worker_reaches_the_stream(bus):
    async def scenario():
        redis_bus.set_client(fakeredis.FakeAsyncRedis(decode_responses=True))
        listener = asyncio.create_task(redis_bus._listen())
        await asyncio.sleep(0.05)  # Abo steht
        task = await _sleeping_stream(bus, "s1", 7)
        # "Worker B": kennt den Stream nicht und meldet den Stopp ueber Redis
        assert await redis_bus.publish_cancel("s1", 7) is True
        for _ in range(50):
            if task.cancelled():
                break
            await asyncio.sleep(0.02)
        listener.cancel()
        return task.cancelled()

    assert asyncio.run(scenario())


def test_a_foreign_user_cannot_stop_over_the_bus(bus):
    async def scenario():
        task = await _sleeping_stream(bus, "s2", 7)
        assert redis_bus.handle_message(json.dumps({"stream_id": "s2", "user_id": 8})) is False
        await asyncio.sleep(0)
        cancelled = task.cancelled()
        task.cancel()
        return cancelled

    assert asyncio.run(scenario()) is False


def test_the_backstop_key_catches_a_stop_before_the_stream_started(bus):
    async def scenario():
        redis_bus.set_client(fakeredis.FakeAsyncRedis(decode_responses=True))
        await redis_bus.publish_cancel("s3", 7)
        return (
            await redis_bus.cancel_requested("s3", 7),
            await redis_bus.cancel_requested("s3", 8),
            await redis_bus.cancel_requested("anders", 7),
        )

    assert asyncio.run(scenario()) == (True, False, False)


def test_without_redis_the_local_stop_still_works(bus, monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    async def scenario():
        await redis_bus.start()  # ohne REDIS_URL: nur Hinweis im Log
        assert redis_bus.status() == "off"
        assert await redis_bus.publish_cancel("s4", 7) is False
        task = await _sleeping_stream(bus, "s4", 7)
        assert bus.cancel("s4", 7)
        try:
            await task
        except asyncio.CancelledError:
            return True
        return False

    assert asyncio.run(scenario())


def test_bad_messages_on_the_channel_are_ignored(bus):
    for data in ("kein json", json.dumps({"stream_id": "x"}), json.dumps([1, 2])):
        assert redis_bus.handle_message(data) is False


def test_health_reports_the_redis_state(db):
    from tests.web import anon, make_app

    assert anon(make_app()).get("/health").json()["redis"] == "off"

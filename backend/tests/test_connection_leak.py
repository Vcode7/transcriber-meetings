import gc
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI, Depends
from httpx import AsyncClient, ASGITransport
from database import get_db, get_db_context


@pytest.mark.asyncio
async def test_fastapi_dependency_connection_close():
    """Verify FastAPI Depends(get_db) automatically closes the AsyncSession upon request completion."""
    mock_session = AsyncMock()
    mock_session.close = AsyncMock()

    with patch("database.AsyncSessionLocal", return_value=mock_session):
        app = FastAPI()

        @app.get("/test-db")
        async def sample_route(db=Depends(get_db)):
            assert db is mock_session
            return {"status": "ok"}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/test-db")
            assert res.status_code == 200
            assert res.json() == {"status": "ok"}

        # Verify close() was called when the request finished
        mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_fastapi_dependency_connection_close_on_exception():
    """Verify FastAPI Depends(get_db) closes the AsyncSession even if an unhandled exception occurs."""
    mock_session = AsyncMock()
    mock_session.close = AsyncMock()
    mock_session.rollback = AsyncMock()

    with patch("database.AsyncSessionLocal", return_value=mock_session):
        app = FastAPI()

        @app.get("/test-error")
        async def error_route(db=Depends(get_db)):
            raise RuntimeError("Route processing error")

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/test-error")
            assert res.status_code == 500

        # Verify rollback and close were called
        mock_session.rollback.assert_awaited_once()
        mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_db_context_manager():
    """Verify get_db_context() works properly as an async context manager."""
    mock_session = AsyncMock()
    mock_session.close = AsyncMock()

    with patch("database.AsyncSessionLocal", return_value=mock_session):
        async with get_db_context() as db:
            assert db is mock_session

        mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_garbage_collection_connection_warning(caplog):
    """Verify garbage collection runs without emitting non-checked-in connection warnings."""
    mock_session = AsyncMock()
    mock_session.close = AsyncMock()

    with patch("database.AsyncSessionLocal", return_value=mock_session):
        async with get_db_context() as db:
            pass

    gc.collect()
    warnings = [rec.message for rec in caplog.records if "non-checked-in connection" in rec.message]
    assert len(warnings) == 0

"""Unit tests for authentication token dependencies."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.api.deps import get_current_user, get_user_by_token
from app.core.security import create_access_token
from app.db.models import User


@pytest.mark.asyncio
async def test_get_user_by_token_success():
    """Verify get_user_by_token retrieves user using JWT sub (user ID)."""
    user_id = uuid.uuid4()
    token = create_access_token(data={"sub": str(user_id)})

    fake_user = User(
        id=user_id,
        email="test@example.com",
        full_name="Test User",
        hashed_password="hashed_pw",
        is_active=True,
    )

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = fake_user
    mock_db.execute.return_value = mock_result

    user = await get_user_by_token(token, mock_db)
    assert user == fake_user
    assert str(user.id) == str(user_id)


@pytest.mark.asyncio
async def test_get_user_by_token_missing_sub():
    """Verify get_user_by_token raises 401 when sub is missing in token."""
    token = create_access_token(data={"email": "test@example.com"})

    mock_db = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await get_user_by_token(token, mock_db)

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_user_by_token_user_not_found():
    """Verify get_user_by_token raises 401 when user id does not exist in DB."""
    user_id = uuid.uuid4()
    token = create_access_token(data={"sub": str(user_id)})

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    with pytest.raises(HTTPException) as exc_info:
        await get_user_by_token(token, mock_db)

    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_no_token():
    """Verify get_current_user raises 401 when no token is provided."""
    mock_db = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=None, db=mock_db)

    assert exc_info.value.status_code == 401
    assert "Not authenticated" in exc_info.value.detail

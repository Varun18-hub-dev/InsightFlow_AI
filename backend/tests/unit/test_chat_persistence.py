"""Unit tests for Chat persistence, UUID handling, and conversation history restoration."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.api.chat import (
    get_conversation,
    list_conversations,
)
from app.db.models import Conversation, Message, User


@pytest.mark.asyncio
async def test_uuid_parsing_in_message_creation():
    """Verify UUID and string conversions for conversation_id work correctly."""
    user_id = uuid.uuid4()
    conv_id = uuid.uuid4()
    msg_id = uuid.uuid4()

    # Create message with UUIDs
    msg = Message(
        id=msg_id,
        conversation_id=conv_id,
        user_id=user_id,
        role="assistant",
        content="Test content",
        sources=[{"document": "test.pdf", "page": 1}],
        created_at=datetime.utcnow(),
    )

    assert isinstance(msg.id, uuid.UUID)
    assert isinstance(msg.conversation_id, uuid.UUID)
    assert str(msg.id) == str(msg_id)
    assert str(msg.conversation_id) == str(conv_id)


@pytest.mark.asyncio
async def test_list_conversations():
    """Verify list_conversations queries only the requesting user's conversations."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email="user@example.com")

    conv1 = Conversation(id=uuid.uuid4(), user_id=user_id, title="Chat 1", created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    conv2 = Conversation(id=uuid.uuid4(), user_id=user_id, title="Chat 2", created_at=datetime.utcnow(), updated_at=datetime.utcnow())

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [conv2, conv1]
    mock_db.execute.return_value = mock_result

    results = await list_conversations(db=mock_db, current_user=user)
    assert len(results) == 2
    assert results[0].title == "Chat 2"
    assert results[1].title == "Chat 1"


@pytest.mark.asyncio
async def test_get_conversation_success():
    """Verify get_conversation returns ordered messages when authorized."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email="user@example.com")
    conv_id = uuid.uuid4()

    conv = Conversation(id=conv_id, user_id=user_id, title="Test Chat", created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    msg1 = Message(id=uuid.uuid4(), conversation_id=conv_id, user_id=user_id, role="user", content="Hello", created_at=datetime.utcnow(), msg_metadata={})
    msg2 = Message(id=uuid.uuid4(), conversation_id=conv_id, user_id=user_id, role="assistant", content="Hi there!", created_at=datetime.utcnow(), msg_metadata={})

    mock_db = AsyncMock()

    # First query checks conversation existence & ownership
    conv_result = MagicMock()
    conv_result.scalar_one_or_none.return_value = conv

    # Second query retrieves messages
    msg_result = MagicMock()
    msg_result.scalars.return_value.all.return_value = [msg1, msg2]

    mock_db.execute.side_effect = [conv_result, msg_result]

    detail = await get_conversation(conversation_id=str(conv_id), db=mock_db, current_user=user)
    assert len(detail.messages) == 2
    assert detail.messages[0].content == "Hello"
    assert detail.messages[1].content == "Hi there!"


@pytest.mark.asyncio
async def test_get_conversation_forbidden():
    """Verify get_conversation raises 404 when conversation belongs to another user."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email="user@example.com")
    conv_id = uuid.uuid4()

    mock_db = AsyncMock()
    conv_result = MagicMock()
    conv_result.scalar_one_or_none.return_value = None  # Not found for this user
    mock_db.execute.return_value = conv_result

    with pytest.raises(HTTPException) as exc_info:
        await get_conversation(conversation_id=str(conv_id), db=mock_db, current_user=user)

    assert exc_info.value.status_code == 404

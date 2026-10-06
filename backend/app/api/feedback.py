import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.base import get_db
from app.db.models import Message, User, UserFeedback
from app.db.schemas import FeedbackCreate, FeedbackResponse

router = APIRouter()
logger = structlog.get_logger()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Verify message belongs to user
    stmt = select(Message).where(
        Message.id == feedback.message_id,
        Message.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Message not found")

    fb = UserFeedback(
        message_id=feedback.message_id,
        user_id=current_user.id,
        rating=feedback.rating,
        feedback_text=feedback.feedback_text,
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)
    logger.info("feedback_submitted", message_id=str(feedback.message_id), rating=feedback.rating)
    return fb


@router.get("/feedback")
async def list_feedback(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = select(UserFeedback).where(UserFeedback.user_id == current_user.id).order_by(UserFeedback.created_at.desc())
    result = await db.execute(stmt)
    feedbacks = result.scalars().all()
    return [{"id": str(f.id), "message_id": str(f.message_id), "rating": f.rating, "feedback_text": f.feedback_text, "created_at": str(f.created_at)} for f in feedbacks]

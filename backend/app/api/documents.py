import os
import uuid
from pathlib import Path

import aiofiles
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.base import get_db
from app.db.models import Document
from app.db.schemas import DocumentResponse
from app.ingestion.document_processor import process_document
from app.ingestion.file_validator import FileValidator
from app.services.pinecone_service import get_vector_store

router = APIRouter()
logger = structlog.get_logger()
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    validator = FileValidator()
    validator.validate_file(file)
    safe_name = validator.sanitize_filename(file.filename)
    doc_id = uuid.uuid4()

    file_path = UPLOAD_DIR / str(current_user.id) / f"{doc_id}_{safe_name}"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    doc = Document(
        id=doc_id,
        user_id=current_user.id,
        filename=safe_name,
        original_filename=file.filename,
        document_type=file.content_type,
        file_path=str(file_path),
        file_size=len(content),
        status="pending"
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    background_tasks.add_task(process_document, doc_id, str(file_path), file.content_type)

    logger.info("document_uploaded", doc_id=str(doc_id), filename=safe_name, user_id=str(current_user.id))
    return doc

@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    stmt = select(Document).where(Document.user_id == current_user.id).order_by(Document.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()

@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    stmt = select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    stmt = select(Document).where(Document.id == document_id, Document.user_id == current_user.id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    vector_store = get_vector_store()
    await vector_store.delete_document(str(doc.id))

    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    await db.delete(doc)
    await db.commit()

    return {"message": "Document deleted successfully"}

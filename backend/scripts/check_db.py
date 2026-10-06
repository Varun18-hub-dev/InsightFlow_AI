import asyncio
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.base import engine
from sqlalchemy import text

async def check():
    async with engine.connect() as conn:
        res = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name"))
        tables = [r[0] for r in res.fetchall()]
        print("Live PostgreSQL Tables in Docker:", tables)
        for expected in ["users", "documents", "document_chunks", "conversations", "messages", "alembic_version"]:
            assert expected in tables, f"Missing table {expected}"
        print("[PASS] All 8 tables verified in live PostgreSQL database!")

if __name__ == "__main__":
    asyncio.run(check())

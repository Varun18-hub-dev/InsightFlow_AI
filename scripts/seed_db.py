#!/usr/bin/env python
"""Database seeding script — creates initial schema and optional test data"""
import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://insightflow:insightflow_pass@localhost:5432/insightflow")

async def seed_db():
    print(f"Connecting to {DATABASE_URL}...")
    engine = create_async_engine(DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        # Create users table (example schema for auth)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Check if admin exists
        result = await conn.execute(text("SELECT id FROM users WHERE email = 'admin@insightflow.ai'"))
        admin = result.fetchone()
        
        if not admin:
            print("Creating default admin user...")
            # Note: In reality, use passlib to hash password. Hardcoded dummy hash here for seeding.
            dummy_hash = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjIQqiRQYq" # Admin123!
            await conn.execute(
                text("INSERT INTO users (email, hashed_password, is_admin) VALUES (:email, :pwd, :is_admin)"),
                {"email": "admin@insightflow.ai", "pwd": dummy_hash, "is_admin": True}
            )
            print("Admin user created (admin@insightflow.ai / Admin123!)")
        else:
            print("Admin user already exists.")
            
    await engine.dispose()
    print("Database seeding completed.")

if __name__ == "__main__":
    asyncio.run(seed_db())

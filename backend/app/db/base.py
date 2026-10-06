import urllib.parse
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.core.config import settings

# Unsupported libpq/psycopg parameters that asyncpg does not accept as connect() keyword arguments
UNSUPPORTED_ASYNCPG_QUERY_PARAMS = {
    "channel_binding",
    "sslmode",
    "sslcompression",
    "gssencmode",
    "endpoint",
}


def _get_async_database_url_and_args(raw_url: str) -> tuple[str, dict]:
    """Normalize PostgreSQL URLs and configure asyncpg connection arguments.

    Converts `postgres://` or `postgresql://` to `postgresql+asyncpg://`.
    Translates unsupported libpq URL query parameters like `sslmode` into
    supported asyncpg `connect_args['ssl']`, while removing libpq-specific
    parameters like `channel_binding` before SQLAlchemy/asyncpg receives them,
    preserving full SSL/TLS encryption for cloud databases like Neon.
    """
    if raw_url.startswith("postgres://"):
        raw_url = "postgresql+asyncpg://" + raw_url[len("postgres://") :]
    elif raw_url.startswith("postgresql://") and not raw_url.startswith("postgresql+asyncpg://"):
        raw_url = "postgresql+asyncpg://" + raw_url[len("postgresql://") :]

    parsed = urllib.parse.urlsplit(raw_url)
    query_params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    connect_args: dict = {}

    sslmode = query_params.pop("sslmode", None)
    ssl_param = query_params.pop("ssl", None)

    if sslmode:
        mode = sslmode[0]
        if mode.lower() == "disable":
            connect_args["ssl"] = False
        else:
            connect_args["ssl"] = mode
    elif ssl_param:
        val = ssl_param[0]
        if val.lower() in ("false", "0", "disable"):
            connect_args["ssl"] = False
        elif val.lower() in ("true", "1"):
            connect_args["ssl"] = True
        else:
            connect_args["ssl"] = val
    elif "neon.tech" in parsed.netloc:
        connect_args["ssl"] = "require"

    # Remove any unsupported libpq-specific query parameters
    for param in UNSUPPORTED_ASYNCPG_QUERY_PARAMS:
        query_params.pop(param, None)

    clean_query = urllib.parse.urlencode([(k, v[0]) for k, v in query_params.items()], doseq=True)
    clean_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, clean_query, parsed.fragment))

    return clean_url, connect_args


def _get_async_database_url(url: str) -> str:
    clean_url, _ = _get_async_database_url_and_args(url)
    return clean_url


db_url, connect_args = _get_async_database_url_and_args(settings.DATABASE_URL)
engine = create_async_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=settings.DEBUG,
    future=True,
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

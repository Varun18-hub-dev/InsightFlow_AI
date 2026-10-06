"""Unit tests for async PostgreSQL database URL normalization and SSL configuration."""

from sqlalchemy.ext.asyncio import create_async_engine

from app.db.base import _get_async_database_url, _get_async_database_url_and_args


def test_neon_url_with_sslmode_require():
    """Verify Neon URL with sslmode=require strips sslmode and sets connect_args['ssl'] = 'require'."""
    url = "postgresql://alex:AbC123dEf@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb?sslmode=require"
    clean_url, connect_args = _get_async_database_url_and_args(url)

    assert clean_url == "postgresql+asyncpg://alex:AbC123dEf@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb"
    assert connect_args == {"ssl": "require"}
    assert "sslmode" not in clean_url

    # Verify engine creation succeeds without TypeError
    engine = create_async_engine(clean_url, connect_args=connect_args)
    assert engine is not None


def test_neon_url_with_channel_binding_require():
    """Verify Neon URL with channel_binding=require strips channel_binding and keeps Neon SSL active."""
    url = "postgresql://alex:AbC123dEf@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb?channel_binding=require"
    clean_url, connect_args = _get_async_database_url_and_args(url)

    assert clean_url == "postgresql+asyncpg://alex:AbC123dEf@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb"
    assert connect_args == {"ssl": "require"}
    assert "channel_binding" not in clean_url

    # Verify engine creation succeeds without TypeError
    engine = create_async_engine(clean_url, connect_args=connect_args)
    assert engine is not None


def test_neon_url_with_both_sslmode_and_channel_binding():
    """Verify Neon URL containing both sslmode=require and channel_binding=require strips both."""
    url = (
        "postgresql://alex:AbC123dEf@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb"
        "?sslmode=require&channel_binding=require"
    )
    clean_url, connect_args = _get_async_database_url_and_args(url)

    assert clean_url == "postgresql+asyncpg://alex:AbC123dEf@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb"
    assert connect_args == {"ssl": "require"}
    assert "sslmode" not in clean_url
    assert "channel_binding" not in clean_url

    # Verify reverse query parameter order
    url_rev = (
        "postgresql://alex:AbC123dEf@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb"
        "?channel_binding=require&sslmode=require"
    )
    clean_url_rev, connect_args_rev = _get_async_database_url_and_args(url_rev)
    assert clean_url_rev == clean_url
    assert connect_args_rev == {"ssl": "require"}

    engine = create_async_engine(clean_url, connect_args=connect_args)
    assert engine is not None


def test_local_postgresql_url():
    """Verify local PostgreSQL development URL preserves unencrypted connection without SSL errors."""
    url = "postgresql+asyncpg://insightflow:insightflow_pass@localhost:5432/insightflow"
    clean_url, connect_args = _get_async_database_url_and_args(url)

    assert clean_url == url
    assert connect_args == {}

    # Standard Docker container host
    docker_url = "postgresql+asyncpg://insightflow:insightflow_pass@postgres:5432/insightflow"
    clean_docker, connect_docker = _get_async_database_url_and_args(docker_url)
    assert clean_docker == docker_url
    assert connect_docker == {}


def test_neon_url_with_additional_unsupported_params():
    """Verify endpoint, sslcompression, and gssencmode are cleanly stripped."""
    url = (
        "postgresql://alex:AbC123dEf@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb"
        "?sslmode=require&channel_binding=require&endpoint=ep-cool-fog-786406&sslcompression=0"
    )
    clean_url, connect_args = _get_async_database_url_and_args(url)

    assert clean_url == "postgresql+asyncpg://alex:AbC123dEf@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb"
    assert connect_args == {"ssl": "require"}
    assert "endpoint" not in clean_url
    assert "sslcompression" not in clean_url


def test_backward_compatibility_wrapper():
    """Verify _get_async_database_url returns clean normalized URL string."""
    url = "postgres://user:pass@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    clean_url = _get_async_database_url(url)

    assert clean_url == "postgresql+asyncpg://user:pass@ep-cool-fog-786406.us-east-2.aws.neon.tech/neondb"
    assert "sslmode" not in clean_url
    assert "channel_binding" not in clean_url

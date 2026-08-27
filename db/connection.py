"""Database Connection & Utility Module.

Provides connection string creation, SHA-256 checksum calculation for in-memory
API payloads, and database engine management.
"""

import os
import hashlib
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

logger = logging.getLogger("DatabaseConnection")

# Load environment variables from .env file
env_path = next(
    (path / ".env" for path in [Path.cwd(), *Path.cwd().parents] if (path / ".env").exists()),
    None
)
if env_path:
    load_dotenv(env_path)


import socket

_engine = None


def get_db_url() -> str:
    """Generates PostgreSQL SQLAlchemy connection URL from environment variables.
    
    Supports both POSTGRES_* and DB_* environment variable naming conventions.
    Automatically falls back from container hostname ('db') to 'localhost'
    if hostname resolution fails outside Docker.
    """
    host = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB") or os.getenv("DB_NAME", "enerji_db")
    user = os.getenv("POSTGRES_USER") or os.getenv("DB_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASS", "postgres")
    
    if host == "db":
        try:
            socket.gethostbyname("db")
        except socket.gaierror:
            host = "localhost"

    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


def get_db_engine(db_url: str = None):
    """Creates and returns a SQLAlchemy Engine with connection pooling and Europe/Istanbul timezone."""
    global _engine
    if db_url is None:
        if _engine is None:
            url = get_db_url()
            _engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"options": "-c timezone=Europe/Istanbul"},
            )
        return _engine
    else:
        return create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            connect_args={"options": "-c timezone=Europe/Istanbul"},
        )


def calculate_checksum(data) -> str:
    """Calculates a SHA-256 digital signature hash for an in-memory API payload (list or dict)."""
    if isinstance(data, (dict, list)):
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    elif isinstance(data, str):
        serialized = data.encode("utf-8")
    else:
        serialized = bytes(data)
    return hashlib.sha256(serialized).hexdigest()

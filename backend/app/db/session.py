"""Database engine, session factory and FastAPI dependency."""
import logging
from collections.abc import Generator

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def make_engine(url: str, **kwargs) -> Engine:
    is_sqlite = url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    if not is_sqlite:
        # PostgreSQL: drop connections the server closed (restart, idle timeout) before use.
        kwargs.setdefault("pool_pre_ping", True)
    new_engine = create_engine(url, connect_args=connect_args, **kwargs)
    if is_sqlite:
        # SQLite ignores foreign keys (and ON DELETE CASCADE) unless enabled per connection.
        @event.listens_for(new_engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return new_engine


engine = make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _add_missing_columns(target: Engine) -> None:
    """Non-destructive upgrade: add new nullable columns to tables created by older code.

    create_all() creates missing tables but never alters existing ones, so a database
    made in an earlier phase would otherwise fail with "no such column".
    """
    inspector = inspect(target)
    existing_tables = set(inspector.get_table_names())
    with target.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                if not column.nullable:
                    logger.warning(
                        "Cannot auto-add NOT NULL column %s.%s; a migration is needed",
                        table.name,
                        column.name,
                    )
                    continue
                col_type = column.type.compile(dialect=target.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'))
                logger.info("Added column %s.%s", table.name, column.name)


def init_db(target: Engine | None = None) -> None:
    """Create missing tables and add missing columns (no data is ever dropped)."""
    import app.models  # noqa: F401  (registers every model on Base)

    target = target or engine
    Base.metadata.create_all(bind=target)
    _add_missing_columns(target)

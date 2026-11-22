"""Copy IntelliDoc data from the SQLite file (Phases 2-5) into PostgreSQL (Phase 6).

    cd backend
    ..\\.venv\\Scripts\\python.exe scripts\\migrate_sqlite_to_postgres.py ^
        --target "postgresql+psycopg://intellidoc:PASSWORD@127.0.0.1:5432/intellidoc"

- Reads only: the SQLite file is never changed (keep it as a backup).
- Row ids are kept, so the ChromaDB vectors (stored as "doc{id}-chunk{n}") still match.
- Refuses to run if the target already holds data, so nothing is overwritten or duplicated.
- Older SQLite files lack newer columns (e.g. owner_id); those are left empty (NULL).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make "app" importable

from sqlalchemy import func, inspect, select, text  # noqa: E402

import app.models  # noqa: E402,F401  (registers every table)
from app.core.config import BASE_DIR  # noqa: E402
from app.db.session import Base, init_db, make_engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=str(BASE_DIR / "intellidoc.db"), help="SQLite file (default: backend/intellidoc.db)")
    parser.add_argument("--target", required=True, help="PostgreSQL SQLAlchemy URL")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    if not source_path.is_file():
        print(f"SQLite file not found: {source_path}")
        return 1
    if not args.target.startswith("postgresql"):
        print("--target must be a postgresql:// URL")
        return 1

    # mode=ro: open the SQLite file read-only, so this script cannot modify it.
    source = make_engine(f"sqlite:///file:{source_path.as_posix()}?mode=ro&uri=true")
    target = make_engine(args.target)
    init_db(target)  # create any missing tables in PostgreSQL

    source_tables = set(inspect(source).get_table_names())
    with target.connect() as conn:
        busy = [t.name for t in Base.metadata.sorted_tables if conn.scalar(select(func.count()).select_from(t))]
    if busy:
        print(f"Target already has data in: {', '.join(busy)}. Nothing copied (no overwrite).")
        return 1

    copied: dict[str, int] = {}
    with source.connect() as src, target.begin() as dst:
        for table in Base.metadata.sorted_tables:  # parents before children (foreign keys)
            if table.name not in source_tables:
                continue
            present = {c["name"] for c in inspect(source).get_columns(table.name)}
            columns = [c for c in table.columns if c.name in present]
            # Selecting through the app's own table definitions converts JSON / dates properly.
            rows = [dict(r._mapping) for r in src.execute(select(*columns).order_by(table.c.id))]
            if rows:
                dst.execute(table.insert(), rows)
            copied[table.name] = len(rows)
            # Move the id counter past the copied ids so new rows do not collide.
            dst.execute(
                text(
                    f"SELECT setval(pg_get_serial_sequence('{table.name}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM \"{table.name}\"), 1), "
                    f"(SELECT MAX(id) FROM \"{table.name}\") IS NOT NULL)"
                )
            )

    for name, count in copied.items():
        print(f"  {name}: {count} rows")
    print(f"Done. {source_path.name} was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

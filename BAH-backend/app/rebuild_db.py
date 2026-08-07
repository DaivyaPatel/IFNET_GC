"""
Rebuilds all database tables in Supabase based on app/database/models.py.

WARNING: This drops all existing tables (and their data) before recreating them.
Only run this when you're okay losing whatever is currently in the database.

Usage:
    python rebuild_db.py
"""

from app.database.database import Base, engine
from app.database import models  # noqa: F401  (import needed so Base knows about all models)


def rebuild():
    confirm = input(
        "This will DROP ALL TABLES and recreate them from models.py.\n"
        "Any existing data will be lost. Type 'yes' to continue: "
    )

    if confirm.strip().lower() != "yes":
        print("Aborted. No changes made.")
        return

    print("Dropping existing tables...")
    Base.metadata.drop_all(bind=engine)

    print("Creating tables from models.py...")
    Base.metadata.create_all(bind=engine)

    print("Done. Tables rebuilt successfully.")


if __name__ == "__main__":
    rebuild()


# python -m app.rebuild_db
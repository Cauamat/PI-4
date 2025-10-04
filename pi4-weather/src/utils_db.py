# src/utils_db.py
from sqlalchemy import create_engine

def get_engine(db_path="data/bronze.db"):
    return create_engine(f"sqlite:///{db_path}", future=True)

"""Config loader para o projeto."""
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    owm_api_key: str = os.getenv("OWM_API_KEY", "")
    db_path: Path = Path(os.getenv("DB_PATH", "data/bronze.db"))
    silver_dir: Path = Path(os.getenv("SILVER_DIR", "data/silver"))
    raw_dir: Path = Path(os.getenv("RAW_DIR", "data/raw"))
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/rain_classifier.pkl"))
    city_name: str = os.getenv("CITY_NAME", "Sao Paulo,BR")
    units: str = os.getenv("UNITS", "metric")

SETTINGS = Settings()

# src/ingest.py
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
from .owm_client import safe_call, get_current, get_forecast_5d3h
from .utils_db import get_engine

load_dotenv()

# Lista de cidades a partir do .env
CITIES = os.getenv("CITIES", "São Paulo:-23.55:-46.63").split(";")

# Esquema fixo para evitar problemas de concatenação e dtype
SCHEMA = [
    "ingested_at", "city", "lat", "lon", "source", "dt",
    "temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
    "rain_1h", "rain_3h", "weather_main", "weather_desc"
]

def _city_list():
    out = []
    for item in CITIES:
        name, lat, lon = item.split(":")
        out.append({"city": name, "lat": float(lat), "lon": float(lon)})
    return out

def flatten_current(js, city):
    # Pode haver casos raros de resposta sem campos esperados; use .get defensivo
    ts = datetime.now(timezone.utc).isoformat()
    weather = (js.get("weather") or [{}])[0]
    main = js.get("main") or {}
    wind = js.get("wind") or {}
    clouds = js.get("clouds") or {}
    rain = js.get("rain") or {}

    return {
        "ingested_at": ts,
        "city": city["city"],
        "lat": city["lat"],
        "lon": city["lon"],
        "source": "current",
        "dt": pd.to_datetime(js.get("dt"), unit="s", utc=True),
        "temp": main.get("temp"),
        "humidity": main.get("humidity"),
        "pressure": main.get("pressure"),
        "wind_speed": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "clouds": clouds.get("all"),
        "rain_1h": rain.get("1h", 0.0),
        "rain_3h": None,  # current não possui 3h
        "weather_main": weather.get("main"),
        "weather_desc": weather.get("description"),
    }

def flatten_forecast(js, city):
    rows = []
    # Algumas respostas podem vir sem "list"
    for item in (js.get("list") or []):
        main = item.get("main") or {}
        wind = item.get("wind") or {}
        clouds = item.get("clouds") or {}
        rain = item.get("rain") or {}
        weather = (item.get("weather") or [{}])[0]

        rows.append({
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "city": city["city"],
            "lat": city["lat"],
            "lon": city["lon"],
            "source": "forecast",
            "dt": pd.to_datetime(item.get("dt"), unit="s", utc=True),
            "temp": main.get("temp"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_deg": wind.get("deg"),
            "clouds": clouds.get("all"),
            "rain_1h": rain.get("1h", 0.0),
            "rain_3h": rain.get("3h", 0.0),
            "weather_main": weather.get("main"),
            "weather_desc": weather.get("description"),
        })
    return rows

def run():
    eng = get_engine()
    os.makedirs("data/silver", exist_ok=True)

    total_rows = 0

    for city in _city_list():
        # Sempre inicialize DataFrames vazios já no ESQUEMA
        df_cur = pd.DataFrame(columns=SCHEMA)
        df_fc  = pd.DataFrame(columns=SCHEMA)

        # Tente coletar "current"
        try:
            cur = safe_call(get_current, city["lat"], city["lon"])
            if cur:
                df_cur = pd.DataFrame([flatten_current(cur, city)])
        except Exception as e:
            print(f"[ingest][WARN] current falhou para {city['city']}: {e}")

        # Tente coletar "forecast"
        try:
            fc = safe_call(get_forecast_5d3h, city["lat"], city["lon"])
            if fc:
                fc_rows = flatten_forecast(fc, city)
                if fc_rows:
                    df_fc = pd.DataFrame(fc_rows)
        except Exception as e:
            print(f"[ingest][WARN] forecast falhou para {city['city']}: {e}")

        # Padroniza colunas no SCHEMA e concatena apenas se houver algo
        df_cur = df_cur.reindex(columns=SCHEMA)
        df_fc  = df_fc.reindex(columns=SCHEMA)

        frames = [df for df in (df_cur, df_fc) if not df.empty]
        if not frames:
            print(f"[ingest][INFO] Sem linhas para inserir em {city['city']}. Pulando…")
            continue

        df_all = pd.concat(frames, ignore_index=True)

        # Grava no SQLite
        with eng.begin() as con:
            df_all.to_sql("weather_raw", con, if_exists="append", index=False)

        # Exporta parquet particionado por city/source
        for (c, s), g in df_all.groupby(["city", "source"], dropna=False):
            out_path = f"data/silver/{c}_{s}_{pd.Timestamp.utcnow():%Y%m%dT%H%M%S}.parquet"
            g.to_parquet(out_path, index=False)

        total_rows += len(df_all)

    print(f"[ingest] Inseridas {total_rows} linhas no banco/parquet.")

if __name__ == "__main__":
    run()

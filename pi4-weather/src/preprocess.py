# src/preprocess.py
import pandas as pd, numpy as np, glob

def load_silver():
    files = sorted(glob.glob("data/silver/*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def make_features(df):
    df = df.sort_values(["city","dt"])
    df["hour"] = df["dt"].dt.hour
    df["dow"]  = df["dt"].dt.dayofweek
    # metas: chuva forte ≥10 mm nas próximas 3h (usando forecast rain_3h)
    df["rain_next3h"] = df["rain_3h"].fillna(0.0)
    df["target_heavy_rain"] = (df["rain_next3h"] >= 10.0).astype(int)

    # lags simples
    for col in ["temp","humidity","pressure","wind_speed","clouds"]:
        df[f"{col}_lag1"] = df.groupby("city")[col].shift(1)
        df[f"{col}_roll3_mean"] = df.groupby("city")[col].rolling(3).mean().reset_index(level=0, drop=True)

    df = df.dropna().reset_index(drop=True)
    feats = [c for c in df.columns if any(p in c for p in ["temp","humidity","pressure","wind_speed","clouds","hour","dow","lag","roll3"])]
    return df, feats, "target_heavy_rain"

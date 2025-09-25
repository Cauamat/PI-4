# src/predict.py
import joblib, pandas as pd
from .preprocess import make_features

def load_model(path="models/rain_classifier.pkl"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]

def prepare_latest_sample(df):
    df, feats, _ = make_features(df)
    return df.sort_values("dt").iloc[-1], feats

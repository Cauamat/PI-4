# src/owm_client.py
import os, time, requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OWM_API_KEY")

BASE = "https://api.openweathermap.org/data/2.5"

def get_current(lat, lon):
    url = f"{BASE}/weather?lat={lat}&lon={lon}&units=metric&lang=pt_br&appid={API_KEY}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.json()

def get_forecast_5d3h(lat, lon):
    url = f"{BASE}/forecast?lat={lat}&lon={lon}&units=metric&lang=pt_br&appid={API_KEY}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.json()

def safe_call(func, *args, retries=3, sleep=2):
    for i in range(retries):
        try:
            return func(*args)
        except Exception:
            if i == retries-1: raise
            time.sleep(sleep)
# app/streamlit_app.py
import os
import glob
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from joblib import load as jload
from datetime import timedelta

st.set_page_config(page_title="PI IV - Weather AI", layout="wide")

# =========================
# Helpers de carregamento
# =========================
@st.cache_data
def load_parquet_silver():
    """
    Lê todos os arquivos Parquet da camada 'silver' e retorna um DataFrame único.
    Espera encontrar os arquivos em ../data/silver/ (rodando a partir de app/)
    """
    base = Path("../data/silver")
    files = sorted(base.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Ajuste de timezone (os dados vêm em UTC)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
        df["dt_local"] = df["dt"].dt.tz_convert("America/Sao_Paulo")
    if "ingested_at" in df.columns:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True, errors="coerce")
    return df

@st.cache_data
def load_model_bundle():
    """
    Carrega o modelo treinado, features e threshold salvos em ../models/rain_classifier.pkl
    """
    path = Path("../models/rain_classifier.pkl")
    if not path.exists():
        return None
    bundle = jload(path)
    # Backward compatibility: se threshold não existir, usar 0.5
    bundle["threshold"] = bundle.get("threshold", 0.5)
    return bundle

def ensure_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

# =========================
# App
# =========================
st.title("PI IV — Weather AI (OpenWeatherMap)")

df = load_parquet_silver()
if df.empty:
    st.warning("Nenhum dado encontrado em ../data/silver/. "
               "Execute o coletor (`python -m src.ingest`) e tente novamente.")
    st.stop()

# Filtros laterais
cities = sorted(df["city"].dropna().unique())
city = st.sidebar.selectbox("Cidade", cities, index=0)

# Janela temporal (opcional): últimos N dias para visualização
days = st.sidebar.slider("Janela (dias) para gráficos", min_value=1, max_value=30, value=7)
cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
dfc = df[(df["city"] == city) & (df["dt"] >= cutoff)].copy().sort_values("dt")

if dfc.empty:
    st.info("Sem dados para esta janela. Aumente o intervalo de dias ou aguarde novas coletas.")
    st.stop()

# =========================
# KPIs rápidos (última observação disponível)
# =========================
last = dfc.dropna(subset=["temp", "humidity", "pressure", "rain_3h"], how="all").tail(1)
if last.empty:
    last = dfc.tail(1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Última Temperatura (°C)", f"{last['temp'].iloc[-1]:.1f}" if pd.notna(last['temp'].iloc[-1]) else "—")
c2.metric("Umidade (%)", f"{last['humidity'].iloc[-1]:.0f}" if pd.notna(last['humidity'].iloc[-1]) else "—")
c3.metric("Pressão (hPa)", f"{last['pressure'].iloc[-1]:.0f}" if pd.notna(last['pressure'].iloc[-1]) else "—")
c4.metric("Chuva 3h (mm)", f"{(last['rain_3h'].iloc[-1] or 0):.1f}" if 'rain_3h' in last else "0.0")

# =========================
# Séries temporais
# =========================
st.subheader(f"Séries — {city} (últimos {days} dias)")

cols_raw = [c for c in ["temp", "humidity", "pressure", "rain_3h"] if c in dfc.columns]
if len(cols_raw) >= 1:
    # renomeia para português (legenda bonita)
    rename_map = {
        "temp": "Temperatura (°C)",
        "humidity": "Umidade (%)",
        "pressure": "Pressão (hPa)",
        "rain_3h": "Chuva 3h (mm)",
    }
    df_plot = dfc[["dt_local"] + cols_raw].rename(columns=rename_map)

    # derrete para formato longo: dt_local | Variável | Valor
    df_plot = df_plot.melt(id_vars=["dt_local"], var_name="Variável", value_name="Valor")

    fig = px.line(
        df_plot, x="dt_local", y="Valor", color="Variável",
        labels={
            "dt_local": "Data/Hora (local)",
            "Valor": "Valor",
            "Variável": "Variável"
        },
        title="Variáveis meteorológicas"
    )
    # opcional: melhorar leitura do eixo X
    fig.update_layout(xaxis_title="Data/Hora (local)", yaxis_title="Valor")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem colunas para plotar (temperatura/umidade/pressão/chuva).")

# =========================
# Previsão (probabilidade de chuva forte em 3h)
# =========================
st.subheader("Previsão — Probabilidade de chuva forte (≥ 10 mm nas próximas 3h)")

bundle = load_model_bundle()
if not bundle:
    st.warning("Modelo ainda não treinado. Rode `python -m src.train` e recarregue a página.")
    st.stop()

model = bundle["model"]
feats = bundle["features"]
threshold = bundle["threshold"]

# --- Recriar features essenciais (compatíveis com treino) ---
# Usamos uma janela recente para calcular lags e médias móveis
lookback_h = st.sidebar.slider("Janela de cálculo (horas)", min_value=3, max_value=24, value=6, step=1)
cut_recent = dfc["dt"].max() - pd.Timedelta(hours=lookback_h)
recent = dfc[dfc["dt"] >= cut_recent].copy().sort_values("dt")

# Gera features básicas (igual/esperado no treino):
recent["hour"] = recent["dt"].dt.hour
recent["dow"] = recent["dt"].dt.dayofweek
for col in ["temp", "humidity", "pressure", "wind_speed", "clouds"]:
    if col in recent.columns:
        recent[f"{col}_lag1"] = recent[col].shift(1)
        recent[f"{col}_roll3_mean"] = recent[col].rolling(3).mean()

# Garantir que todas as features esperadas existam
recent = ensure_columns(recent, feats)

# Amostra: última linha com features válidas
sample = recent.dropna(subset=[c for c in feats if ("lag" in c or "roll3" in c)], how="any")
if sample.empty:
    # fallback: tenta última linha preenchendo faltantes com média
    sample = recent.tail(1)

X = sample.tail(1).reindex(columns=feats)
# Preencher faltantes numéricos com média da própria amostra (safe)
if X.isna().any().any():
    X = X.fillna(X.mean(numeric_only=True))

# Probabilidade e decisão
proba = float(model.predict_proba(X)[:, 1][0])
is_heavy = proba >= threshold

colp, colm = st.columns([1, 2])
colp.metric("Prob. de chuva forte (3h)", f"{proba*100:.1f}%")
if is_heavy:
    colm.error(f"ALERTA: acima do limiar salvo ({threshold:.2f})")
else:
    colm.info(f"Abaixo do limiar salvo ({threshold:.2f})")

# Mostrar insumos da amostra usada
with st.expander("Ver amostra (features) usada na previsão"):
    st.write(X)

# =========================
# Eventos recentes de chuva (para contextualizar)
# =========================
st.subheader("Eventos recentes de chuva (últimos 3 dias)")
recent_days = pd.Timestamp.utcnow() - pd.Timedelta(days=3)
view = df[(df["city"] == city) & (df["dt"] >= recent_days)].copy()
view["rain_3h"] = view["rain_3h"].fillna(0)
view = view.sort_values("dt_local", ascending=False)
events = view[view["rain_3h"] >= 10.0][["dt_local", "rain_3h", "temp", "humidity", "pressure", "wind_speed"]]
if events.empty:
    st.info("Sem eventos de chuva forte (≥ 10 mm/3h) nos últimos 3 dias para esta cidade.")
else:
    st.dataframe(events.rename(columns={"dt_local": "Data/Hora (local)", "rain_3h": "Chuva 3h (mm)"}))

# =========================
# Rodapé
# =========================
st.caption(
    "Dados coletados da OpenWeatherMap (current + 5d/3h). "
    "Modelo treinado com Logistic Regression (balanceado) e threshold salvo no bundle."
)

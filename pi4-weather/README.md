# PI4-Weather — IA do Clima

Pipeline completo para coleta de dados meteorológicos da [OpenWeatherMap](https://openweathermap.org/), 
pré-processamento, treino de um classificador de chuvas intensas (≥10 mm/3h) e visualização em um dashboard interativo com Streamlit.

Este projeto faz parte do **Projeto Integrador IV** (Ciência da Computação / Ciência de Dados) e tem como objetivo aplicar conceitos de:
- ingestão automática de dados (ETL),
- análise exploratória,
- machine learning para eventos raros,
- e apresentação dos resultados em uma interface acessível.

---

## Instalação

```bash
# criar ambiente virtual
python -m venv .venv 
. .venv/Scripts/activate   # Windows PowerShell

# instalar dependências
pip install -r requirements.txt

# copiar variáveis de ambiente
copy .env.example .env     # no Windows
Edite o arquivo .env e adicione sua chave da API do OpenWeatherMap (OWM_API_KEY).

Fluxo do Projeto
Ingestão

bash
Copiar código
python -m src.ingest
Coleta dados current e 5-day/3-hour da API OWM e grava em SQLite + Parquet (data/bronze / data/silver).

Pré-processamento

bash
Copiar código
python -m src.preprocess
Limpa os dados, cria features (lags, médias móveis, hora/dia da semana) e gera datasets prontos para treino.

Treino do modelo

bash
Copiar código
python -m src.train
Treina um classificador (Regressão Logística balanceada) e salva models/rain_classifier.pkl com modelo + features + threshold calibrado.

Dashboard

bash
Copiar código
cd app
streamlit run streamlit_app.py
Interface interativa: KPIs, séries temporais, probabilidade de chuva forte e eventos recentes.

Estrutura do Projeto
bash
Copiar código
pi4-weather/
├─ .env.example          # template com variáveis de ambiente
├─ requirements.txt      # dependências do projeto
├─ src/
│   ├─ ingest.py         # coleta (API OWM → SQLite/Parquet)
│   ├─ preprocess.py     # tratamento e engenharia de features
│   ├─ train.py          # treino do modelo e escolha do threshold
│   ├─ utils_db.py       # helpers para SQLite
│   └─ owm_client.py     # cliente da API OWM
├─ data/
│   ├─ bronze.db         # banco SQLite (camada bronze)
│   └─ silver/           # Parquets pré-processados
├─ models/
│   └─ rain_classifier.pkl  # modelo treinado + threshold
└─ app/
    └─ streamlit_app.py  # dashboard
    


Próximos Passos
Expandir para outros modelos (RandomForest, XGBoost, Redes Neurais).

Split temporal (treino no passado, teste no futuro).

Avaliar thresholds diferentes por cidade/região.

Acrescentar análises sazonais (chuva por estação do ano).

Melhorar visual do dashboard (filtros adicionais, comparativo entre cidades).



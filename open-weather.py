import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")

lat = -23.533773 # Latitude de São Paulo
lon = -46.625290 # Longitude de São Paulo
url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&lang=pt_br&units=metric"

response = requests.get(url)
response.raise_for_status()
data = response.json()
dados = {
    "cidade": data["name"],
    "temperatura": data["main"]["temp"],
    "descricao": data["weather"][0]["description"],
    "velocidade_vento": data["wind"]["speed"]
    }
print(dados)
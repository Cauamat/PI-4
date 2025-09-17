import requests

with open("api_key.txt", "r") as file:
    api_key = file.read().strip()

lat = -23.533773 # Latitude de São Paulo
lon = -46.625290 # Longitude de São Paulo

url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"

response = requests.get(url)
response.raise_for_status()
data = response.json()
print(data)
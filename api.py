import requests
import json
import xml.etree.ElementTree as ET

# Dados para enviar na requisição, já ajustados conforme seu exemplo
data = {
  "locations": [
    "Ponta Grossa",
    "Minas Gerais"
  ],
  "config": {
    "route": {
      "optimized_route": True,
      "optimized_route_destination": "best",
      "calculate_return": True,
      "alternative_routes": "0",
      "avoid_locations": True,
      "avoid_locations_key": "",
      "type_route": "efficient"
    },
    "vehicle": {
      "type": "truck",
      "axis": "3",
      "top_speed": 90
    },
    "tolls": {
      "retroactive_date": ""
    },
    "freight_table": {
      "category": "C",
      "freight_load": "geral",
      "axis": "all"
    },
    "fuel_consumption": {
      "fuel_price": "6.00",
      "km_fuel": "5.0"
    },
    "private_places": {
      "max_distance_from_location_to_route": "1000",
      "categories": True,
      "areas": True,
      "contacts": True,
      "products": True,
      "services": True
    }
  },
  "show": {
    "tolls": True,
    "freight_table": True,
    "maneuvers": "false",
    "truck_scales": True,
    "static_image": True,
    "link_to_qualp": True,
    "private_places": False,
    "polyline": False,
    "simplified_polyline": False,
    "ufs": True,
    "fuel_consumption": True,
    "link_to_qualp_report": True
  },
  "format": "xml",
  "exception_key": ""
}


# Cabeçalhos da requisição
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Access-Token': 'jsulQgSYJbr5mfFlVyuWWXZennF0YO5v'
}

# Fazendo a requisição POST
response = requests.post('https://api.qualp.com.br/rotas/v4', json=data, headers=headers)

# Verificando se a requisição foi bem-sucedida
if response.status_code == 200:
    # A resposta é texto/XML, então não precisa usar .json()
    xml_response = response.text

    # Parseando o XML
    root = ET.fromstring(xml_response)

    # Encontrando todos os itens de pedágios
    pedagios = root.findall('.//pedagios/item')

    # Lista para armazenar os dados dos pedágios
    dados_pedagios = []

# Inicializando o total acumulado das tarifas dos pedágios
total_tarifas_pedagios = 0

# Encontrando todos os itens de pedágios e iterando sobre eles
pedagios = root.findall('.//pedagios/item')
for pedagio in pedagios:
    # Extrair as tarifas de cada pedágio e acumular
    tarifas = [float(item.text) for item in pedagio.findall('tarifa/item')]
    total_tarifas_pedagios += sum(tarifas)

# Após iterar por todos os pedágios, temos o total acumulado
distancia = root.find('.//distancia/valor').text
distancia_t = float(distancia)
duracao = root.find('.//duracao/valor').text
duracao_t = float(duracao)/3600
combustivel_texto = root.find('.//consumo_combustivel').text
combustivel = float(combustivel_texto) * 6
tabela_frete = root.find('.//tabela_frete/dados/C/item')
geral = tabela_frete.find('geral').text
valor_frete = float(geral)

print(f"Tarifa para carga geral: {valor_frete}")
print(f"Distância: {distancia_t}")
print(f"Duração: {duracao_t}")
print(f"Abast: {combustivel}")
print(f"Abast: {total_tarifas_pedagios}")
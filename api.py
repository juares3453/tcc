import requests
import json
import xml.etree.ElementTree as ET
import pyodbc
import requests
import pycep_correios
from geopy.geocoders import Nominatim

# R. Dragão do Mar, 81 - Praia de Iracema, Fortaleza - CE, 60060-390
endereco = pycep_correios.get_address_from_cep('60060390')

geolocator = Nominatim(user_agent="test_app")
location = geolocator.geocode(endereco['logradouro'] + ", " + endereco['cidade'] + " - " + endereco['bairro'])

print(location.latitude, location.longitude)


import requests
import json
import xml.etree.ElementTree as ET
import pyodbc
import requests
import pandas as pd
import pycep_correios
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sqlalchemy import create_engine
import time

server = 'JUARES-PC'
database = 'softran_rasador'
username = 'sa'
password = 'sof1209'
driver = 'ODBC Driver 17 for SQL Server' # Ou o driver apropriado para sua instalação

# Cria a string de conexão para o SQLAlchemy
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'

# Cria a engine de conexão
engine = create_engine(connection_string)

# Sua consulta SQL para buscar os dados
sql_query = """
SELECT distinct
    DATEPART(day, A.data) AS Dia,
    DATEPART(month, A.data) AS Mes,
    DATEPART(year, A.data) AS Ano,
    A.CdEmpresa,
	B.NrPlaca,
    C.DsTpVeiculo,
    D.DsModelo,
    B.DsAnoFabricacao,
    ISNULL(A.QtConfLeitorCar, 0) AS conf_carregamento,
    ISNULL(A.QtConfLeitorSmart, 0) AS conf_entrega,
    DATEDIFF(HOUR, CONVERT(time, A.HrSaida), CONVERT(time, A.HrChegada)) AS tempo_total,
    A.KM_C - A.KM_S AS km_rodado,
    A.NrAuxiliares AS auxiliares,
    A.VlCapacVeic AS capacidade,
    E.CdRomaneio,
	E.NrCep
FROM TC_HistEntregaFilial A
INNER JOIN SISVeicu B ON A.NrPlaca = B.NrPlaca
LEFT JOIN Sistpvei C ON B.CdTipoVeiculo = C.CdTpVeiculo
LEFT JOIN SISMdVei D ON B.CdModelo = D.CdModelo
LEFT JOIN CCERomIt E ON A.CdRomaneio = E.CdRomaneio AND A.CdEmpresa = E.CdEmpresa
WHERE ISDATE(A.HrChegada) = 1 
  AND ISDATE(A.HrSaida) = 1 
  AND A.KM_C <> 0 
  AND A.KM_C > A.KM_S
  AND E.CdRomaneio is not null
order by CdRomaneio
"""

# Executando a consulta e armazenando os resultados em um DataFrame
df = pd.read_sql(sql_query, engine)
engine.dispose()

# Configure um agente de usuário customizado com seu email ou URL do seu aplicativo
geolocator = Nominatim(user_agent="juarescanalle@gmail.com")

# Configure um RateLimiter para adicionar atrasos entre as solicitações
# e aumente o tempo limite para 10 segundos
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, error_wait_seconds=10)

# Função para aplicar geocodificação
def geocode_address(cep):
    try:
        location = geocode(f"{cep}, Brasil")
        return (location.latitude, location.longitude) if location else (None, None)
    except:
        return (None, None)

# Aplicando geocodificação no DataFrame
df['latitude_longitude'] = df['NrCep'].apply(geocode_address)

# Separando as tuplas de latitude e longitude em colunas distintas
df[['latitude', 'longitude']] = pd.DataFrame(df['latitude_longitude'].tolist(), index=df.index)

time.sleep(1)
# Salvando o DataFrame enriquecido em um novo arquivo CSV
df.to_csv('dados_enriquecidos.csv', index=False)

print("Dados enriquecidos salvos com sucesso!")

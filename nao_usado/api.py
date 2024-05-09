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
from tqdm import tqdm

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
SELECT DISTINCT
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
  AND E.CdRomaneio IS NOT NULL
ORDER BY E.CdRomaneio;
"""

# Executando a consulta e armazenando os resultados em um DataFrame
df = pd.read_sql(sql_query, engine)
engine.dispose()

geolocator = Nominatim(user_agent="seu_nome_de_aplicativo", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, error_wait_seconds=10)

def geocode_address(row):
    try:
        # Aqui usamos o CEP diretamente da linha
        location = geocode(f"{row['NrCep']}, Brasil")
        if location:
            return pd.Series([location.latitude, location.longitude])
        else:
            return pd.Series([None, None])
    except Exception as e:
        print(f"Erro ao geocodificar {row['NrCep']}: {e}")
        return pd.Series([None, None])


# Antes de aplicar a geocodificação, criamos a barra de progresso usando tqdm
# `tqdm.pandas()` patch o pandas apply para mostrar a barra de progresso
tqdm.pandas(desc="Geocodificando CEPs")

# Usamos `apply` com `axis=1` para passar a linha inteira à função
df[['latitude', 'longitude']] = df.progress_apply(geocode_address, axis=1)

# Não há necessidade de usar time.sleep(1) no final do script, a menos que você queira atrasar deliberadamente a execução do script por algum motivo
# Salva o DataFrame modificado
#df.to_csv("dados_enriquecidos.csv", index=False)

# Defina o nome da tabela onde você deseja inserir os dados
table_name = "TC_HistEntregaFilialLatLon"

# Insere os dados no banco, substituindo a tabela se ela já existir.
# A opção 'replace' é usada para substituir a tabela se ela já existir. Se desejar adicionar dados a uma tabela existente sem substituí-la, use 'append'.
# O parâmetro 'index=False' evita que o índice do DataFrame seja inserido como uma coluna na tabela do banco de dados.
# A opção 'if_exists' pode ser ajustada para 'append' se desejar adicionar os dados a uma tabela existente sem apagá-la.
df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)


print("Dados enriquecidos salvos com sucesso!")
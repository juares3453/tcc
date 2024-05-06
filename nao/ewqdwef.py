import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, insert
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
import pandas as pd

# Database credentials and connection setup
server = 'JUARES-PC'
database = 'softran_rasador'
username = 'sa'
password = 'sof1209'
driver = 'ODBC Driver 17 for SQL Server'
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'
engine = create_engine(connection_string)

# Cabeçalhos da requisição para a API
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOlwvXC8wLjAuMC4wOjgwMDFcL2FwaVwvdmVsb2dcL2F1dGgiLCJpYXQiOjE3MTI1NDM5MjcsImV4cCI6MTcxNzcyNzkyNywibmJmIjoxNzEyNTQzOTI3LCJqdGkiOiJiVGo3TUw4NG9TQ05QS3REIiwiYWN0aW9uIjoiIiwiaWQiOjY1NDEsInRva2VuIjoiIn0.jbpq7JrUrkPc9kP1kmRMTaxwyJ33AqGGVILz31qgyLs'
}

# Define schema
metadata = MetaData()
resultados_api = Table('ResumoViagem', metadata,
                       Column('CdEmpresa', Integer),
                       Column('CdRomaneio', Integer),
                       Column('TotalDistancia', Float),
                       Column('TotalDuracao', Float),
                       Column('TotalPedagio', Float),
                       Column('TotalCombustivel', Float),
                       Column('TotalViagem', Float),
                       )
metadata.create_all(engine)

# Setup retry strategy with exponential backoff
retry_strategy = Retry(
    total=5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # Updated parameter here
    backoff_factor=2
)

adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)

# Define the API request function
def make_request(url, data, headers):
    try:
        response = http.post(url, json=data, headers=headers, timeout=20)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        print(f"HTTP Error: {e}")
        return None

# SQL query to load data
sql_query = """
SELECT A.*, B.latitude AS LatOrigem, B.longitude AS LongOrigem FROM TC_HistEntregaFilialLatLon A
LEFT JOIN TC_HistEntregaFilialLatLonEmpresa B ON A.CdEmpresa = B.CdEmpresa 
WHERE A.latitude IS NOT NULL
"""
df = pd.read_sql_query(sql_query, engine)

# Session setup for database operations
Session = sessionmaker(bind=engine)
session = Session()

# Iterate over each row of the DataFrame
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing records"):
    data = {
        "sn_lat_lng": False,
        "sn_rota_alternativa": False,
        "sn_pedagio": True,
        "sn_balanca": True,
        "sn_calcu_volta": True,
        "tag": "TRUCK",
        "qtd_eixos": 6,
        "veiculo_km_litro": 3.00,
        "valor_medio_combustivel": 6.00,
        "rotas": [
            {"lat": row['LatOrigem'], "lng": row['LongOrigem']},
            {"lat": row['latitude'], "lng": row['longitude']}
        ]
    }
    response = make_request('https://velog.vertti.com.br/api/velog/roteiro', data, headers)

    if response and response.status_code == 200:
        track_info = response.json()[0]['track']  # Now correctly accessing the first item
        for rota in track_info.get('rotas', []):
            for leg in rota.get('legs', []):
                stmt = insert(resultados_api).values(
                    CdEmpresa=row['CdEmpresa'],
                    CdRomaneio=row['CdRomaneio'],
                    TotalDistancia=leg.get('distance', 0),
                    TotalDuracao=leg.get('duration', 0),
                    TotalPedagio=rota.get('vl_total_pedagio_original', 0),
                    TotalCombustivel=rota.get('vl_total_combustivel_original', 0),
                    TotalViagem=rota.get('vl_total_viagem_original', 0)
                )
                session.execute(stmt)
    else:
        print(f"API request failed with status {response.status_code if response else 'No response'}")

# Commit all changes
session.commit()

# Close the session
session.close()

print("Data processing completed.")

import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, insert
import requests
from tqdm import tqdm

# Credenciais para conexão ao banco de dados
server = 'JUARES-PC'
database = 'softran_rasador'
username = 'sa'
password = 'sof1209'
driver = 'ODBC Driver 17 for SQL Server'

# Cria a string de conexão para o SQLAlchemy
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'
engine = create_engine(connection_string)

# Define schema
metadata = MetaData()
resultados_api = Table('ResumoViagem', metadata,
                       Column('Id', Integer, primary_key=True, autoincrement=True),
                       Column('CdEmpresa', Integer),
                       Column('CdRomaneio', Integer),
                       Column('TotalDistancia', Float),
                       Column('TotalDuracao', Float),
                       Column('TotalPedagio', Float),
                       Column('TotalCombustivel', Float),
                       Column('TotalViagem', Float),
                       )
metadata.create_all(engine)

# Consulta SQL para carregar dados
sql_query = """
SELECT A.*, B.latitude AS LatOrigem, B.longitude AS LongOrigem FROM TC_HistEntregaFilialLatLon A
LEFT JOIN TC_HistEntregaFilialLatLonEmpresa B ON A.CdEmpresa = B.CdEmpresa 
WHERE A.latitude IS NOT NULL
"""
df = pd.read_sql_query(sql_query, engine)

# Cabeçalhos da requisição para a API
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOlwvXC8wLjAuMC4wOjgwMDFcL2FwaVwvdmVsb2dcL2F1dGgiLCJpYXQiOjE3MTI1NDM5MjcsImV4cCI6MTcxNzcyNzkyNywibmJmIjoxNzEyNTQzOTI3LCJqdGkiOiJiVGo3TUw4NG9TQ05QS3REIiwiYWN0aW9uIjoiIiwiaWQiOjY1NDEsInRva2VuIjoiIn0.jbpq7JrUrkPc9kP1kmRMTaxwyJ33AqGGVILz31qgyLs'
}

# Data insertion function
def insert_data(empresa, romaneio, distance, duration, pedagio, combustivel, viagem):
    with engine.connect() as conn:
        stmt = insert(resultados_api).values(
            CdEmpresa=empresa,
            CdRomaneio=romaneio,
            TotalDistancia=distance,
            TotalDuracao=duration,
            TotalPedagio=pedagio,
            TotalCombustivel=combustivel,
            TotalViagem=viagem
        )
        conn.execute(stmt)
        conn.commit()  # Ensure to commit the transaction


# Itera sobre cada linha do DataFrame
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

    # Realiza a requisição POST
    response = requests.post('https://velog.vertti.com.br/api/velog/roteiro', json=data, headers=headers)


        # API Call and data processing
    response = requests.post('https://velog.vertti.com.br/api/velog/roteiro', json=data, headers=headers)
    if response.status_code == 200:
        track_info = response.json()[0]['track']  # Assuming correct path to data
        for rota in track_info.get('rotas', []):
            for leg in rota.get('legs', []):
                insert_data(
                    empresa=row['CdEmpresa'],
                    romaneio=row['CdRomaneio'],
                    distance=leg.get('distance', 0),
                    duration=leg.get('duration', 0),
                    pedagio=rota.get('vl_total_pedagio_original', 0),
                    combustivel=rota.get('vl_total_combustivel_original', 0),
                    viagem=rota.get('vl_total_viagem_original', 0)
                )
    else:
        print("Failed to fetch data from API")

    # Output confirmation
    print("Data insertion completed.")

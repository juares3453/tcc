import pandas as pd
from sqlalchemy import create_engine
import requests
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, insert
from tqdm import tqdm

# Credenciais para conexão ao banco de dados
server = 'JUARES-PC'
database = 'softran_rasador'
username = 'sa'
password = 'sof1209'
driver = 'ODBC Driver 17 for SQL Server'  # Ou o driver apropriado para sua instalação

# Cria a string de conexão para o SQLAlchemy
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'

# Cria a engine de conexão
engine = create_engine(connection_string)

# Consulta SQL
sql_query = """
select A.*, B.latitude as LatOrigem, B.longitude as LongOrigem from TC_HistEntregaFilialLatLon A
	left join TC_HistEntregaFilialLatLonEmpresa B ON A.CdEmpresa = B.CdEmpresa 
where A.latitude is not null 
"""

# Executa a consulta e carrega os dados em um DataFrame
df = pd.read_sql_query(sql_query, engine)

# Cabeçalhos da requisição para a API
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOlwvXC8wLjAuMC4wOjgwMDFcL2FwaVwvdmVsb2dcL2F1dGgiLCJpYXQiOjE3MTI1NDM5MjcsImV4cCI6MTcxNzcyNzkyNywibmJmIjoxNzEyNTQzOTI3LCJqdGkiOiJiVGo3TUw4NG9TQ05QS3REIiwiYWN0aW9uIjoiIiwiaWQiOjY1NDEsInRva2VuIjoiIn0.jbpq7JrUrkPc9kP1kmRMTaxwyJ33AqGGVILz31qgyLs'
}


# Estrutura básica dos dados para a requisição
data_template = {
    "sn_lat_lng": False,
    "sn_rota_alternativa": False,
    "sn_pedagio": True,
    "sn_balanca": True,
    "sn_calcu_volta": True,
    "tag": "TRUCK",
    "qtd_eixos": 6,
    "veiculo_km_litro": 3.00,
    "valor_medio_combustivel": 6.00,
    "rotas": []
}

# Agrupa os dados por CdEmpresa e CdRomaneio
grouped = df.groupby(['CdEmpresa', 'CdRomaneio'])

# Itera sobre cada grupo
for (empresa, romaneio), group in grouped:
    data = data_template.copy()
    rotas = []
    for index, row in group.iterrows():
        # Adiciona a origem e o destino para cada item no romaneio
        rotas.append({"lat": row['LatOrigem'], "lng": row['LongOrigem']})
        rotas.append({"lat": row['latitude'], "lng": row['longitude']})
    data['rotas'] = rotas

    # Fazendo a requisição POST
    response = requests.post('https://velog.vertti.com.br/api/velog/roteiro', json=data, headers=headers)


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
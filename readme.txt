adicionar algumas variáveis as informações:
- trajeto de transferencia
- produtividade de cada filial e como calcular isso
- vazao de cada local
- maximo de 3 pessoas por caminhao
- numero de caminhoes que podem ser carregados pra poderem sair pra entrega na parte da manha


- ajustar dados de custo
- analisar os graficos
- colocar nos algoritmos


WITH Dados AS (
    SELECT
        ISNULL((SELECT TOP 1 F.DsNome FROM GTCFunDp F WHERE F.NrCPF= A.CdMotorista),0) AS [Motorista],
        ISNULL((SELECT TOP 1 '1' FROM SISVeicu X WHERE X.NrPlaca = A.NrPlaca AND X.InSituacaoVeiculo = 1),0) AS [InTipoVeiculo],
        *
    FROM 
        softran_rasador.dbo.TC_HistEntregaFilial A
),
Tabela AS (
    SELECT 
        nrnotafiscal,
        cdromaneio,
        MIN(CONVERT(DATETIME, CONVERT(VARCHAR, CAST(DtMovimentacao AS DATE)) + ' 06:00:00', 120)) AS DataMinima,
        MAX(CAST(DtMovimentacao AS DATETIME)) AS HoraMaxima,
        DATEDIFF(HOUR, 
                 MIN(CONVERT(DATETIME, CONVERT(VARCHAR, CAST(DtMovimentacao AS DATE)) + ' 06:00:00', 120)), 
                 MAX(CAST(DtMovimentacao AS DATETIME))) AS DiferencaEmHoras
    FROM 
        Dados A
    LEFT JOIN 
        softran_rasador.dbo.ESP35303 B ON A.NrPlaca = B.NrPlaca 
    LEFT JOIN 
        softran_rasador.dbo.ESP35302 C ON C.NrCodigoBarras = B.NrCodigoBarras 
    WHERE 
        B.InCategoria = '6'
        AND A.Data = CAST(B.DtMovimentacao AS DATE)
    GROUP BY 
        CdRomaneio,
        NrNotaFiscal
)
SELECT
    DATEPART(day, data) AS Dia,
    DATEPART(month, data) AS Mes,
    DATEPART(year, data) AS Ano,
    A.Filial,
    ISNULL(QtConfLeitorCar,0) as conf_carregamento,
    ISNULL(QtConfLeitorSmart,0) as conf_entrega,
    DATEDIFF(HOUR, CONVERT(time, HrSaida), CONVERT(time, HrChegada)) as tempo_total,
    KM_C - KM_S as km_rodado,
    NrAuxiliares as auxiliares,
    VlCapacVeic as capacidade,
    QtEntregas as entregas_total,
    QtEntregaEx as entregas_realizadas,
    QtVolume as volumes_total,
    QtVolumeEx as volumes_entregues,
    QtPeso as peso_total,
    QtPesoEx as peso_entregue,
    Frete as frete_total,
    FreteEx as frete_entregue
FROM 
    softran_rasador.dbo.TC_HistEntregaFilial A
LEFT JOIN 
    Tabela B ON A.CdRomaneio = B.CdRomaneio 
WHERE 
    ISDATE(Hrchegada) = 1
    AND ISDATE(Hrsaida) = 1
    AND KM_C <> 0 
    AND KM_C > 0
ORDER BY 
    Ano, Mes, Dia

    # Cabeçalhos da requisição
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOlwvXC8wLjAuMC4wOjgwMDFcL2FwaVwvdmVsb2dcL2F1dGgiLCJpYXQiOjE3MTI1NDM5MjcsImV4cCI6MTcxNzcyNzkyNywibmJmIjoxNzEyNTQzOTI3LCJqdGkiOiJiVGo3TUw4NG9TQ05QS3REIiwiYWN0aW9uIjoiIiwiaWQiOjY1NDEsInRva2VuIjoiIn0.jbpq7JrUrkPc9kP1kmRMTaxwyJ33AqGGVILz31qgyLs'
}


# Dados para enviar na requisição, já ajustados conforme seu exemplo
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
  {
    "cep": 95700000,
  },
  {
    "cep": 95700000,
  }
]
}
# Fazendo a requisição POST
response = requests.post('https://velog.vertti.com.br/api/velog/roteiro', json=data, headers=headers)

print(response.json()) 

if response.status_code == 200:
    print(response.json()) 
else:
    print(f"Erro na requisição: {response.status_code}")
    


def geocode_address(cep):
    api_key = "sua_api_key_aqui"  # Substitua pelo seu API Key real
    base_url = "https://geocode.maps.co/search"
    params = {"q": cep, "api_key": api_key}
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data and 'features' in data and len(data['features']) > 0:
            # Supondo que a estrutura inclui uma chave 'features' com os resultados
            lat = data['features'][0]['geometry']['coordinates'][1]  # Latitude
            lon = data['features'][0]['geometry']['coordinates'][0]  # Longitude
            return lat, lon
        else:
            print("Nenhum resultado encontrado para o CEP fornecido.")
            return None, None
    else:
        print(f"Erro na requisição da API de geocodificação: HTTP {response.status_code}")
        return None, None

# Iterar sobre os resultados
for row in cursor.fetchall():
    dia, mes, ano, cdEmpresa, nrPlaca, dsTpVeiculo, dsModelo, dsAnoFabricacao, conf_carregamento, conf_entrega, tempo_total, km_rodado, auxiliares, capacidade, cdRomaneio, nrCep = row

    # Usar o CEP para obter as coordenadas
    lat, lon = geocode_address(nrCep)
    if lat is not None and lon is not None:
        print(f"Coordenadas para o CEP {nrCep}: Latitude {lat}, Longitude {lon}")
    else:
        print(f"Não foi possível encontrar coordenadas para o CEP {nrCep}")

# Não esqueça de fechar a conexão
cursor.close()
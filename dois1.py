from flask import Flask, render_template, send_file
import matplotlib as mpl
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

mpl.use('Agg')
mpl.rcParams['figure.max_open_warning'] = 50
analise = Flask(__name__)

campos1 = ['Dia', 'Mes', 'Ano', 'DsTpVeiculo', 'VlCusto', 'km_rodado', 'VlCapacVeic',
       'NrAuxiliares', '%CapacidadeCarre', '%CapacidadeEntr', '%Entregas', '%VolumesEntr', '%PesoEntr', '%FreteCobrado', 'FreteEx',
       'Lucro', '%Lucro']

csv_filepath1 = os.path.join('df1.csv')

def get_dataframe1(csv_filepath1):
    df1 = pd.read_csv(csv_filepath1, encoding='cp1252', delimiter=';')
    return df1

@analise.route('/gerar_graficos')
def gerar_graficos():
    csv_filepath1 = os.path.join('df1.csv')
    df1 = get_dataframe1(csv_filepath1)

    # Substituição de vírgulas por pontos na coluna 'VlCusto' e 'Lucro'
    df1['VlCusto'] = df1['VlCusto'].str.replace(',', '.')
    df1['Lucro'] = df1['Lucro'].str.replace(',', '.')

    # Conversão de tipos de dados
    df1['VlCusto'] = pd.to_numeric(df1['VlCusto'], errors='coerce')
    df1['Lucro'] = pd.to_numeric(df1['Lucro'], errors='coerce')

    # Tratamento de Valores Nulos
    imputer = SimpleImputer(strategy='mean')
    df1[['VlCusto', 'Lucro']] = imputer.fit_transform(df1[['VlCusto', 'Lucro']])

    # Conversão de Tipos de Dados
    df1['Dia'] = df1['Dia'].astype(int)
    df1['Mes'] = df1['Mes'].astype(int)
    df1['Ano'] = df1['Ano'].astype(int)

    # Unificação de colunas de data
    df1['data'] = pd.to_datetime(df1[['Ano', 'Mes', 'Dia']].astype(str).agg('-'.join, axis=1), errors='coerce')

    # Remoção de Colunas Desnecessárias
    df1 = df1.drop(columns=['Ano', 'Mes', 'Dia']) 

    # Codificação de Variáveis Categóricas
    label_encoder = LabelEncoder()
    df1['DsTpVeiculo'] = label_encoder.fit_transform(df1['DsTpVeiculo'])
    df1['DsModelo'] = label_encoder.fit_transform(df1['DsModelo'])
    df1['DsAnoFabricacao'] = label_encoder.fit_transform(df1['DsAnoFabricacao'])
    df1['VlCapacVeic'] = label_encoder.fit_transform(df1['VlCapacVeic'])

    # Normalização e Padronização
    scaler = MinMaxScaler()
    df1[['km_rodado', 'VlCusto', 'FreteEx', 'Lucro', '%Lucro', 'DsTpVeiculo', 'DsModelo', 'DsAnoFabricacao', 'VlCapacVeic', 'NrAuxiliares', '%CapacidadeCarre', '%CapacidadeEntr', '%Entregas', '%VolumesEntr', '%PesoEntr', '%FreteCobrado']] = scaler.fit_transform(df1[['km_rodado', 'VlCusto', 'FreteEx', 'Lucro', '%Lucro', 'DsTpVeiculo', 'DsModelo', 'DsAnoFabricacao', 'VlCapacVeic', 'NrAuxiliares', '%CapacidadeCarre', '%CapacidadeEntr', '%Entregas', '%VolumesEntr', '%PesoEntr', '%FreteCobrado']])

    # Tratamento de Outliers
    Q1 = df1[['km_rodado', 'VlCusto', 'FreteEx', 'Lucro']].quantile(0.25)
    Q3 = df1[['km_rodado', 'VlCusto', 'FreteEx', 'Lucro']].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[~((df1[['km_rodado', 'VlCusto', 'FreteEx', 'Lucro']] < (Q1 - 1.5 * IQR)) | (df1[['km_rodado', 'VlCusto', 'FreteEx', 'Lucro']] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Verificação de Duplicatas
    df1.drop_duplicates(inplace=True)

    # Informações após o tratamento
    print("Shape do DataFrame:")
    print(df1.shape)
    print(" ")
    print("Valores nulos por coluna:")
    print(df1.isnull().sum())
    print(" ")
    print("Tipos de dados:")
    print(df1.dtypes)
    print(" ")
    print("Primeiras linhas do DataFrame:")
    print(df1.head())

    primeiro_dia = df1['data'].min().strftime("%d %b %Y") 
    ultimo_dia = df1['data'].max().strftime("%d %b %Y") 
    total_dias = df1['data'].max() - df1['data'].min()

    print(f"Primeira registro do caso 2: {primeiro_dia}")
    print(f"Último registro do caso 2: {ultimo_dia}")
    print(f"Total de dias do caso 2: {total_dias}")

    # Visualizações
    plt.figure(figsize=(10, 6))
    sns.histplot(df1['VlCusto'].dropna(), bins=50, kde=True)
    plt.title('Distribuição de VlCusto')
    plt.xlabel('VlCusto')
    plt.ylabel('Frequência')
    plt.savefig('static/graficos/new/dois/vlcusto_dois1.png') 
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df1['Lucro'].dropna(), bins=50, kde=True)
    plt.title('Distribuição de Lucro')
    plt.xlabel('Lucro')
    plt.ylabel('Frequência')
    plt.savefig('static/graficos/new/dois/lucro_dois1.png') 
    plt.close()

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df1, x='DsTpVeiculo', y='VlCusto')
    plt.title('Dispersão de VlCusto por Tipo de Veículo')
    plt.xlabel('Tipo de Veículo')
    plt.ylabel('VlCusto')
    plt.savefig('static/graficos/new/dois/vlcusto_tpveiculo_dois1.png') 
    plt.close()

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df1, x='DsTpVeiculo', y='Lucro')
    plt.title('Dispersão de Lucro por Tipo de Veículo')
    plt.xlabel('Tipo de Veículo')
    plt.ylabel('Lucro')
    plt.savefig('static/graficos/new/dois/dstpveiculo_lucro_dois1.png') 
    plt.close()

    plt.figure(figsize=(14, 8))
    df1['mes_ano'] = df1['data'].dt.to_period('M')
    ocorrencias_por_mes = df1.groupby('mes_ano').size()
    ocorrencias_por_mes.plot(kind='line')
    plt.title('Número de Ocorrências ao Longo do Tempo')
    plt.xlabel('Mês/Ano')
    plt.ylabel('Número de Ocorrências')
    plt.xticks(rotation=45)
    plt.savefig('static/graficos/new/dois/entregas_dois1.png') 
    plt.close()

    # Remoção de Colunas Desnecessárias
    df1 = df1.drop(columns=['mes_ano'])

    plt.figure(figsize=(12, 10))
    sns.heatmap(df1.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap de Correlações')
    plt.savefig('static/graficos/new/dois/corr_dois1.png') 
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df1, x='km_rodado', y='VlCusto')
    plt.title('Relação entre Quilometragem Rodada e VlCusto')
    plt.xlabel('Quilometragem Rodada')
    plt.ylabel('VlCusto')
    plt.savefig('static/graficos/new/dois/kmrodado_custo_dois1.png') 
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df1, x='DsTpVeiculo')
    plt.title('Distribuição de Tipos de Veículo')
    plt.xlabel('Tipo de Veículo')
    plt.ylabel('Frequência')
    plt.savefig('static/graficos/new/dois/tpveiculo_dois1.png') 
    plt.close()

    # Scatter plot para visualizar concentração
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df1, x='km_rodado', y='VlCusto', hue='DsTpVeiculo', palette='Set1')
    plt.title('Relação entre Quilometragem Rodada e VlCusto por Tipo de Veículo')
    plt.xlabel('Quilometragem Rodada')
    plt.ylabel('VlCusto')
    plt.legend(title='Tipo de Veículo')
    plt.savefig('static/graficos/new/dois/tpveiculo_km_vlcusto_dois1.png') 
    plt.close()

    return "Processamento concluído e informações exibidas no console."

if __name__ == '__main__':
    analise.run(debug=True)
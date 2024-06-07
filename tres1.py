from flask import Flask
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

mpl.use('Agg')
mpl.rcParams['figure.max_open_warning'] = 50
analise = Flask(__name__)

# Diretório para salvar os gráficos
graficos_dir = 'static/graficos'
os.makedirs(graficos_dir, exist_ok=True)

csv_filepath2 = os.path.join('df2.csv')

# Função para carregar e preparar os dados
def get_dataframe(filepath):
    df = pd.read_csv(filepath, encoding='cp1252', delimiter=';')
    return df

@analise.route('/gerar_graficos')
def gerar_graficos():
    df2 = get_dataframe(csv_filepath2)

    # Substituição de vírgulas por pontos na coluna 'VlCusto'
    df2['VlCusto'] = df2['VlCusto'].str.replace(',', '.')

    # Conversão de tipos de dados
    df2['VlCusto'] = pd.to_numeric(df2['VlCusto'], errors='coerce')

    # Conversão de colunas numéricas para tipos numéricos, tratando erros
    colunas_numericas = ['dtcte', 'mescte', 'anocte', 'dtemissao', 'mesemissao', 'anoemissao', 'dtocor', 'mesocor', 'anoocor', 'dtbaixa', 'mesbaixa', 'anobaixa', 'diasemissao', 'diasresolucao', 'NrBo', 'VlCusto']
    for coluna in colunas_numericas:
        df2[coluna] = pd.to_numeric(df2[coluna], errors='coerce')

    # Tratamento de Valores Nulos
    imputer_num = SimpleImputer(strategy='mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df2[colunas_numericas] = imputer_num.fit_transform(df2[colunas_numericas])

    colunas_categoricas = ['Resp', 'CLIENTE', 'DsLocal', 'tp_ocor', 'Situacao', 'dsocorrencia']
    df2[colunas_categoricas] = imputer_cat.fit_transform(df2[colunas_categoricas])

    # Codificação de Variáveis Categóricas
    label_encoder = LabelEncoder()
    for coluna in colunas_categoricas:
        df2[coluna] = label_encoder.fit_transform(df2[coluna].astype(str))

     # Conversão de Tipos de Dados
    df2['anocte'] = df2['anocte'].astype(int)
    df2['mescte'] = df2['mescte'].astype(int)
    df2['dtcte'] = df2['dtcte'].astype(int)
    df2['anoemissao'] = df2['anoemissao'].astype(int)
    df2['mesemissao'] = df2['mesemissao'].astype(int)
    df2['dtemissao'] = df2['dtemissao'].astype(int)
    df2['anoocor'] = df2['anoocor'].astype(int)
    df2['mesocor'] = df2['mesocor'].astype(int)
    df2['dtocor'] = df2['dtocor'].astype(int)
    df2['anobaixa'] = df2['anobaixa'].astype(int)
    df2['mesbaixa'] = df2['mesbaixa'].astype(int)
    df2['dtbaixa'] = df2['dtbaixa'].astype(int)

    # Combinação de colunas de datas
    df2['data_cte'] = pd.to_datetime(df2[['anocte', 'mescte', 'dtcte']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d', errors='coerce')
    df2['data_emissao_bo'] = pd.to_datetime(df2[['anoemissao', 'mesemissao', 'dtemissao']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d', errors='coerce')
    df2['data_ocor'] = pd.to_datetime(df2[['anoocor', 'mesocor', 'dtocor']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d', errors='coerce')
    df2['data_baixa'] = pd.to_datetime(df2[['anobaixa', 'mesbaixa', 'dtbaixa']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d', errors='coerce')

    # Exclusão de colunas de dia, mês e ano originais
    df2 = df2.drop(columns=[ 'dtemissao', 'mesemissao', 'anoemissao', 'dtocor', 'mesocor', 'anoocor', 'dtbaixa', 'mesbaixa', 'anobaixa', 'data_cte', 'data_emissao_bo', 'data_ocor', 'data_baixa', 'dtcte', 'mescte', 'anocte'])

    # Remoção de duplicatas
    df2.drop_duplicates(inplace=True)

    # Normalização dos dados
    scaler = MinMaxScaler()
    colunas_para_normalizar = ['diasresolucao', 'diasemissao', 'NrBo', 'dsocorrencia', 'CLIENTE', 'DsLocal', 'tp_ocor', 'VlCusto', 'Situacao']
    df2[colunas_para_normalizar] = scaler.fit_transform(df2[colunas_para_normalizar])

    # Análise e tratamento de outliers
    Q1 = df2['VlCusto'].quantile(0.25)
    Q3 = df2['VlCusto'].quantile(0.75)
    IQR = Q3 - Q1
    df2 = df2[~((df2['VlCusto'] < (Q1 - 1.5 * IQR)) | (df2['VlCusto'] > (Q3 + 1.5 * IQR)))]

    # Impressão das informações no console
    print("Shape do DataFrame:")
    print(df2.shape)
    print(" ")
    print("Valores nulos por coluna:")
    print(df2.isnull().sum())
    print(" ")
    print("Tipos de dados:")
    print(df2.dtypes)
    print(" ")
    print("Primeiras linhas do DataFrame:")
    print(df2.head())

    return "Processamento concluído e informações exibidas no console."

if __name__ == '__main__':
    analise.run(debug=True)
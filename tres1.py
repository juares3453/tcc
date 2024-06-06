from flask import Flask
import matplotlib as mpl
import os
from io import StringIO
import itertools
from tqdm import tqdm
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

mpl.use('Agg')
mpl.rcParams['figure.max_open_warning'] = 50
analise = Flask(__name__)

# Diretório para salvar os gráficos
graficos_dir = 'static/graficos'
os.makedirs(graficos_dir, exist_ok=True)

# Lista de campos
campos2 = ['Resp', 'CLIENTE', 'dtcte','mescte','anocte','dtemissao','mesemissao','anoemissao','dtocor','mesocor','anoocor','dtbaixa','mesbaixa',
 'anobaixa','diasemissao','diasresolucao','DsLocal', 'tp_ocor', 'Situacao','NrBo','dsocorrencia','VlCusto']

csv_filepath2 = os.path.join('df2.csv')

# Função para remover valores negativos
def remover_valores_negativos(df):
    for coluna in df.columns:
        if pd.api.types.is_numeric_dtype(df[coluna]):
            df[coluna] = df[coluna].apply(lambda x: x if x >= 0 else np.nan)
    return df

# Função para carregar e preparar os dados
def get_dataframe(filepath):
    df = pd.read_csv(filepath, encoding='cp1252', delimiter=';')
    df = remover_valores_negativos(df)
    return df

@analise.route('/gerar_graficos')
def gerar_graficos():
    csv_filepath = os.path.join('df2.csv')
    df = get_dataframe(csv_filepath)

    # Substituição de vírgulas por pontos na coluna 'VlCusto'
    df['VlCusto'] = df['VlCusto'].str.replace(',', '.')

    # Conversão de tipos de dados
    df['VlCusto'] = pd.to_numeric(df['VlCusto'], errors='coerce')

    # Tratamento de datas - preenchendo valores ausentes temporariamente com valores válidos
    df[['dtbaixa', 'mesbaixa', 'anobaixa']] = df[['dtbaixa', 'mesbaixa', 'anobaixa']].fillna(0)

    # Conversão de colunas para inteiros
    df[['dtbaixa', 'mesbaixa', 'anobaixa']] = df[['dtbaixa', 'mesbaixa', 'anobaixa']].astype(int)

    # Codificação de variáveis categóricas
    label_encoder = LabelEncoder()
    df['tp_ocor'] = label_encoder.fit_transform(df['tp_ocor'])
    df['Situacao'] = label_encoder.fit_transform(df['Situacao'])
    df['dsocorrencia'] = label_encoder.fit_transform(df['dsocorrencia'])
    df['DsLocal'] = label_encoder.fit_transform(df['DsLocal'])
    df['CLIENTE'] = label_encoder.fit_transform(df['CLIENTE'])

     # Combinação de colunas de dia, mês e ano em colunas de datas completas
    df['data_cte'] = pd.to_datetime(df[['anocte', 'mescte', 'dtcte']].astype(str).agg('-'.join, axis=1), errors='coerce')
    df['data_emissao_bo'] = pd.to_datetime(df[['anoemissao', 'mesemissao', 'dtemissao']].astype(str).agg('-'.join, axis=1), errors='coerce')
    df['data_ocor'] = pd.to_datetime(df[['anoocor', 'mesocor', 'dtocor']].astype(str).agg('-'.join, axis=1), errors='coerce')
    df['data_baixa'] = pd.to_datetime(df[['anobaixa', 'mesbaixa', 'dtbaixa']].astype(str).agg('-'.join, axis=1), errors='coerce')

    # Substituição de NaT por uma data padrão
    df['data_baixa'].fillna(pd.Timestamp('1900-01-01'), inplace=True)
    
    # Exclusão de colunas de dia, mês e ano originais, se necessário
    df = df.drop(columns=['dtcte', 'mescte', 'anocte', 'dtemissao', 'mesemissao', 'anoemissao', 'dtocor', 'mesocor', 'anoocor', 'dtbaixa', 'mesbaixa', 'anobaixa'])

    # Tratamento de valores nulos
    imputer = SimpleImputer(strategy='mean')
    df[['diasresolucao', 'DsLocal', 'diasemissao']] = imputer.fit_transform(df[['diasresolucao', 'DsLocal', 'diasemissao']])

    # Remoção de duplicatas
    df.drop_duplicates(inplace=True)

    # Normalização dos dados
    scaler = MinMaxScaler()
    colunas_para_normalizar = ['diasresolucao', 'diasemissao', 'NrBo', 'dsocorrencia', 'CLIENTE', 'VlCusto', 'DsLocal', 'Resp']
    df[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])

    # Análise e tratamento de outliers
    Q1 = df['VlCusto'].quantile(0.25)
    Q3 = df['VlCusto'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['VlCusto'] < (Q1 - 1.5 * IQR)) | (df['VlCusto'] > (Q3 + 1.5 * IQR)))]

    # Impressão das informações no console
    print("Shape do DataFrame:")
    print(df.shape)
    print(" ")
    print("Valores nulos por coluna:")
    print(df.isnull().sum())
    print(" ")
    print("Tipos de dados:")
    print(df.dtypes)
    print(" ")
    print("Primeiras linhas do DataFrame:")
    print(df.head())

    return "Processamento concluído e informações exibidas no console."

if __name__ == '__main__':
    analise.run(debug=True)

from flask import Flask
import matplotlib as mpl
import os
from io import StringIO
import itertools
from tqdm import tqdm
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

import pandas as pd
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

# Função para carregar e preparar os dados
def get_dataframe(filepath):
    df2 = pd.read_csv(filepath, encoding='cp1252', delimiter=';')
    return df2

@analise.route('/gerar_graficos')
def gerar_graficos():
    csv_filepath2 = os.path.join('df2.csv')
    df2 = get_dataframe(csv_filepath2)

    # Substituição de vírgulas por pontos na coluna 'VlCusto'
    df2['VlCusto'] = df2['VlCusto'].str.replace(',', '.')

    # Conversão de tipos de dados
    df2['VlCusto'] = pd.to_numeric(df2['VlCusto'], errors='coerce')

    # Tratamento de datas - preenchendo valores ausentes temporariamente com valores válidos
    df2[['dtbaixa', 'mesbaixa', 'anobaixa']] = df2[['dtbaixa', 'mesbaixa', 'anobaixa']].fillna(0)

    # Conversão de colunas para inteiros
    df2[['dtbaixa', 'mesbaixa', 'anobaixa']] = df2[['dtbaixa', 'mesbaixa', 'anobaixa']].astype(int)

    # Codificação de variáveis categóricas
    label_encoder = LabelEncoder()
    df2['tp_ocor'] = label_encoder.fit_transform(df2['tp_ocor'])
    df2['Situacao'] = label_encoder.fit_transform(df2['Situacao'])
    df2['dsocorrencia'] = label_encoder.fit_transform(df2['dsocorrencia'])
    df2['DsLocal'] = label_encoder.fit_transform(df2['DsLocal'])
    df2['CLIENTE'] = label_encoder.fit_transform(df2['CLIENTE'])

     # Combinação de colunas de dia, mês e ano em colunas de datas completas
    df2['data_cte'] = pd.to_datetime(df2[['anocte', 'mescte', 'dtcte']].astype(str).agg('-'.join, axis=1), errors='coerce')
    df2['data_emissao_bo'] = pd.to_datetime(df2[['anoemissao', 'mesemissao', 'dtemissao']].astype(str).agg('-'.join, axis=1), errors='coerce')
    df2['data_ocor'] = pd.to_datetime(df2[['anoocor', 'mesocor', 'dtocor']].astype(str).agg('-'.join, axis=1), errors='coerce')
    df2['data_baixa'] = pd.to_datetime(df2[['anobaixa', 'mesbaixa', 'dtbaixa']].astype(str).agg('-'.join, axis=1), errors='coerce')

    # Substituição de NaT por uma data padrão
    df2['data_baixa'].fillna(pd.Timestamp('1900-01-01'), inplace=True)
    
    # Exclusão de colunas de dia, mês e ano originais, se necessário
    df2 = df2.drop(columns=['dtcte', 'mescte', 'anocte', 'dtemissao', 'mesemissao', 'anoemissao', 'dtocor', 'mesocor', 'anoocor', 'dtbaixa', 'mesbaixa', 'anobaixa', 'data_cte', 'data_emissao_bo', 'data_ocor', 'data_baixa'])

    # Tratamento de valores nulos
    imputer = SimpleImputer(strategy='mean')
    df2[['diasresolucao', 'DsLocal', 'diasemissao']] = imputer.fit_transform(df2[['diasresolucao', 'DsLocal', 'diasemissao']])

    # Remoção de duplicatas
    df2.drop_duplicates(inplace=True)

    # Normalização dos dados
    scaler = MinMaxScaler()
    colunas_para_normalizar = ['diasresolucao', 'diasemissao', 'NrBo', 'dsocorrencia', 'CLIENTE', 'VlCusto', 'DsLocal', 'Resp']
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

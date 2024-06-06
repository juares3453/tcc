from flask import Flask, render_template, send_file
import matplotlib as mpl
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

mpl.use('Agg')
mpl.rcParams['figure.max_open_warning'] = 50
analise = Flask(__name__)

# Diretório para salvar os gráficos
graficos_dir = 'static/graficos'
os.makedirs(graficos_dir, exist_ok=True)

# Lista de campos
campos = ['Dia', 'Mes', 'Ano', 'Filial', 'tempo_total', 'km_rodado', 'auxiliares', 'capacidade', 'entregas_total',
          'entregas_realizadas', 'volumes_total', 'volumes_entregues', 'peso_total', 
          'peso_entregue', 'frete_total', 'frete_entregue']

csv_filepath = os.path.join('df.csv')

def get_dataframe(csv_filepath):
    df = pd.read_csv(csv_filepath, encoding='cp1252', delimiter=';')
    return df

@analise.route('/gerar_graficos')
def dashboard_um_console():
    df = get_dataframe(csv_filepath)
    df_old = pd.read_csv(csv_filepath, encoding='cp1252', delimiter=';')

    # Tratamento de Valores Nulos
    imputer = SimpleImputer(strategy='mean')
    df[['frete_total', 'frete_entregue']] = imputer.fit_transform(df[['frete_total', 'frete_entregue']])

    # Conversão de Tipos de Dados
    df['Dia'] = df['Dia'].astype(int)
    df['Mes'] = df['Mes'].astype(int)
    df['Ano'] = df['Ano'].astype(int)

    # Unificação de colunas de data
    df['data'] = pd.to_datetime(df[['Ano', 'Mes', 'Dia']].astype(str).agg('-'.join, axis=1), errors='coerce')

    # Remoção de Colunas Desnecessárias
    df = df.drop(columns=['Ano', 'Mes', 'Dia', 'data'])

    # Codificação de Variáveis Categóricas
    label_encoder = LabelEncoder()
    df['Filial'] = label_encoder.fit_transform(df['Filial'])

    # Normalização e Padronização
    scaler = MinMaxScaler()
    df[['km_rodado', 'tempo_total', 'frete_total', 'frete_entregue', 'Filial', 'conf_carregamento', 
        'conf_entrega', 'auxiliares', 'capacidade',  'entregas_total', 'entregas_realizadas', 'volumes_total',  
        'volumes_entregues', 'peso_total', 'peso_entregue']] = scaler.fit_transform(df[['km_rodado', 'tempo_total', 'frete_total', 
        'frete_entregue', 'Filial', 'conf_carregamento', 'conf_entrega',  'auxiliares', 'capacidade', 
        'entregas_total', 'entregas_realizadas', 'volumes_total', 'volumes_entregues', 'peso_total', 'peso_entregue']])

    # Tratamento de Outliers
    Q1 = df[['km_rodado', 'tempo_total', 'frete_total', 'frete_entregue']].quantile(0.25)
    Q3 = df[['km_rodado', 'tempo_total', 'frete_total', 'frete_entregue']].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[['km_rodado', 'tempo_total', 'frete_total', 'frete_entregue']] < (Q1 - 1.5 * IQR)) 
    | (df[['km_rodado', 'tempo_total', 'frete_total', 'frete_entregue']] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Verificação de Duplicatas
    df.drop_duplicates(inplace=True)

    # Informações após o tratamento
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

    return "RESULTADO"

if __name__ == '__main__':
    analise.run(debug=True)
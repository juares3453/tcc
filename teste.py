from flask import Flask, render_template, send_file
import pandas as pd
from io import StringIO
from sqlalchemy import create_engine
import seaborn as sns
from io import BytesIO
import os

import matplotlib
matplotlib.use('Agg')  # Definir um backend que não depende de GUI
import matplotlib.pyplot as plt

teste = Flask(__name__)

# Diretório para salvar os gráficos
graficos_dir = 'static/graficos'
os.makedirs(graficos_dir, exist_ok=True)

# Configuração da conexão com o banco de dados
server = 'JUARES-PC'
database = 'softran_rasador'
username = 'sa'
password = 'sof1209'
connection_str = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=SQL+Server'
engine = create_engine(connection_str)

# Função para ler comandos SQL de um arquivo
def ler_sql_do_arquivo(nome_do_arquivo):
    with open(nome_do_arquivo, 'r') as arquivo:
        return arquivo.read()
    
# Lendo os comandos SQL dos arquivos
sql_comando = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC.sql")
sql_comando1 = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC Plus.sql")

# Função para obter um DataFrame a partir de um comando SQL
def get_dataframe(sql_comando):
    with engine.connect() as conn:
        df = pd.read_sql(sql_comando, conn)

    def index_of_dic(dic, key):
        return dic[key]

    def StrList_to_UniqueIndexList(lista):
        group = set(lista)
        print(group)

        dic = {}
        i = 0
        for g in group:
            if g not in dic:
                dic[g] = i
                i += 1

        print(dic)
        return [index_of_dic(dic, p) for p in lista]

    # Supondo que 'df1' é o seu DataFrame
    df['Filial'] = StrList_to_UniqueIndexList(df['Filial'])
    return df

def get_dataframe1(sql_comando1):
    with engine.connect() as conn:
        df1 = pd.read_sql(sql_comando1, conn)
    return df1

def gerar_grafico(df, tipo_grafico, nome_arquivo):
    plt.figure()  # Cria uma nova figura

    df['conf_carregamento'] = pd.to_numeric(df['conf_carregamento'], errors='coerce')

    if tipo_grafico == 'histograma':
        plt.figure(figsize=(10,6))
        sns.histplot(df.Filial)
        plt.title('Distribuição Filial - Carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
    elif tipo_grafico == 'boxplot':
        plt.figure(figsize=(10,6))
        sns.boxplot(x='Filial', y='conf_carregamento', data=df)  
        plt.title('Distribuição Filial - Carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
    elif tipo_grafico == 'displot':
        plt.figure(figsize=(10,6))
        sns.distplot(df.Filial,color='r')
        plt.title('Distribuição Filial - Carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
    elif tipo_grafico == 'hist':
        plt.figure(figsize=(10,6))
        plt.hist(df.Filial,color='y')
        plt.title('Distribuição Filial',size=18)   
    elif tipo_grafico == 'hist_car':    
        plt.figure(figsize=(10,6))
        plt.hist(df.conf_carregamento,color='y')
        plt.title('Distribuição Conf_carregamento',size=18) 
    elif tipo_grafico == 'box_iqr':   
        plt.figure(figsize = (10,6))
        sns.boxplot(df.Filial)
        plt.title('Distribution Filial',size=18)
        Q1 = df['Filial'].quantile(0.25)
        Q3 = df['Filial'].quantile(0.75)
        IQR = Q3 - Q1
        plt.ylabel(IQR,size=14)
        df[(df['Filial']< Q1-1.5* IQR) | (df['Filial']> Q3+1.5* IQR)]
    elif tipo_grafico == 'box_iqr_carre':   
        plt.figure(figsize = (10,6))
        sns.boxplot(df.conf_carregamento)
        plt.title('Distribution conf_carregamento',size=18)
        Q1 = df['conf_carregamento'].quantile(0.25)
        Q3 = df['conf_carregamento'].quantile(0.75)
        IQR = Q3 - Q1
        plt.ylabel(IQR,size=14)
        df[(df['conf_carregamento']< Q1-1.5* IQR) | (df['conf_carregamento']> Q3+1.5* IQR)]
        
    caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
    plt.savefig(caminho_arquivo)
    plt.close()
    return os.path.join(f'{nome_arquivo}.png')

# Rota principal que renderiza a página inicial
@teste.route('/')
def index():
    return render_template('index.html')

# Rota para visualizar o dashboard
@teste.route('/dashboard')
def dashboard():
    df = get_dataframe(sql_comando)

    #Data outliers
    df[df.duplicated(keep='first')]
    df.drop_duplicates(keep='first',inplace=True)

    # Captura a saída de df.info()
    buffer = StringIO()
    df.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    dados_texto = {
        'colunas': df.columns.tolist(),
        'dados_originais': df.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df.shape,
        'describe': df.describe().to_html(classes='table'),
        'limpeza': df.isnull().sum()
    }

    graficos = {
        'histograma': gerar_grafico(df, 'histograma', 'histograma_filial'),
        'boxplot': gerar_grafico(df, 'boxplot', 'boxplot_filial'),
        'displot': gerar_grafico(df, 'displot', 'displot_filial'),
        'hist': gerar_grafico(df, 'hist', 'hist_filial'),
        'hist_car': gerar_grafico(df, 'hist_car', 'hist_carregamento'),
        'box_iqr': gerar_grafico(df, 'box_iqr', 'box_filial'),
        'box_iqr_carre': gerar_grafico(df, 'box_iqr_carre', 'box_carre')
    }

    return render_template('dashboard.html', graficos=graficos, dados_texto=dados_texto)

if __name__ == '__main__':
    teste.run(debug=True)

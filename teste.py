from flask import Flask, render_template, send_file
import pandas as pd
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
server = '10.0.0.14'
database = 'softran_rasador'
username = 'softran'
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
    return df

def get_dataframe1(sql_comando1):
    with engine.connect() as conn:
        df1 = pd.read_sql(sql_comando1, conn)
    return df

def gerar_grafico(df, tipo_grafico, nome_arquivo):
    plt.figure()  # Cria uma nova figura
    if tipo_grafico == 'histograma':
        sns.histplot(df['Filial'])  # Exemplo
    elif tipo_grafico == 'boxplot':
        sns.boxplot(x='Filial', y='conf_carregamento', data=df)  # Exemplo
    # Adicione mais condições conforme necessário

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
    sql_comando = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC.sql")
    df = get_dataframe(sql_comando)

    graficos = {
        'histograma': gerar_grafico(df, 'histograma', 'histograma_filial'),
        'boxplot': gerar_grafico(df, 'boxplot', 'boxplot_filial')
        # Adicione mais gráficos conforme necessário
    }

    return render_template('dashboard.html', graficos=graficos)

if __name__ == '__main__':
    teste.run(debug=True)

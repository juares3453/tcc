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
sql_comando = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC um.sql")
sql_comando1 = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC dois.sql")
sql_comando2 = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC tres.sql")

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

    def index_of_dic1(dic1, key1):
        return dic1[key1]

    def StrList_to_UniqueIndexList1(lista):
        group = set(lista)
        print(group)

        dic1 = {}
        i = 0
        for g in group:
             if g not in dic1:
                dic1[g] = i
                i += 1

        print(dic1)
        return [index_of_dic1(dic1, p) for p in lista]

    df1['DsTpVeiculo'] = StrList_to_UniqueIndexList1(df1['DsTpVeiculo'])

    def index_of_dic2(dic2, key2):
        return dic2[key2]

    def StrList_to_UniqueIndexList2(lista):
        group = set(lista)
        print(group)

        dic2 = {}
        i = 0
        for g in group:
            if g not in dic2:
                dic2[g] = i
                i += 1

        print(dic2)
        return [index_of_dic2(dic2, p) for p in lista]

    df1['DsModelo'] = StrList_to_UniqueIndexList2(df1['DsModelo'])
    return df1

def get_dataframe1(sql_comando2):
    with engine.connect() as conn:
        df2 = pd.read_sql(sql_comando2, conn)
    return df2

def gerar_grafico(df, tipo_grafico, nome_arquivo):
    plt.figure()  # Cria uma nova figura

    df['conf_carregamento'] = pd.to_numeric(df['conf_carregamento'], errors='coerce')

    if tipo_grafico == 'histograma':
        plt.figure(figsize=(10,6))
        sns.histplot(df.Filial)
        plt.title('Distribuição Filial - Carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'boxplot':
        plt.figure(figsize=(10,6))
        sns.boxplot(x='Filial', y='conf_carregamento', data=df)  
        plt.title('Distribuição Filial - Carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'displot':
        plt.figure(figsize=(10,6))
        sns.distplot(df.Filial,color='r')
        plt.title('Distribuição Filial - Carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'hist':
        plt.figure(figsize=(10,6))
        plt.hist(df.Filial,color='y')
        plt.title('Distribuição Filial',size=18)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'hist_car':    
        plt.figure(figsize=(10,6))
        plt.hist(df.conf_carregamento,color='y')
        plt.title('Distribuição Conf_carregamento',size=18)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'box_iqr':   
        plt.figure(figsize = (10,6))
        sns.boxplot(df.Filial)
        plt.title('Distribution Filial',size=18)
        Q1 = df['Filial'].quantile(0.25)
        Q3 = df['Filial'].quantile(0.75)
        IQR = Q3 - Q1
        plt.ylabel(IQR,size=14)
        df[(df['Filial']< Q1-1.5* IQR) | (df['Filial']> Q3+1.5* IQR)]
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
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
    elif tipo_grafico == 'countplot_carre':   
        plt.figure(figsize=(10,6))
        sns.countplot(x = 'conf_carregamento', data = df)
        plt.title('Distribution conf_carregamento',size=18)
        plt.xlabel('conf_carregamento',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'countplot_filial':  
        plt.figure(figsize=(10,6))
        sns.countplot(x = 'Filial', data = df)
        plt.title('Distribution Filial',size=18)
        plt.xlabel('Filial',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'countplot_fil':  
        plt.figure(figsize = (10,6))
        sns.countplot(df.conf_carregamento)
        plt.title('Distribution conf_carregamento',size=18)
        plt.xlabel('conf_carregamento',size=14)
        plt.ylabel('Count',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'countplot_filiall':  
        plt.figure(figsize = (10,6))
        sns.countplot(df.Filial)
        plt.title('Distribution Filial',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('Count',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'scatterplot':  
        plt.figure(figsize = (10,6))
        sns.scatterplot(x='Filial',y='conf_carregamento',color='r',data=df)
        plt.title('Filial vs Conf_carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'boxplot_fc':  
        plt.figure(figsize = (10,6))
        sns.set_style('darkgrid')
        sns.boxplot(x='Filial',y='conf_carregamento',data=df)
        plt.title('Filial vs Conf_carregamento',size=18)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'heatmap':
        plt.figure(figsize = (10,6))
        sns.heatmap(df.corr(),annot=True,square=True,
            cmap='RdBu',
            vmax=1,
            vmin=-1)
        plt.title('Correlations Between Variables',size=18)
        plt.xticks(size=13)
        plt.yticks(size=13)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'pairplot':
        sns.pairplot(df, 
                 markers="+",
                 diag_kind="kde",
                 kind='reg',
                 plot_kws={'line_kws':{'color':'#aec6cf'}, 
                           'scatter_kws': {'alpha': 0.7, 
                                           'color': 'red'}},
                 corner=True)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'displot_filial_entrega':
        plt.figure(figsize=(10,6))
        sns.distplot(df.Filial,color='r')
        plt.title('Distribuição Filial - Entrega',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_entrega',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'histplot_filial_entrega':
        plt.figure(figsize=(10,6))
        sns.histplot(df.Filial)
        plt.title('Distribuição Filial - Entrega',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_entrega',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'boxplot_entrega':
        plt.figure(figsize = (10,6))
        sns.boxplot(df.Filial)
        plt.title('Distribution conf_entrega',size=18)
        Q1 = df['conf_entrega'].quantile(0.25)
        Q3 = df['conf_entrega'].quantile(0.75)
        IQR = Q3 - Q1
        df[(df['conf_entrega']< Q1-1.5* IQR) | (df['conf_entrega']> Q3+1.5* IQR)]
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'boxplot_filial_entrega':
        plt.figure(figsize = (10,6))
        sns.boxplot(df.Filial)
        plt.title('Distribution conf_entrega',size=18)
        Q1 = df['conf_entrega'].quantile(0.25)
        Q3 = df['conf_entrega'].quantile(0.75)
        IQR = Q3 - Q1
        df[(df['conf_entrega']< Q1-1.5* IQR) | (df['conf_entrega']> Q3+1.5* IQR)]
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'gender_entrega':
        plt.figure(figsize=(10,6))
        sns.countplot(x = 'Filial', data = df)
        plt.title('Distribution Conferencia entrega',size=18)
        plt.xlabel('conf_entrega',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()


    return os.path.join(f'{nome_arquivo}.png')

def gerar_grafico1(df1, tipo_grafico, nome_arquivo):
    plt.figure()  # Cria uma nova figura

    if tipo_grafico == 'histograma':
        plt.figure(figsize=(10,6))
        sns.histplot(df1.Filial)
        plt.title('Distribuição Filial - Carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'boxplot':
        plt.figure(figsize=(10,6))
        sns.boxplot(x='Filial', y='conf_carregamento', data=df1)  
        plt.title('Distribuição Filial - Carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'displot':
        plt.figure(figsize=(10,6))
        sns.distplot(df1.Filial,color='r')
        plt.title('Distribuição Filial - Carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'hist':
        plt.figure(figsize=(10,6))
        plt.hist(df1.Filial,color='y')
        plt.title('Distribuição Filial',size=18)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'hist_car':    
        plt.figure(figsize=(10,6))
        plt.hist(df1.conf_carregamento,color='y')
        plt.title('Distribuição Conf_carregamento',size=18)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'box_iqr':   
        plt.figure(figsize = (10,6))
        sns.boxplot(df1.Filial)
        plt.title('Distribution Filial',size=18)
        Q1 = df11['Filial'].quantile(0.25)
        Q3 = df1['Filial'].quantile(0.75)
        IQR = Q3 - Q1
        plt.ylabel(IQR,size=14)
        df1[(df1['Filial']< Q1-1.5* IQR) | (df1['Filial']> Q3+1.5* IQR)]
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'box_iqr_carre':   
        plt.figure(figsize = (10,6))
        sns.boxplot(df1.conf_carregamento)
        plt.title('Distribution conf_carregamento',size=18)
        Q1 = df1['conf_carregamento'].quantile(0.25)
        Q3 = df1['conf_carregamento'].quantile(0.75)
        IQR = Q3 - Q1
        plt.ylabel(IQR,size=14)
        df1[(df1['conf_carregamento']< Q1-1.5* IQR) | (df1['conf_carregamento']> Q3+1.5* IQR)]
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'countplot_carre':   
        plt.figure(figsize=(10,6))
        sns.countplot(x = 'conf_carregamento', data = df1)
        plt.title('Distribution conf_carregamento',size=18)
        plt.xlabel('conf_carregamento',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'countplot_filial':  
        plt.figure(figsize=(10,6))
        sns.countplot(x = 'Filial', data = df1)
        plt.title('Distribution Filial',size=18)
        plt.xlabel('Filial',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'countplot_fil':  
        plt.figure(figsize = (10,6))
        sns.countplot(df1.conf_carregamento)
        plt.title('Distribution conf_carregamento',size=18)
        plt.xlabel('conf_carregamento',size=14)
        plt.ylabel('Count',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'countplot_filiall':  
        plt.figure(figsize = (10,6))
        sns.countplot(df1.Filial)
        plt.title('Distribution Filial',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('Count',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'scatterplot':  
        plt.figure(figsize = (10,6))
        sns.scatterplot(x='Filial',y='conf_carregamento',color='r',data=df1)
        plt.title('Filial vs Conf_carregamento',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_carregamento',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'boxplot_fc':  
        plt.figure(figsize = (10,6))
        sns.set_style('darkgrid')
        sns.boxplot(x='Filial',y='conf_carregamento',data=df1)
        plt.title('Filial vs Conf_carregamento',size=18)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'heatmap':
        plt.figure(figsize = (10,6))
        sns.heatmap(df1.corr(),annot=True,square=True,
            cmap='RdBu',
            vmax=1,
            vmin=-1)
        plt.title('Correlations Between Variables',size=18)
        plt.xticks(size=13)
        plt.yticks(size=13)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'pairplot':
        sns.pairplot(df1, 
                 markers="+",
                 diag_kind="kde",
                 kind='reg',
                 plot_kws={'line_kws':{'color':'#aec6cf'}, 
                           'scatter_kws': {'alpha': 0.7, 
                                           'color': 'red'}},
                 corner=True)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'displot_filial_entrega':
        plt.figure(figsize=(10,6))
        sns.distplot(df1.Filial,color='r')
        plt.title('Distribuição Filial - Entrega',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_entrega',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'histplot_filial_entrega':
        plt.figure(figsize=(10,6))
        sns.histplot(df1.Filial)
        plt.title('Distribuição Filial - Entrega',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_entrega',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()
    elif tipo_grafico == 'histplot_filial_entrega':
        plt.figure(figsize=(10,6))
        sns.histplot(df1.Filial)
        plt.title('Distribuição Filial - Entrega',size=18)
        plt.xlabel('Filial',size=14)
        plt.ylabel('conf_entrega',size=14)
        caminho_arquivo = os.path.join('static', 'graficos', f'{nome_arquivo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()

    return os.path.join(f'{nome_arquivo}.png')

# Rota principal que renderiza a página inicial
@teste.route('/')
def index():
    return render_template('index.html')

# Rota para visualizar o dashboard
@teste.route('/dashboard_um')
def dashboard_um():
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
        'box_iqr_carre': gerar_grafico(df, 'box_iqr_carre', 'box_carre'),
        'countplot_carre': gerar_grafico(df, 'countplot_carre', 'countplot'),
        'countplot_filial' : gerar_grafico(df, 'countplot_filial', 'countplot_filial'),
        #'countplot_fil': gerar_grafico(df, 'countplot_fil', 'countplot_fil')
        #'countplot_filiall': gerar_grafico(df, 'countplot_filiall', 'countplot_filiall'),
        'scatterplot':  gerar_grafico(df, 'scatterplot', 'scatterplot'),
        'boxplot_fc': gerar_grafico(df, 'boxplot_fc', 'boxplot_fc'),
        'heatmap': gerar_grafico(df, 'heatmap', 'heatmap'),
        #'pairplot': gerar_grafico(df, 'pairplot', 'pairplot')
        'displot_filial_entrega': gerar_grafico(df, 'displot_filial_entrega', 'displot_filial_entrega'),
        'histplot_filial_entrega': gerar_grafico(df, 'histplot_filial_entrega', 'histplot_filial_entrega')
        
            
    }

    return render_template('dashboard_um.html', graficos=graficos, dados_texto=dados_texto)

@teste.route('/dashboard_dois')
def dashboard_dois():
    df1 = get_dataframe(sql_comando1)

    #Data outliers
    df1[df1.duplicated(keep='first')]
    df1.drop_duplicates(keep='first',inplace=True)

    # Captura a saída de df.info()
    buffer = StringIO()
    df1.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    dados_texto = {
        'colunas': df1.columns.tolist(),
        'dados_originais': df1.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df1.shape,
        'describe': df1.describe().to_html(classes='table'),
        'limpeza': df1.isnull().sum()
    }

    graficos = {
        'histograma': gerar_grafico1(df1, 'histograma', 'histograma_filial'),
        'boxplot': gerar_grafico1(df1, 'boxplot', 'boxplot_filial'),
        'displot': gerar_grafico1(df1, 'displot', 'displot_filial'),
        'hist': gerar_grafico1(df1, 'hist', 'hist_filial'),
        'hist_car': gerar_grafico1(df1, 'hist_car', 'hist_carregamento'),
        'box_iqr': gerar_grafico1(df1, 'box_iqr', 'box_filial'),
        'box_iqr_carre': gerar_grafico1(df1, 'box_iqr_carre', 'box_carre'),
        'countplot_carre': gerar_grafico1(df1, 'countplot_carre', 'countplot'),
        'countplot_filial' : gerar_grafico1(df1, 'countplot_filial', 'countplot_filial'),
        'countplot_fil': gerar_grafico1(df1, 'countplot_fil', 'countplot_fil'),
        'countplot_filiall': gerar_grafico1(df1, 'countplot_filiall', 'countplot_filiall'),
        'scatterplot':  gerar_grafico1(df1, 'scatterplot', 'scatterplot'),
        'boxplot_fc': gerar_grafico1(df1, 'boxplot_fc', 'boxplot_fc'),
        'heatmap': gerar_grafico1(df1, 'heatmap', 'heatmap'),
        'pairplot': gerar_grafico1(df1, 'pairplot', 'pairplot'),
        'displot_filial_entrega': gerar_grafico1(df1, 'displot_filial_entrega', 'displot_filial_entrega'),
        'histplot_filial_entrega': gerar_grafico1(df1, 'histplot_filial_entrega', 'histplot_filial_entrega'),
        
            
    }

    return render_template('dashboard_dois.html', graficos=graficos, dados_texto=dados_texto)

@teste.route('/dashboard_tres')
def dashboard_tres():
    df2 = get_dataframe(sql_comando2)

    #Data outliers
    df2[df2.duplicated(keep='first')]
    df2.drop_duplicates(keep='first',inplace=True)

    # Captura a saída de df.info()
    buffer = StringIO()
    df2.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    dados_texto = {
        'colunas': df2.columns.tolist(),
        'dados_originais': df2.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df2.shape,
        'describe': df2.describe().to_html(classes='table'),
        'limpeza': df2.isnull().sum()
    }

    graficos = {
        'histograma': gerar_grafico(df2, 'histograma', 'histograma_filial'),
        'boxplot': gerar_grafico(df2, 'boxplot', 'boxplot_filial'),
        'displot': gerar_grafico(df2, 'displot', 'displot_filial'),
        'hist': gerar_grafico(df2, 'hist', 'hist_filial'),
        'hist_car': gerar_grafico(df2, 'hist_car', 'hist_carregamento'),
        'box_iqr': gerar_grafico(df2, 'box_iqr', 'box_filial'),
        'box_iqr_carre': gerar_grafico(df2, 'box_iqr_carre', 'box_carre'),
        'countplot_carre': gerar_grafico(df2, 'countplot_carre', 'countplot'),
        'countplot_filial' : gerar_grafico(df2, 'countplot_filial', 'countplot_filial'),
        'countplot_fil': gerar_grafico(df2, 'countplot_fil', 'countplot_fil'),
        'countplot_filiall': gerar_grafico(df2, 'countplot_filiall', 'countplot_filiall'),
        'scatterplot':  gerar_grafico(df2, 'scatterplot', 'scatterplot'),
        'boxplot_fc': gerar_grafico(df2, 'boxplot_fc', 'boxplot_fc'),
        'heatmap': gerar_grafico(df2, 'heatmap', 'heatmap'),
        'pairplot': gerar_grafico(df2, 'pairplot', 'pairplot'),
        'displot_filial_entrega': gerar_grafico(df2, 'displot_filial_entrega', 'displot_filial_entrega'),
        'histplot_filial_entrega': gerar_grafico(df2, 'histplot_filial_entrega', 'histplot_filial_entrega'),
        
            
    }

    return render_template('dashboard_tres.html', graficos=graficos, dados_texto=dados_texto)


if __name__ == '__main__':
    teste.run(debug=True)

from flask import Flask, render_template, send_file
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from io import StringIO
mpl.rcParams['figure.max_open_warning'] = 50

import matplotlib
matplotlib.use('Agg')  # Definir um backend que não depende de GUI
import matplotlib.pyplot as plt

analise = Flask(__name__)

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

# Lista de campos
campos = ['Mes', 'Ano', 'Filial', 'conf_carregamento', 'conf_entrega', 'Dia', 
          'tempo_total', 'km_rodado', 'auxiliares', 'capacidade', 'entregas_total',
          'entregas_realizadas', 'volumes_total', 'volumes_entregues', 'peso_total', 
          'peso_entregue', 'frete_total', 'frete_entregue']

campos1 = ['Dia', 'Mes', 'Ano', 'DsTpVeiculo', 'DsModelo', 'DsAnoFabricacao', 'VlCusto', 'km_rodado', 'VlCapacVeic',
       'NrAuxiliares', '%CapacidadeCarre', '%CapacidadeEntr', '%Entregas', '%VolumesEntr', '%PesoEntr', '%FreteCobrado', 'FreteEx',
       'Lucro', '%Lucro']

campos2 = ['Resp', 'CLIENTE', 'dtcte','mescte','anocte','dtemissao','mesemissao','anoemissao','dtocor','mesocor','anoocor','dtbaixa','mesbaixa',
 'anobaixa','diasemissao','diasresolucao','DsLocal', 'tp_ocor', 'Situacao','NrBo','dsocorrencia','VlCusto']

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

        dic = {}
        i = 0
        for g in group:
            if g not in dic:
                dic[g] = i
                i += 1

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

        dic1 = {}
        i = 0
        for g in group:
             if g not in dic1:
                dic1[g] = i
                i += 1

        return [index_of_dic1(dic1, p) for p in lista]

    df1['DsTpVeiculo'] = StrList_to_UniqueIndexList1(df1['DsTpVeiculo'])

    def index_of_dic2(dic2, key2):
        return dic2[key2]

    def StrList_to_UniqueIndexList2(lista):
        group = set(lista)

        dic2 = {}
        i = 0
        for g in group:
            if g not in dic2:
                dic2[g] = i
                i += 1

        return [index_of_dic2(dic2, p) for p in lista]

    df1['DsModelo'] = StrList_to_UniqueIndexList2(df1['DsModelo'])
    return df1

def get_dataframe2(sql_comando2):
    with engine.connect() as conn:
        df2 = pd.read_sql(sql_comando2, conn)

    def index_of_dic3(dic3, key3):
        return dic3[key3]

    def StrList_to_UniqueIndexList3(lista):
        group = set(lista)

        dic3 = {}
        i = 0
        for g in group:
             if g not in dic3:
                dic3[g] = i
                i += 1

        return [index_of_dic3(dic3, p) for p in lista]

    df2['DsLocal'] = StrList_to_UniqueIndexList3(df2['DsLocal'])

    def index_of_dic4(dic4, key4):
        return dic4[key4]

    def StrList_to_UniqueIndexList4(lista):
        group = set(lista)

        dic4 = {}
        i = 0
        for g in group:
            if g not in dic4:
                dic4[g] = i
                i += 1

        return [index_of_dic4(dic4, p) for p in lista]

    df2['tp_ocor'] = StrList_to_UniqueIndexList4(df2['tp_ocor'])

    def index_of_dic5(dic5, key5):
        return dic5[key5]

    def StrList_to_UniqueIndexList5(lista):
        group = set(lista)

        dic5 = {}
        i = 0
        for g in group:
            if g not in dic5:
                dic5[g] = i
                i += 1

        return [index_of_dic5(dic5, p) for p in lista]

    df2['Situacao'] = StrList_to_UniqueIndexList5(df2['Situacao'])

    def index_of_dic6(dic6, key6):
        return dic6[key6]

    def StrList_to_UniqueIndexList6(lista):
        group = set(lista)

        dic6 = {}
        i = 0
        for g in group:
            if g not in dic6:
                dic6[g] = i
                i += 1

        return [index_of_dic6(dic6, p) for p in lista]

    df2['dsocorrencia'] = StrList_to_UniqueIndexList6(df2['dsocorrencia'])
    return df2

# Função para gerar e salvar gráficos
def gerar_e_salvar_graficos(df, campos, nome_prefixo):
    for campo in campos:
        with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
                plt.figure(figsize=(10, 6))

                # Critérios para usar plt.hist ou sns.histplot
                if campo in ['peso_total', 'peso_entregue', 'frete_total', 'frete_entregue']:
                    plt.hist(df[campo], bins=30, color='blue', alpha=0.7)
                    plt.ylabel('Contagem', size=14)
                    plt.title(f'Histograma (plt.hist) de {campo}', size=18)
                elif df[campo].dtype in ['int64', 'float64']:
                    sns.histplot(df[campo], kde=True, color='green')
                    plt.ylabel('Densidade', size=14)
                    plt.title(f'Histograma (sns.histplot) de {campo}', size=18)
                else:
                    sns.countplot(x=campo, data=df)
                    plt.ylabel('Contagem', size=14)
                    plt.title(f'Distribuição de {campo}', size=18)

                plt.xlabel(campo, size=14)
                plt.xticks(rotation=45)
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_{campo}.png')
                plt.savefig(caminho_arquivo)
            finally:
                plt.close('all')

def gerar_e_salvar_graficos1(df1, campos1, nome_prefixo):
    for campo1 in campos1:
        with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
                plt.figure(figsize=(10, 6))

                # Critérios para usar plt.hist ou sns.histplot
                if campo1 in  ['VlCusto', 'km_rodado', 'Lucro', '%Lucro']:
                    plt.hist(df1[campo1], bins=30, color='blue', alpha=0.7)
                    plt.ylabel('Contagem', size=14)
                    plt.title(f'Histograma (plt.hist) de {campo1}', size=18)
                elif df1[campo1].dtype in ['int64', 'float64']:
                    sns.histplot(df1[campo1], kde=True, color='green')
                    plt.ylabel('Densidade', size=14)
                    plt.title(f'Histograma (sns.histplot) de {campo1}', size=18)
                else:
                    sns.countplot(x=campo1, data=df1)
                    plt.ylabel('Contagem', size=14)
                    plt.title(f'Distribuição de {campo1}', size=18)

                plt.xlabel(campo1, size=14)
                plt.xticks(rotation=45)
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_{campo1}.png')
                plt.savefig(caminho_arquivo)
            finally:
                plt.close('all')

def gerar_e_salvar_graficos2(df2, campos2, nome_prefixo):
    for campo2 in campos2:
        with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
                plt.figure(figsize=(10, 6))

    # Usar plt.hist para campos específicos
                if campo2 in ['dtcte','mescte','anocte','dtemissao','mesemissao','anoemissao','dtocor','mesocor','anoocor']:
                    plt.hist(df2[campo2], bins=30, color='blue', alpha=0.7)
                    plt.ylabel('Contagem', size=14)
                    plt.title(f'Histograma (plt.hist) de {campo2}', size=18)
                elif df2[campo2].dtype in ['int64', 'float64']:
                    sns.histplot(df2[campo2], kde=True, color='green')
                    plt.ylabel('Densidade', size=14)
                    plt.title(f'Histograma (sns.histplot) de {campo2}', size=18)
                else:
                    sns.countplot(x=campo2, data=df2)
                    plt.ylabel('Contagem', size=14)
                    plt.title(f'Distribuição de {campo2}', size=18)

                plt.xlabel(campo2, size=14)
                plt.xticks(rotation=45)
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_{campo2}.png')
                plt.savefig(caminho_arquivo)
            finally:
                plt.close('all')


@analise.route('/')
def index():
    return render_template('index.html')

# Exemplo de uso da função em uma rota Flask
@analise.route('/gerar_graficos')
def gerar_graficos():
    df = get_dataframe(sql_comando)
    gerar_e_salvar_graficos(df, campos, 'df')
    df1 = get_dataframe1(sql_comando1)
    gerar_e_salvar_graficos(df1, campos1, 'df1')
    df2 = get_dataframe2(sql_comando2)
    gerar_e_salvar_graficos(df2, campos2, 'df2')
    return "Gráficos gerados e salvos com sucesso!"

@analise.route('/dashboard_um')
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
        'describe_include0': df.describe(include='O').to_html(classes='table'),
        'limpeza': df.isnull().sum()
    }

        # Lista para armazenar os caminhos dos gráficos
    caminhos_graficos = [os.path.join(graficos_dir, f'df_{campo}.png') for campo in campos]

    return render_template('dashboard_um.html', dados_texto=dados_texto,  caminhos_graficos=caminhos_graficos)

@analise.route('/dashboard_dois')
def dashboard_dois():
    df1 = get_dataframe1(sql_comando1)

    #Data outliers1
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
        'describe_include0': df1.describe(include='O').to_html(classes='table'),
        'limpeza': df1.isnull().sum()
    }

    caminhos_graficos = [os.path.join(graficos_dir, f'df_{campo1}.png') for campo1 in campos1]

    return render_template('dashboard_dois.html', dados_texto=dados_texto,  caminhos_graficos=caminhos_graficos)

@analise.route('/dashboard_tres')
def dashboard_tres():
    df2 = get_dataframe2(sql_comando2)

    #Data outliers1
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
        'describe_include0': df2.describe(include='O').to_html(classes='table'),
        'limpeza': df2.isnull().sum()
    }

 
    caminhos_graficos = [os.path.join(graficos_dir, f'df_{campo2}.png') for campo2 in campos2]

    return render_template('dashboard_tres.html', dados_texto=dados_texto,  caminhos_graficos=caminhos_graficos)

# Rota para exibir um gráfico específico
@analise.route('/grafico/<campo>')
def mostrar_grafico(campo):
    caminho_arquivo = os.path.join(graficos_dir, f'{campo}.png')
    return send_file(caminho_arquivo, mimetype='image/png')

if __name__ == '__main__':
    analise.run(debug=True)
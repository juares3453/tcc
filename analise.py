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

    df2['DsLocal'] = StrList_to_UniqueIndexList1(df2['DsLocal'])

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

    df2['tp_ocor'] = StrList_to_UniqueIndexList2(df2['tp_ocor'])

    def index_of_dic3(dic3, key3):
        return dic3[key3]

    def StrList_to_UniqueIndexList3(lista):
        group = set(lista)
        print(group)

        dic3 = {}
        i = 0
        for g in group:
            if g not in dic3:
                dic3[g] = i
                i += 1

        print(dic3)
        return [index_of_dic3(dic3, p) for p in lista]

    df2['Situacao'] = StrList_to_UniqueIndexList3(df2['Situacao'])

    def index_of_dic4(dic4, key4):
        return dic4[key4]

    def StrList_to_UniqueIndexList4(lista):
        group = set(lista)
        print(group)

        dic4 = {}
        i = 0
        for g in group:
            if g not in dic4:
                dic4[g] = i
                i += 1

        print(dic4)
        return [index_of_dic4(dic4, p) for p in lista]

    df2['dsocorrencia'] = StrList_to_UniqueIndexList4(df2['dsocorrencia'])
    return df2

# Função para gerar e salvar gráficos
def gerar_e_salvar_graficos(df):
    for campo in campos:
        plt.figure(figsize=(10, 6))
        
        if df[campo].dtype in ['int64', 'float64']:
            sns.histplot(df[campo], kde=True)
        else:
            sns.countplot(x=campo, data=df)
        
        plt.title(f'Distribuição de {campo}', size=18)
        plt.xlabel(campo, size=14)
        plt.ylabel('Contagem' if df[campo].dtype not in ['int64', 'float64'] else 'Densidade', size=14)
        plt.xticks(rotation=45)

        # Salvar gráfico
        caminho_arquivo = os.path.join(graficos_dir, f'{campo}.png')
        plt.savefig(caminho_arquivo)
        plt.close()

def gerar_e_salvar_graficos(df1):
    for campo1 in campos1:
        plt.figure(figsize=(10, 6))
        
        if df1[campo1].dtype in ['int64', 'float64']:
            sns.histplot(df1[campo1], kde=True)
        else:
            sns.countplot(x=campo1, data=df1)
        
        plt.title(f'Distribuição de {campo1}', size=18)
        plt.xlabel(campo1, size=14)
        plt.ylabel('Contagem' if df1[campo1].dtype not in ['int64', 'float64'] else 'Densidade', size=14)
        plt.xticks(rotation=45)

        # Salvar gráfico
        caminho_arquivo = os.path.join(graficos_dir, f'{campo1}.png')
        plt.savefig(caminho_arquivo)
        plt.close()

def gerar_e_salvar_graficos(df2):
    for campo2 in campos2:
        plt.figure(figsize=(10, 6))
        
        if df2[campo2].dtype in ['int64', 'float64']:
            sns.histplot(df2[campo2], kde=True)
        else:
            sns.countplot(x=campo2, data=df2)
        
        plt.title(f'Distribuição de {campo2}', size=18)
        plt.xlabel(campo2, size=14)
        plt.ylabel('Contagem' if df2[campo2].dtype not in ['int64', 'float64'] else 'Densidade', size=14)
        plt.xticks(rotation=45)

        # Salvar gráfico
        caminho_arquivo = os.path.join(graficos_dir, f'{campo2}.png')
        plt.savefig(caminho_arquivo)
        plt.close()

# Exemplo de uso da função em uma rota Flask
@analise.route('/gerar_graficos')
def gerar_graficos():
    df = get_dataframe(sql_comando)  # Certifique-se de que esta função retorna o DataFrame correto
    gerar_e_salvar_graficos(df)
    df1 = get_dataframe(sql_comando1)  # Certifique-se de que esta função retorna o DataFrame correto
    gerar_e_salvar_graficos(df1)
    df2 = get_dataframe(sql_comando2)  # Certifique-se de que esta função retorna o DataFrame correto
    gerar_e_salvar_graficos(df2)
    return "Gráficos gerados e salvos com sucesso!"

# Rota para exibir um gráfico específico
@analise.route('/grafico/<campo>')
def mostrar_grafico(campo):
    caminho_arquivo = os.path.join(graficos_dir, f'{campo}.png')
    return send_file(caminho_arquivo, mimetype='image/png')

if __name__ == '__main__':
    analise.run(debug=True)
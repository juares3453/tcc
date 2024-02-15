from flask import Flask, render_template, send_file
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')  # Definir um backend que não depende de GUI
import matplotlib.pyplot as plt
import os
from io import StringIO
import itertools
from tqdm import tqdm
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import numpy as np
from sklearn.impute import SimpleImputer
mpl.rcParams['figure.max_open_warning'] = 50

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

    # Supondo que 'df1' é o seu DataFrame
    df['conf_carregamento'] = StrList_to_UniqueIndexList2(df['conf_carregamento'])

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

    # Supondo que 'df1' é o seu DataFrame
    df['conf_entrega'] = StrList_to_UniqueIndexList1(df['conf_entrega'])
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

    def index_of_dic7(dic7, key7):
        return dic7[key7]

    def StrList_to_UniqueIndexList7(lista):
        group = set(lista)

        dic7 = {}
        i = 0
        for g in group:
            if g not in dic7:
                dic7[g] = i
                i += 1

        return [index_of_dic7(dic7, p) for p in lista]

    df2['CLIENTE'] = StrList_to_UniqueIndexList7(df2['CLIENTE'])
    return df2

# Função para gerar e salvar gráficos
def gerar_e_salvar_graficos(df, campos, nome_prefixo):
 with tqdm(total=len(campos), desc="Gerando gráficos parte 1") as pbar:
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

                # Atualiza a barra de progresso
                pbar.update(1)
            finally:
                plt.close('all')

def gerar_e_salvar_graficos1(df1, campos1, nome_prefixo):
 with tqdm(total=len(campos1), desc="Gerando gráficos parte 2") as pbar:
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
                pbar.update(1)
            finally:
                plt.close('all')

def gerar_e_salvar_graficos2(df2, campos2, nome_prefixo):
 with tqdm(total=len(campos1), desc="Gerando gráficos parte 3") as pbar:
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
                pbar.update(1)
            finally:
                plt.close('all')

def gerar_e_salvar_graficos3(df, campos, nome_prefixo):
  with tqdm(total=len(campos), desc="Gerando gráficos parte 4") as pbar:  
    for campo in campos:
        with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
                plt.figure(figsize=(10, 6))

                # Cria um boxplot
                plt.boxplot(df[campo], vert=False, notch=True, patch_artist=True)
                plt.title(f'Box (plt.boxplot) de {campo}', size=18)
                plt.xlabel(campo, size=14)
                plt.xticks(rotation=45)
                
                # Calcula os potenciais outliers usando IQR
                Q1 = df[campo].quantile(0.25)
                Q3 = df[campo].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[campo] < Q1 - 1.5 * IQR) | (df[campo] > Q3 + 1.5 * IQR)]

                # Salva os outliers em um arquivo CSV
                #outliers.to_csv(os.path.join(graficos_dir, f'{nome_prefixo}_{campo}_outliers.csv'), index=False)

                # Remove os outliers do DataFrame original (opcional)
                df = df[~df.index.isin(outliers.index)]

                # Salva o boxplot como uma imagem
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_{campo}_boxplot.png')
                plt.savefig(caminho_arquivo)
                pbar.update(1)
            finally:
                plt.close('all')

def gerar_e_salvar_graficos4(df1, campos1, nome_prefixo):
 with tqdm(total=len(campos1), desc="Gerando gráficos parte 5") as pbar:
    for campo1 in campos1:
        with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
                plt.figure(figsize=(10, 6))

                # Cria um boxplot
                plt.boxplot(df1[campo1], vert=False, notch=True, patch_artist=True)
                plt.title(f'Box (plt.boxplot) de {campo1}', size=18)
                plt.xlabel(campo1, size=14)
                plt.xticks(rotation=45)
                
                # Calcula os potenciais outliers usando IQR
                Q1 = df1[campo1].quantile(0.25)
                Q3 = df1[campo1].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df1[(df1[campo1] < Q1 - 1.5 * IQR) | (df1[campo1] > Q3 + 1.5 * IQR)]

                # Salva os outliers em um arquivo CSV
                #outliers.to_csv(os.path.join(graficos_dir, f'{nome_prefixo}_{campo1}_outliers.csv'), index=False)

                # Remove os outliers do DataFrame original (opcional)
                df1 = df1[~df1.index.isin(outliers.index)]

                # Salva o boxplot como uma imagem
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_{campo1}_boxplot.png')
                plt.savefig(caminho_arquivo)
                pbar.update(1)
            finally:
                plt.close('all')

def gerar_e_salvar_graficos5(df2, campos2, nome_prefixo):
 with tqdm(total=len(campos2), desc="Gerando gráficos parte 6") as pbar:
    for campo2 in campos2:
        with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
                # Calcula os potenciais outliers usando IQR
             
                Q1 = df2[campo2].quantile(0.25)
                Q3 = df2[campo2].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df2[(df2[campo2] < Q1 - 1.5 * IQR) | (df2[campo2] > Q3 + 1.5 * IQR)]

                # Salva os outliers em um arquivo CSV
                #outliers.to_csv(os.path.join(graficos_dir, f'{nome_prefixo}_{campo2}_outliers.csv'), index=False)

                # Remove os outliers do DataFrame original (opcional)
                df2 = df2[~df2.index.isin(outliers.index)]

                plt.figure(figsize=(10, 6))

                # Cria um boxplot

                plt.boxplot(df2[campo2], vert=False, notch=True, patch_artist=True)
                plt.title(f'Box (plt.boxplot) de {campo2}', size=18)
                plt.xlabel(campo2, size=14)
                plt.xticks(rotation=45)
                
                # Salva o boxplot como uma imagem
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_{campo2}_boxplot.png')
                plt.savefig(caminho_arquivo)
                pbar.update(1)
            finally:
                plt.close('all')

def gerar_e_salvar_graficos_pairplot(df, campos, nome_prefixo):
  with tqdm(total=len(campos1), desc="Gerando gráficos parte 7") as pbar:  
    with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
               # Selecione apenas os campos numéricos do DataFrame
                df_numeric = df[campos]

                # Plote a matriz de gráficos de dispersão
                sns.pairplot(df_numeric)
             
                # Salva o boxplot como uma imagem
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_pairplot.png')
                plt.savefig(caminho_arquivo)
                pbar.update(1)
            finally:
                plt.close('all')

def gerar_e_salvar_graficos_scatterplot(df, campos, nome_prefixo):    
               
# Cria todas as combinações possíveis de pares de campos
 combinacoes = list(itertools.combinations(campos, 2))
 
 with tqdm(total=len(combinacoes), desc="Gerando gráficos parte 8") as pbar:  
    with plt.rc_context(rc={'figure.max_open_warning': 0}):
        for campo1, campo2 in combinacoes:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=campo1, y=campo2, color='r', data=df)
            plt.title(f'{campo1} vs {campo2}', size=18)
            plt.xlabel(campo1, size=14)
            plt.ylabel(campo2, size=14)
            
            # Salva o gráfico de dispersão como uma imagem
            caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_{campo1}_{campo2}_scatterplot.png')
            plt.savefig(caminho_arquivo)
            plt.close()
            pbar.update(1)


def gerar_e_salvar_graficos_pairplot_numerical_values(df, campos, nome_prefixo):
   
 with tqdm(total=len(campos), desc="Gerando gráficos parte 9") as pbar:   
    with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
               # Selecione apenas os campos numéricos do DataFrame
                df_numeric = df[campos]

                # Plote a matriz de gráficos de dispersão
                plt.figure(figsize=(15, 8))
                sns.pairplot(df_numeric, 
                 markers="+",
                 diag_kind="kde",
                 kind='reg',
                 plot_kws={'line_kws':{'color':'#aec6cf'}, 
                           'scatter_kws': {'alpha': 0.7, 
                                           'color': 'red'}},
                 corner=True);
             
                # Salva o boxplot como uma imagem
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_pairplot_numerical.png')
                plt.savefig(caminho_arquivo)
                pbar.update(1)
            finally:
                plt.close('all')

           
def gerar_e_salvar_graficos_heatmap(df, nome_prefixo):
   
 with tqdm(total=len(df), desc="Gerando gráficos parte 10") as pbar:   
    with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
                plt.figure(figsize=(20, 20))
                # Plote a matriz de gráficos de dispersão
                sns.heatmap(df.corr(),annot=True,square=True,
                cmap='RdBu',
                vmax=1,
                vmin=-1)
                plt.xticks(size=13)
                plt.yticks(size=13)
             
                # Salva o boxplot como uma imagem
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_heatmap.png')
                plt.savefig(caminho_arquivo)
                pbar.update(1)
            finally:
                plt.close('all')
                
@analise.route('/')
def index():
    return render_template('index.html')

# Exemplo de uso da função em uma rota Flask
@analise.route('/gerar_graficos')
def gerar_graficos():
    df = get_dataframe(sql_comando)
    df1 = get_dataframe1(sql_comando1)
    df2 = get_dataframe2(sql_comando2)
    
    gerar_e_salvar_graficos(df, campos, 'df')
    gerar_e_salvar_graficos(df1, campos1, 'df1')
    gerar_e_salvar_graficos(df2, campos2, 'df2')
    gerar_e_salvar_graficos3(df, campos, 'df')
    gerar_e_salvar_graficos4(df1, campos1, 'df1')
    gerar_e_salvar_graficos5(df2, campos2, 'df2')
    gerar_e_salvar_graficos_pairplot(df, campos, 'df')
    gerar_e_salvar_graficos_pairplot(df1, campos1, 'df1')
    gerar_e_salvar_graficos_pairplot(df2, campos2, 'df2')
    gerar_e_salvar_graficos_pairplot_numerical_values(df, campos, 'df')
    gerar_e_salvar_graficos_pairplot_numerical_values(df1, campos1, 'df1')
    gerar_e_salvar_graficos_pairplot_numerical_values(df2, campos2, 'df2')
    gerar_e_salvar_graficos_scatterplot(df, campos, 'df')        
    gerar_e_salvar_graficos_scatterplot(df1, campos1, 'df1')
    gerar_e_salvar_graficos_scatterplot(df2, campos2, 'df2')
    gerar_e_salvar_graficos_heatmap(df, 'df')       
    gerar_e_salvar_graficos_heatmap(df1, 'df1')
    gerar_e_salvar_graficos_heatmap(df2, 'df2')
    
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

    combinacoes = list(itertools.combinations(campos, 2))

    # Calculando correlação para todos os pares de campos
    correlacoes = {}
    for campo1 in campos:
        for campo2 in campos:
            if campo1 != campo2:
                correlacao = df.corr()[campo1][campo2]
                chave = f'{campo1} - {campo2}'
                correlacoes[chave] = correlacao

    Soma_distancia_quadratica = []
    K = range(1,10)
    for k in K:
     km = KMeans(n_clusters=k)
     km = km.fit(df)
     Soma_distancia_quadratica.append(km.inertia_)
     
    plt.plot(K, Soma_distancia_quadratica, 'b+-')
    plt.xlabel('k')
    plt.ylabel('Soma das distâncias quadráticas')
    plt.title('Método de Elbow para identificar o melhor valor do parâmetro k')
    caminho_arquivo = os.path.join(graficos_dir, 'df_cotovelo.png')
    plt.savefig(caminho_arquivo)

    dados_texto = {
        'colunas': df.columns.tolist(),
        'dados_originais': df.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df.shape,
        'describe': df.describe().to_html(classes='table'),
        #'describe_include0': df.describe(include='O').to_html(classes='table'),
        'limpeza': df.isnull().sum(),
        'correlacoes': correlacoes,
        'soma_quadratica': Soma_distancia_quadratica
    }
    
    # Lista para armazenar os caminhos dos gráficos
    caminhos_graficos = [f'graficos/df_{campo}.png' for campo in campos]
    caminhos_graficos1 = [f'graficos/df_{campo}_boxplot.png' for campo in campos]
    caminhos_graficos4 = [f'graficos/df_pairplot.png']
    caminhos_graficos7 = [f'graficos/df_{campo1}_{campo2}_scatterplot.png' for campo1, campo2 in combinacoes]
    caminhos_graficos10 = [f'graficos/df_heatmap.png']
    caminhos_graficos11 = [f'graficos/df_pairplot_numerical.png']
    caminhos_graficos16 = [f'graficos/df_cotovelo.png']

    return render_template('dashboard_um.html', dados_texto=dados_texto,  caminhos_graficos=caminhos_graficos, caminhos_graficos1=caminhos_graficos1, caminhos_graficos4=caminhos_graficos4, caminhos_graficos7=caminhos_graficos7 , caminhos_graficos10=caminhos_graficos10,
                           caminhos_graficos11=caminhos_graficos11, caminhos_graficos16=caminhos_graficos16 )

@analise.route('/dashboard_dois')
def dashboard_dois():
    df1 = get_dataframe1(sql_comando1)

    #Data outliers1
    df1[df1.duplicated(keep='first')]
    df1.drop_duplicates(keep='first',inplace=True)

    combinacoes = list(itertools.combinations(campos1, 2))

    # Captura a saída de df.info()
    buffer = StringIO()
    df1.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    correlacoes = {}
    for campo1 in campos1:
        for campo2 in campos1:
            if campo1 != campo2:
                correlacao = df1.corr()[campo1][campo2]
                chave = f'{campo1} - {campo2}'
                correlacoes[chave] = correlacao

   

    imputer = SimpleImputer(strategy='mean')
    df1_filled = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)

    Soma_distancia_quadratica = []
    K = range(1,10)
    for k in K:
     km = KMeans(n_clusters=k)
     km = km.fit(df1_filled)
     Soma_distancia_quadratica.append(km.inertia_)
     
    plt.plot(K, Soma_distancia_quadratica, 'b+-')
    plt.xlabel('k')
    plt.ylabel('Soma das distâncias quadráticas')
    plt.title('Método de Elbow para identificar o melhor valor do parâmetro k')
    caminho_arquivo = os.path.join(graficos_dir, 'df1_cotovelo.png')
    plt.savefig(caminho_arquivo)


    dados_texto = {
        'colunas': df1.columns.tolist(),
        'dados_originais': df1.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df1.shape,
        'describe': df1.describe().to_html(classes='table'),
        #'describe_include0': df1.describe(include='O').to_html(classes='table'),
        'limpeza': df1.isnull().sum(),
        'correlacoes': correlacoes,
        'soma_quadratica': Soma_distancia_quadratica
    }
    
    caminhos_graficos = [f'graficos/df1_{campo1}.png' for campo1 in campos1]
    caminhos_graficos2 = [f'graficos/df1_{campo1}_boxplot.png' for campo1 in campos1]
    caminhos_graficos5 = [f'graficos/df1_pairplot.png']
    caminhos_graficos8 = [f'graficos/df1_{campo1}_{campo2}_scatterplot.png' for campo1, campo2 in combinacoes]
    caminhos_graficos12 = [f'graficos/df1_heatmap.png']
    caminhos_graficos13 = [f'graficos/df1_pairplot_numerical.png']
    caminhos_graficos17 = [f'graficos/df1_cotovelo.png']


    return render_template('dashboard_dois.html', dados_texto=dados_texto,  caminhos_graficos=caminhos_graficos, caminhos_graficos2=caminhos_graficos2, caminhos_graficos5=caminhos_graficos5, caminhos_graficos8=caminhos_graficos8, caminhos_graficos12=caminhos_graficos12,
                           caminhos_graficos13=caminhos_graficos13, caminhos_graficos17=caminhos_graficos17  )

@analise.route('/dashboard_tres')
def dashboard_tres():
    df2 = get_dataframe2(sql_comando2)

    #Data outliers1
    df2[df2.duplicated(keep='first')]
    df2.drop_duplicates(keep='first',inplace=True)

    combinacoes = list(itertools.combinations(campos2, 2))

    # Captura a saída de df.info()
    buffer = StringIO()
    df2.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    correlacoes = {}
    for campo1 in campos2:
        for campo2 in campos2:
            if campo1 != campo2:
                correlacao = df2.corr()[campo1][campo2]
                chave = f'{campo1} - {campo2}'
                correlacoes[chave] = correlacao

    imputer = SimpleImputer(strategy='mean')
    df2_filled = pd.DataFrame(imputer.fit_transform(df2), columns=df2.columns)

    Soma_distancia_quadratica = []
    K = range(1,10)
    for k in K:
     km = KMeans(n_clusters=k)
     km = km.fit(df2_filled)
     Soma_distancia_quadratica.append(km.inertia_)
     
    plt.plot(K, Soma_distancia_quadratica, 'b+-')
    plt.xlabel('k')
    plt.ylabel('Soma das distâncias quadráticas')
    plt.title('Método de Elbow para identificar o melhor valor do parâmetro k')
    caminho_arquivo = os.path.join(graficos_dir, 'df2_cotovelo.png')
    plt.savefig(caminho_arquivo)

    dados_texto = {
        'colunas': df2.columns.tolist(),
        'dados_originais': df2.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df2.shape,
        'describe': df2.describe().to_html(classes='table'),
       # 'describe_include0': df2.describe(include='O').to_html(classes='table'),
        'limpeza': df2.isnull().sum(),
        'correlacoes': correlacoes,
        'soma_quadratica': Soma_distancia_quadratica
    }


    caminhos_graficos = [f'graficos/df2_{campo2}.png' for campo2 in campos2]
    caminhos_graficos3 = [f'graficos/df2_{campo2}_boxplot.png' for campo2 in campos2]
    caminhos_graficos6 = [f'graficos/df2_pairplot.png']
    caminhos_graficos9 = [f'graficos/df2_{campo1}_{campo2}_scatterplot.png' for campo1, campo2 in combinacoes]
    caminhos_graficos14 = [f'graficos/df2_heatmap.png']
    caminhos_graficos15 = [f'graficos/df2_pairplot_numerical.png']
    caminhos_graficos18 = [f'graficos/df2_cotovelo.png']

    return render_template('dashboard_tres.html', dados_texto=dados_texto,  caminhos_graficos=caminhos_graficos, caminhos_graficos3=caminhos_graficos3, caminhos_graficos6=caminhos_graficos6, caminhos_graficos9=caminhos_graficos9, caminhos_graficos14=caminhos_graficos14,
                           caminhos_graficos15=caminhos_graficos15,  caminhos_graficos18=caminhos_graficos18 )


# Rota para exibir um gráfico específico
@analise.route('/grafico/<campo>')
def mostrar_grafico(campo):
    caminho_arquivo = os.path.join(graficos_dir, f'{campo}.png')
    return send_file(caminho_arquivo, mimetype='image/png')

if __name__ == '__main__':
    analise.run(debug=True)
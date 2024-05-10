from flask import Flask, render_template, send_file
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')  # Definir um backend que não depende de GUI
import os
from io import StringIO
import itertools
from tqdm import tqdm
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from IPython.display import HTML, display
from sklearn.tree import export_text
from flask import send_file
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree, _tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.model_selection import cross_val_predict

mpl.rcParams['figure.max_open_warning'] = 50

analise = Flask(__name__)

# Diretório para salvar os gráficos
graficos_dir = 'static/graficos'
os.makedirs(graficos_dir, exist_ok=True)

# Configuração da conexão com o banco de dados
# server = 'JUARES-PC'
# database = 'softran_rasador'
# username = 'sa'
# password = 'sof1209'
# connection_str = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=SQL+Server'
# engine = create_engine(connection_str)

# Lista de campos
campos = ['Dia', 'Mes', 'Ano', 'Filial', 'conf_carregamento', 'conf_entrega',
          'tempo_total', 'km_rodado', 'auxiliares', 'capacidade', 'entregas_total',
          'entregas_realizadas', 'volumes_total', 'volumes_entregues', 'peso_total', 
          'peso_entregue', 'frete_total', 'frete_entregue']

campos1 = ['Dia', 'Mes', 'Ano', 'DsTpVeiculo', 'DsModelo', 'DsAnoFabricacao', 'VlCusto', 'km_rodado', 'VlCapacVeic',
       'NrAuxiliares', '%CapacidadeCarre', '%CapacidadeEntr', '%Entregas', '%VolumesEntr', '%PesoEntr', '%FreteCobrado', 'FreteEx',
       'Lucro', '%Lucro']

campos2 = ['Resp', 'CLIENTE', 'dtcte','mescte','anocte','dtemissao','mesemissao','anoemissao','dtocor','mesocor','anoocor','dtbaixa','mesbaixa',
 'anobaixa','diasemissao','diasresolucao','DsLocal', 'tp_ocor', 'Situacao','NrBo','dsocorrencia','VlCusto']

# # Função para ler comandos SQL de um arquivo
# def ler_sql_do_arquivo(nome_do_arquivo):
#     with open(nome_do_arquivo, 'r') as arquivo:
#         return arquivo.read()
    
# # Lendo os comandos SQL dos arquivos
# sql_comando = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC um.sql")
# sql_comando1 = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC dois.sql")
# sql_comando2 = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC tres.sql")

csv_filepath = 'C:\\Users\\juare\\Desktop\\TCC\\df.csv'
csv_filepath1 = 'C:\\Users\\juare\\Desktop\\TCC\\df1.csv'
csv_filepath2 = 'C:\\Users\\juare\\Desktop\\TCC\\df2.csv'

def remover_valores_negativos(df):
    for coluna in df.columns:
        if pd.api.types.is_numeric_dtype(df[coluna]):
            df[coluna] = df[coluna].apply(lambda x: x if x >= 0 else np.nan)
    return df

# Função para obter um DataFrame a partir de um comando SQL
def get_dataframe(csv_filepath):
    df = pd.read_csv(csv_filepath, encoding='cp1252', delimiter=';')

    # with engine.connect() as conn:
    #     df = pd.read_sql(sql_comando, conn)
    df = remover_valores_negativos(df)
    df.dropna(inplace=True)

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

def get_dataframe1(csv_filepath1):
    df1 = pd.read_csv(csv_filepath1, encoding='cp1252', delimiter=';')
    df1 = remover_valores_negativos(df1)
    df1.dropna(inplace=True)

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
    df1['VlCusto'] = df1['VlCusto'].str.replace(',', '.').astype(float)
    df1['Lucro'] = df1['Lucro'].str.replace(',', '.').astype(float)
    return df1

def get_dataframe2(csv_filepath2):
    df2 = pd.read_csv(csv_filepath2, encoding='cp1252', delimiter=';')
    df2 = remover_valores_negativos(df2)
    df2.dropna(inplace=True)

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
    df2['VlCusto'] = df2['VlCusto'].str.replace(',', '.').astype(float)
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
                plt.yticks(rotation =0)
             
                # Salva o boxplot como uma imagem
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_heatmap.png')
                plt.savefig(caminho_arquivo)
                pbar.update(1)
            finally:
                plt.close('all')

def perform_clustering_and_generate_graphs(df, n_clusters_range, nome_prefixo):
    df_std = StandardScaler().fit_transform(df)  # Standardizing data
    for n_clusters in n_clusters_range:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(df_std)

        # Generating silhouette analysis graph
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        silhouette_avg = silhouette_score(df_std, cluster_labels)
        sample_silhouette_values = silhouette_samples(df_std, cluster_labels)

        # 1st subplot - The silhouette plot
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(df_std) + (n_clusters + 1) * 10])
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = len(ith_cluster_silhouette_values)
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            # Add cluster number in the middle of the silhouette
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), color="red", fontweight='bold')
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for various clusters")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(df_std[:, 0], df_std[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Plot the centroids as a white X
        centroids = clusterer.cluster_centers_
        ax2.scatter(centroids[:, 2], centroids[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
        # Add cluster number near the centroids
        for i, centroid in enumerate(centroids):
            ax2.scatter(centroid[2], centroid[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')

        plt.savefig(f'{graficos_dir}/{nome_prefixo}_silhouette_{n_clusters}.png')
        plt.close()


def perform_and_plot_kmeans(dataframe, nome_prefixo, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)

    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(reduced_data)

    # Define a reasonable clipping value to limit plot ranges
    clip_value = 10  # Adjust based on your specific dataset characteristics

    # Set up plot limits more safely
    x_min, x_max = max(reduced_data[:, 0].min() - 1, -clip_value), min(reduced_data[:, 0].max() + 1, clip_value)
    y_min, y_max = max(reduced_data[:, 1].min() - 1, -clip_value), min(reduced_data[:, 1].max() + 1, clip_value)
    h = 0.5  # Adjust the step size for practicality and performance
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict cluster indexes for each point in the mesh
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(12, 7))
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
    plt.title(f'K-means clustering on PCA-reduced data with {n_clusters} clusters')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.savefig(f'{graficos_dir}/{nome_prefixo}_kmeans_pca_plot.png')
    plt.close()

def pretty_print(df):
    return display( HTML( df.to_html().replace("\\n","<br>") ) )

def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
  inner_tree: _tree.Tree = tree.tree_
  classes = tree.classes_
  class_rules_dict = dict()

  def tree_dfs(node_id=0, current_rule=[]):
    # feature[i] holds the feature to split on, for the internal node i.
    split_feature = inner_tree.feature[node_id]
    if split_feature != _tree.TREE_UNDEFINED: # internal node
      name = feature_names[split_feature]
      threshold = inner_tree.threshold[node_id]
      # left child
      left_rule = current_rule + ["({} <= {})".format(name, threshold)]
      tree_dfs(inner_tree.children_left[node_id], left_rule)
      # right child
      right_rule = current_rule + ["({} > {})".format(name, threshold)]
      tree_dfs(inner_tree.children_right[node_id], right_rule)
    else: # leaf
      dist = inner_tree.value[node_id][0]
      dist = dist/dist.sum()
      max_idx = dist.argmax()
      if len(current_rule) == 0:
        rule_string = "ALL"
      else:
        rule_string = " and ".join(current_rule)
      # register new rule to dictionary
      selected_class = classes[max_idx]
      class_probability = dist[max_idx]
      class_rules = class_rules_dict.get(selected_class, [])
      class_rules.append((rule_string, class_probability))
      class_rules_dict[selected_class] = class_rules

  tree_dfs() # start from root, node_id = 0
  return class_rules_dict

def cluster_report(data: pd.DataFrame, clusters, min_samples_leaf=50, pruning_level=0.01):
    # Create Model
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level)
    tree.fit(data, clusters)

    # Generate Report
    feature_names = data.columns
    class_rule_dict = get_class_rules(tree, feature_names)

    report_class_list = []
    for class_name in class_rule_dict.keys():
        rule_list = class_rule_dict[class_name]
        combined_string = ""
        for rule in rule_list:
            combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
        report_class_list.append((class_name, combined_string))

    cluster_instance_df = pd.Series(clusters).value_counts().reset_index()
    cluster_instance_df.columns = ['class_name', 'instance_count']
    report_df = pd.DataFrame(report_class_list, columns=['class_name', 'rule_list'])
    report_df = pd.merge(cluster_instance_df, report_df, on='class_name', how='left')
    pretty_print(report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']])

    return report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']]

def kmeans_elbow_viz(data, nome_prefixo):
    sum_of_squared_distance = []
    n_cluster = range(1, 11)

    for k in n_cluster:
        kmean_model = KMeans(n_clusters=k)
        kmean_model.fit(data)
        sum_of_squared_distance.append(kmean_model.inertia_)

    plt.plot(n_cluster, sum_of_squared_distance, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow method for optimal K')
    plt.savefig(f'{graficos_dir}/{nome_prefixo}_kmeans_elbow.png')
    plt.close()  # Fechando a figura para evitar a exibição indesejada no HTML

def kmeans_scatterplot(data, nome_prefixo, n_clusters, **kwargs):
    # Preprocessing and dimension reduction
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('dim_reduction', PCA(n_components=2, random_state=0))
    ])
    
    # Transform data
    pc = pipeline.fit_transform(data)
    
    # Clustering
    kmeans_model = KMeans(n_clusters, **kwargs)
    y_cluster = kmeans_model.fit_predict(pc)

    # Create scatterplot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=pc[:,0], y=pc[:,1], hue=y_cluster, palette='bright', ax=ax)
    ax.set(xlabel="PC1", ylabel="PC2", title="KMeans Clustering - Dataset")
    ax.legend(title='Cluster')
    
    # Save plot
    plt.savefig(f'{graficos_dir}/{nome_prefixo}_kmeans_scatterplot.png')  
    plt.close()
    
    # Create a new DataFrame with clusters
    new_data = data.copy()
    new_data['Cluster'] = y_cluster
    
    return new_data
                
@analise.route('/')
def index():
    return render_template('index.html')

# Exemplo de uso da função em uma rota Flask
@analise.route('/gerar_graficos')
def gerar_graficos():
    df = get_dataframe(csv_filepath)
    df1 = get_dataframe1(csv_filepath1)
    df2 = get_dataframe2(csv_filepath2)
    
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
    perform_clustering_and_generate_graphs(df, range(2, 11), 'df')
    perform_clustering_and_generate_graphs(df1, range(2, 11), 'df1')
    perform_clustering_and_generate_graphs(df2, range(2, 11), 'df2')
    perform_and_plot_kmeans(df, 'df', 2)
    perform_and_plot_kmeans(df1,  'df1', 3)
    perform_and_plot_kmeans(df2, 'df2', 3)
    kmeans_elbow_viz(df, 'df')
    kmeans_elbow_viz(df1,  'df1')
    kmeans_elbow_viz(df2, 'df2') 
    
    return "Gráficos gerados e salvos com sucesso!"

@analise.route('/dashboard_um')
def dashboard_um():
    df = get_dataframe(csv_filepath)
    csv_filepath_old = 'C:\\Users\\juare\\Desktop\\TCC\\df.csv'
    df_old = pd.read_csv(csv_filepath_old, encoding='cp1252', delimiter=';')

    #Data outliers
    df[df.duplicated(keep='first')]
    df.drop_duplicates(keep='first',inplace=True)

    # Captura a saída de df.info()
    buffer = StringIO()
    df.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    # Captura a saída de df.info()
    buffer_old = StringIO()
    df_old.info(buf=buffer_old)
    infos_variaveis_old = buffer.getvalue()

    combinacoes = list(itertools.combinations(campos, 2))

    # Calculando correlação para todos os pares de campos
    correlacoes = {}
    for campo1 in campos:
        for campo2 in campos:
            if campo1 != campo2:
                correlacao = df.corr()[campo1][campo2]
                chave = f'{campo1} - {campo2}'
                correlacoes[chave] = correlacao

    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)
    df_std = pd.DataFrame(data = df_std,columns = df.columns)

    Soma_distancia_quadratica = []
    K = range(1,11)
    for k in K:
     km = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
     km = km.fit(df_std)
     Soma_distancia_quadratica.append(km.inertia_)

    silhouette_scores = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=10)
        labels = kmeans.fit_predict(df_std)
        score = silhouette_score(df_std, labels)
        silhouette_scores.append(score)
        print(f'For n_clusters={k}, Silhouette score is {score}')
    
    plt.figure(figsize = (10,8))
    plt.plot(range(1, 11), Soma_distancia_quadratica, marker = 'o', linestyle = '-.',color='red')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Soma_distancia_quadratica')
    plt.title('Elbow')
    caminho_arquivo = os.path.join(graficos_dir, 'df_cotovelo.png')
    plt.savefig(caminho_arquivo)

    K = range(2,11)
    for k in K:
     km = KMeans(n_clusters=k)
     km = km.fit(df_std)
     s_score=metrics.silhouette_score(df_std, km.labels_, metric='euclidean',sample_size=24527)

    kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
    kmeans.fit(df_std)
    df_segm_kmeans= df_std.copy()
    df_std['Segment'] = kmeans.labels_
    df_segm_analysis = df_std.groupby(['Segment']).mean()
    df_to_dict = df_segm_analysis.to_dict()

    new_data = kmeans_scatterplot(df, 'df', 2)
    html_data = new_data.head().to_html(classes='table')
 
    X = new_data.iloc[:,0:17]
    y = new_data.iloc[:,18]
    uniformiza = MinMaxScaler()
    novo_X = uniformiza.fit_transform(X)
    tree = DecisionTreeClassifier()
    tree_para = {'criterion':['entropy','gini'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],'min_samples_leaf':[1,2,3,4,5]}
    grid = GridSearchCV(tree, tree_para,verbose=5, cv=10)
    grid.fit(novo_X,y)
    best_clf = grid.best_estimator_
    best = best_clf
    
    # Criação de conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(novo_X,y,test_size=0.3,random_state=100)
    train = (X_train.shape, y_train.shape) # shape - mostra quantas linhas e colunas foram geradas
    test = (X_test.shape, y_test.shape)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=40, min_samples_leaf=3)
    tree.fit(X_train,y_train)
    predictions_test = tree.predict(X_test)
    accuracy_test = accuracy_score(y_test,predictions_test)*100
    report_test = classification_report(y_test,predictions_test)
    
    #Test
    cf = confusion_matrix(y_test,predictions_test)
    lbl1 = ['high', 'medium', 'low']
    lbl2 = ['high', 'medium', 'low']
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Test")
    plt.savefig(f'static/graficos/df_confusion_matrix_test.png')  # Salvando o gráfico
    plt.close()

    #Cross Validation
    predictions_test = cross_val_predict(tree,novo_X,y,cv=10)
    accuracy_test1 = accuracy_score(y,predictions_test)*100
    
    predictions = cross_val_predict (tree,novo_X,y,cv=10)
    cf = confusion_matrix(y,predictions)
    lbl1 = ['high', 'medium', 'low']
    lbl2 = ['high', 'medium', 'low']
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Cross Validation")
    plt.savefig(f'static/graficos/df_confusion_matrix_cv.png')  # Salvando o gráfico
    plt.close()

    report = classification_report(y,predictions)
    
    #Gera arvore de decisao
    plt.figure(figsize=(100, 100))
    plot_tree(tree, filled=True, fontsize=7)
    plt.title("Decision Tree")
    plt.savefig(f'static/graficos/df_decision_tree.png')  # Salvando o gráfico
    plt.close()

    # Lista para armazenar os caminhos dos gráficos
    caminhos_graficos = [f'graficos/df_{campo}.png' for campo in campos]
    caminhos_graficos1 = [f'graficos/df_{campo}_boxplot.png' for campo in campos]
    caminhos_graficos4 = [f'graficos/df_pairplot.png']
    caminhos_graficos7 = [f'graficos/df_{campo1}_{campo2}_scatterplot.png' for campo1, campo2 in combinacoes]
    caminhos_graficos10 = [f'graficos/df_heatmap.png']
    caminhos_graficos11 = [f'graficos/df_pairplot_numerical.png']
    caminhos_graficos16 = [f'graficos/df_cotovelo.png']
    caminhos_graficos19 = [f'graficos/df_confusion_matrix_test.png']
    caminhos_graficos20 = [f'graficos/df_confusion_matrix_cv.png']
    caminhos_graficos21 = [f'graficos/df_decision_tree.png']

    dados_texto = {
        'colunas_old': df_old.columns.tolist(),
        'dados_originais_old': df_old.head(5).to_html(classes='table'),
        'infos_variaveis_old': infos_variaveis_old,
        'shape_old': df_old.shape,
        'describe_old': df_old.describe().to_html(classes='table'),
        'limpeza_old': df_old.isnull().sum(),
        #'describe_include0': df.describe(include='O').to_html(classes='table'),
        'limpeza': df.isnull().sum(),
        'colunas': df.columns.tolist(),
        'dados_novos': df.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df.shape,
        'describe': df.describe().to_html(classes='table'),
        #'describe_include0': df.describe(include='O').to_html(classes='table'),
        'limpeza': df.isnull().sum(),
        'correlacoes': correlacoes,
        'soma_quadratica': Soma_distancia_quadratica,
        'df_segm_analysis': df_to_dict,
        'best': best,
        'train': train,
        'test': test,
        'accuracy_test': accuracy_test,
        'report_test': report_test,
        'accuracy_test1': accuracy_test1,
        'report': report
    }

    # Perform clustering using KMeans
    kmeans = KMeans(n_clusters=6, random_state=42)
    cluster_labels = kmeans.fit_predict(df)
    cluster_analysis_report = cluster_report(df, cluster_labels, min_samples_leaf=50, pruning_level=0.01)

    return render_template('dashboard_um.html', dados_texto=dados_texto,  
                        caminhos_graficos11=caminhos_graficos11, caminhos_graficos16=caminhos_graficos16, silhouette_scores=silhouette_scores, cluster_analysis_report=cluster_analysis_report, html_data=html_data,  caminhos_graficos=caminhos_graficos, caminhos_graficos1=caminhos_graficos1, caminhos_graficos4=caminhos_graficos4, caminhos_graficos7=caminhos_graficos7 , caminhos_graficos10=caminhos_graficos10,
                        caminhos_graficos19=caminhos_graficos19, caminhos_graficos20=caminhos_graficos20, caminhos_graficos21=caminhos_graficos21)

@analise.route('/dashboard_dois')
def dashboard_dois():
    df1 = get_dataframe1(csv_filepath1)

    csv_filepath_old1 = 'C:\\Users\\juare\\Desktop\\TCC\\df1.csv'
    df1_old = pd.read_csv(csv_filepath_old1, encoding='cp1252', delimiter=';')

    #Data outliers1
    df1[df1.duplicated(keep='first')]
    df1.drop_duplicates(keep='first',inplace=True)

    combinacoes = list(itertools.combinations(campos1, 2))

    # Captura a saída de df.info()
    buffer = StringIO()
    df1.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    buffer_old1 = StringIO()
    df1.info(buf=buffer_old1)
    infos_variaveis_old1 = buffer_old1.getvalue()

    correlacoes = {}
    for campo1 in campos1:
        for campo2 in campos1:
            if campo1 != campo2:
                correlacao = df1.corr()[campo1][campo2]
                chave = f'{campo1} - {campo2}'
                correlacoes[chave] = correlacao

   
    # Remove rows with missing values
    df1.dropna(inplace=True)       
    
    scaler = StandardScaler()
    df1_std = scaler.fit_transform(df1)
    df1_std = pd.DataFrame(data = df1_std,columns = df1.columns)

    Soma_distancia_quadratica = []
    K = range(1,11)
    for k in K:
     km = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
     km = km.fit(df1_std)
     Soma_distancia_quadratica.append(km.inertia_)

    silhouette_scores = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=10)
        labels = kmeans.fit_predict(df1_std)
        score = silhouette_score(df1_std, labels)
        silhouette_scores.append(score)
        print(f'For n_clusters={k}, Silhouette score is {score}')
     
    plt.figure(figsize = (10,8))
    plt.plot(range(1, 11), Soma_distancia_quadratica, marker = 'o', linestyle = '-.',color='red')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Soma_distancia_quadratica')
    plt.title('Elbow')
    caminho_arquivo = os.path.join(graficos_dir, 'df1_cotovelo.png')
    plt.savefig(caminho_arquivo)

    kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
    kmeans.fit(df1_std)
    df1_segm_kmeans= df1_std.copy()
    df1_std['Segment'] = kmeans.labels_
    df1_segm_analysis = df1_std.groupby(['Segment']).mean()
    df1_to_dict = df1_segm_analysis.to_dict()

    new_data = kmeans_scatterplot(df1, 'df1', 3)
    html_data = new_data.head().to_html(classes='table')

    X = new_data.iloc[:,0:18]
    y = new_data.iloc[:,19]
    uniformiza = MinMaxScaler()
    novo_X = uniformiza.fit_transform(X)
    tree = DecisionTreeClassifier()
    tree_para = {'criterion':['entropy','gini'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],'min_samples_leaf':[1,2,3,4,5]}
    grid = GridSearchCV(tree, tree_para,verbose=5, cv=10)
    grid.fit(novo_X,y)
    best_clf = grid.best_estimator_
    best = best_clf
    
    # Criação de conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(novo_X,y,test_size=0.3,random_state=100)
    train = (X_train.shape, y_train.shape) # shape - mostra quantas linhas e colunas foram geradas
    test = (X_test.shape, y_test.shape)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=70, min_samples_leaf=5)
    tree.fit(X_train,y_train)
    predictions_test = tree.predict(X_test)
    accuracy_test = accuracy_score(y_test,predictions_test)*100
    report_test = classification_report(y_test,predictions_test)
    
    #Test
    cf = confusion_matrix(y_test,predictions_test)
    lbl1 = ['high', 'medium', 'low']
    lbl2 = ['high', 'medium', 'low']
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Test")
    plt.savefig(f'static/graficos/df1_confusion_matrix_test.png')  # Salvando o gráfico
    plt.close()

    #Cross Validation
    predictions_test = cross_val_predict(tree,novo_X,y,cv=10)
    accuracy_test1 = accuracy_score(y,predictions_test)*100
    
    predictions = cross_val_predict (tree,novo_X,y,cv=10)
    cf = confusion_matrix(y,predictions)
    lbl1 = ['high', 'medium', 'low']
    lbl2 = ['high', 'medium', 'low']
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Cross Validation")
    plt.savefig(f'static/graficos/df1_confusion_matrix_cv.png')  # Salvando o gráfico
    plt.close()

    report = classification_report(y,predictions)
    
    #Gera arvore de decisao
    plt.figure(figsize=(40, 30))
    plot_tree(tree, filled=True, fontsize=7)
    plt.title("Decision Tree")
    plt.savefig(f'static/graficos/df1_decision_tree.png')  # Salvando o gráfico
    plt.close()

    dados_texto = {
        'colunas_old': df1_old.columns.tolist(),
        'dados_originais_old': df1_old.head(5).to_html(classes='table'),
        'infos_variaveis_old': infos_variaveis_old1,
        'shape_old': df1_old.shape,
        'describe_old': df1_old.describe().to_html(classes='table'),
        'limpeza_old': df1_old.isnull().sum(),
        'colunas': df1.columns.tolist(),
        'dados_originais': df1.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df1.shape,
        'describe': df1.describe().to_html(classes='table'),
        #'describe_include0': df1.describe(include='O').to_html(classes='table'),
        'limpeza': df1.isnull().sum(),
        'correlacoes': correlacoes,
        'soma_quadratica': Soma_distancia_quadratica,
        'df_segm_analysis': df1_to_dict,
        'best': best,
        'train': train,
        'test': test,
        'accuracy_test': accuracy_test,
        'report_test': report_test,
        'accuracy_test1': accuracy_test1,
        'report': report
    }

    caminhos_graficos = [f'graficos/df1_{campo1}.png' for campo1 in campos1]
    caminhos_graficos2 = [f'graficos/df1_{campo1}_boxplot.png' for campo1 in campos1]
    caminhos_graficos5 = [f'graficos/df1_pairplot.png']
    caminhos_graficos8 = [f'graficos/df1_{campo1}_{campo2}_scatterplot.png' for campo1, campo2 in combinacoes]
    caminhos_graficos12 = [f'graficos/df1_heatmap.png']
    caminhos_graficos13 = [f'graficos/df1_pairplot_numerical.png']
    caminhos_graficos17 = [f'graficos/df1_cotovelo.png']
    caminhos_graficos22 = [f'graficos/df1_confusion_matrix_test.png']
    caminhos_graficos23 = [f'graficos/df1_confusion_matrix_cv.png']
    caminhos_graficos24 = [f'graficos/df1_decision_tree.png']

   
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(df1)
    cluster_analysis_report1 = cluster_report(df1, cluster_labels, min_samples_leaf=50, pruning_level=0.01)

    return render_template('dashboard_dois.html', dados_texto=dados_texto,  caminhos_graficos=caminhos_graficos, caminhos_graficos2=caminhos_graficos2, caminhos_graficos5=caminhos_graficos5, caminhos_graficos8=caminhos_graficos8, caminhos_graficos12=caminhos_graficos12,
                           caminhos_graficos13=caminhos_graficos13, caminhos_graficos17=caminhos_graficos17, silhouette_scores=silhouette_scores,  cluster_analysis_report1=cluster_analysis_report1, html_data=html_data, caminhos_graficos22=caminhos_graficos22, caminhos_graficos23=caminhos_graficos23, caminhos_graficos24=caminhos_graficos24)

@analise.route('/dashboard_tres')
def dashboard_tres():
    df2 = get_dataframe2(csv_filepath2)

    csv_filepath_old2 = 'C:\\Users\\juare\\Desktop\\TCC\\df2.csv'
    df2_old = pd.read_csv(csv_filepath_old2, encoding='cp1252', delimiter=';')

    #Data outliers1
    df2[df2.duplicated(keep='first')]
    df2.drop_duplicates(keep='first',inplace=True)

    combinacoes = list(itertools.combinations(campos2, 2))

    # Captura a saída de df.info()
    buffer = StringIO()
    df2.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    buffer_old2 = StringIO()
    df2.info(buf=buffer_old2)
    infos_variaveis_old2 = buffer_old2.getvalue()

    correlacoes = {}
    for campo1 in campos2:
        for campo2 in campos2:
            if campo1 != campo2:
                correlacao = df2.corr()[campo1][campo2]
                chave = f'{campo1} - {campo2}'
                correlacoes[chave] = correlacao

    # Remove rows with missing values
    df2.dropna(inplace=True)       

    scaler = StandardScaler()
    df2_std = scaler.fit_transform(df2)
    df2_std = pd.DataFrame(data = df2_std,columns = df2.columns)

    Soma_distancia_quadratica = []
    K = range(1,11)
    for k in K:
     km = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
     km = km.fit(df2_std)
     Soma_distancia_quadratica.append(km.inertia_)

    silhouette_scores = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=10)
        labels = kmeans.fit_predict(df2_std)
        score = silhouette_score(df2_std, labels)
        silhouette_scores.append(score)
        print(f'For n_clusters={k}, Silhouette score is {score}')
     
    plt.figure(figsize = (10,8))
    plt.plot(range(1, 11), Soma_distancia_quadratica, marker = 'o', linestyle = '-.',color='red')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Soma_distancia_quadratica')
    plt.title('Elbow')
    caminho_arquivo = os.path.join(graficos_dir, 'df2_cotovelo.png')
    plt.savefig(caminho_arquivo)

    kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
    kmeans.fit(df2_std)
    df2_segm_kmeans= df2_std.copy()
    df2_std['Segment'] = kmeans.labels_
    df2_segm_analysis = df2_std.groupby(['Segment']).mean()
    df2_to_dict = df2_segm_analysis.to_dict()
    
    new_data = kmeans_scatterplot(df2, 'df2', 3)
    html_data = new_data.head().to_html(classes='table')

    X = new_data.iloc[:,0:21]
    y = new_data.iloc[:,22]
    uniformiza = MinMaxScaler()
    novo_X = uniformiza.fit_transform(X)
    tree = DecisionTreeClassifier()
    tree_para = {'criterion':['entropy','gini'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],'min_samples_leaf':[1,2,3,4,5]}
    grid = GridSearchCV(tree, tree_para,verbose=5, cv=10)
    grid.fit(novo_X,y)
    best_clf = grid.best_estimator_
    best = best_clf
    
    # Criação de conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(novo_X,y,test_size=0.3,random_state=100)
    train = (X_train.shape, y_train.shape) # shape - mostra quantas linhas e colunas foram geradas
    test = (X_test.shape, y_test.shape)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=5)
    tree.fit(X_train,y_train)
    predictions_test = tree.predict(X_test)
    accuracy_test = accuracy_score(y_test,predictions_test)*100
    report_test = classification_report(y_test,predictions_test)
    
    #Test
    cf = confusion_matrix(y_test,predictions_test)
    lbl1 = ['high', 'medium', 'low']
    lbl2 = ['high', 'medium', 'low']
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Test")
    plt.savefig(f'static/graficos/df2_confusion_matrix_test.png')  # Salvando o gráfico
    plt.close()

    #Cross Validation
    predictions_test = cross_val_predict(tree,novo_X,y,cv=10)
    accuracy_test1 = accuracy_score(y,predictions_test)*100
    
    predictions = cross_val_predict (tree,novo_X,y,cv=10)
    cf = confusion_matrix(y,predictions)
    lbl1 = ['high', 'medium', 'low']
    lbl2 = ['high', 'medium', 'low']
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Cross Validation")
    plt.savefig(f'static/graficos/df2_confusion_matrix_cv.png')  # Salvando o gráfico
    plt.close()

    report = classification_report(y,predictions)
    
    #Gera arvore de decisao
    plt.figure(figsize=(70, 60))
    plot_tree(tree, filled=True, fontsize=7)
    plt.title("Decision Tree")
    plt.savefig(f'static/graficos/df2_decision_tree.png')  # Salvando o gráfico
    plt.close()

    dados_texto = {
        'colunas_old': df2_old.columns.tolist(),
        'dados_originais_old': df2_old.head(5).to_html(classes='table'),
        'infos_variaveis_old': infos_variaveis_old2,
        'shape_old': df2_old.shape,
        'describe_old': df2_old.describe().to_html(classes='table'),
        'limpeza_old': df2_old.isnull().sum(),
        'colunas': df2.columns.tolist(),
        'dados_originais': df2.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df2.shape,
        'describe': df2.describe().to_html(classes='table'),
       # 'describe_include0': df2.describe(include='O').to_html(classes='table'),
        'limpeza': df2.isnull().sum(),
        'correlacoes': correlacoes,
        'soma_quadratica': Soma_distancia_quadratica,
        'df_segm_analysis': df2_to_dict,
        'best': best,
        'train': train,
        'test': test,
        'accuracy_test': accuracy_test,
        'report_test': report_test,
        'accuracy_test1': accuracy_test1,
        'report': report
    }
       
    caminhos_graficos = [f'graficos/df2_{campo2}.png' for campo2 in campos2]
    caminhos_graficos3 = [f'graficos/df2_{campo2}_boxplot.png' for campo2 in campos2]
    caminhos_graficos6 = [f'graficos/df2_pairplot.png']
    caminhos_graficos9 = [f'graficos/df2_{campo1}_{campo2}_scatterplot.png' for campo1, campo2 in combinacoes]
    caminhos_graficos14 = [f'graficos/df2_heatmap.png']
    caminhos_graficos15 = [f'graficos/df2_pairplot_numerical.png']
    caminhos_graficos18 = [f'graficos/df2_cotovelo.png']
    caminhos_graficos25 = [f'graficos/df2_confusion_matrix_test.png']
    caminhos_graficos26 = [f'graficos/df2_confusion_matrix_cv.png']
    caminhos_graficos27 = [f'graficos/df2_decision_tree.png']

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(df2)
    cluster_analysis_report2 = cluster_report(df2, cluster_labels, min_samples_leaf=50, pruning_level=0.01)

    return render_template('dashboard_tres.html', dados_texto=dados_texto,  caminhos_graficos=caminhos_graficos, caminhos_graficos3=caminhos_graficos3, caminhos_graficos6=caminhos_graficos6, caminhos_graficos9=caminhos_graficos9, caminhos_graficos14=caminhos_graficos14,
                           caminhos_graficos15=caminhos_graficos15,  caminhos_graficos18=caminhos_graficos18, silhouette_scores=silhouette_scores, cluster_analysis_report2=cluster_analysis_report2, html_data=html_data, caminhos_graficos25=caminhos_graficos25, caminhos_graficos26=caminhos_graficos26, caminhos_graficos27=caminhos_graficos27)

# Rota para exibir um gráfico específico
@analise.route('/grafico/<campo>')
def mostrar_grafico(campo):
    caminho_arquivo = os.path.join(graficos_dir, f'{campo}.png')
    return send_file(caminho_arquivo, mimetype='image/png')

if __name__ == '__main__':
    analise.run(debug=True)
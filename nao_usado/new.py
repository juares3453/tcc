from flask import Flask, render_template, send_file
import matplotlib as mpl
import os
from io import StringIO
import itertools
from tqdm import tqdm
import pandas as pd
import seaborn as sb
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

campos1 = ['Dia', 'Mes', 'Ano', 'DsTpVeiculo', 'VlCusto', 'km_rodado', 'VlCapacVeic',
           'NrAuxiliares', '%CapacidadeCarre', '%CapacidadeEntr', '%Entregas', '%VolumesEntr', '%PesoEntr', '%FreteCobrado', 'FreteEx',
           'Lucro', '%Lucro']

campos2 = ['Resp', 'CLIENTE', 'dtcte','mescte','anocte','dtemissao','mesemissao','anoemissao','dtocor','mesocor','anoocor','dtbaixa','mesbaixa',
           'anobaixa','diasemissao','diasresolucao','DsLocal', 'tp_ocor', 'Situacao','NrBo','dsocorrencia','VlCusto']

csv_filepath = 'C:\\Users\\babid\\TCC\\df.csv'
csv_filepath1 = 'C:\\Users\\babid\\TCC\\df1.csv'
csv_filepath2 = 'C:\\Users\\babid\\TCC\\df2.csv'

def remover_valores_negativos(df):
    for coluna in df.columns:
        if pd.api.types.is_numeric_dtype(df[coluna]):
            df[coluna] = df[coluna].apply(lambda x: x if x >= 0 else np.nan)
    return df

def get_dataframe(csv_filepath):
    df = pd.read_csv(csv_filepath, encoding='cp1252', delimiter=';')
    df_new = df.drop(['conf_carregamento', 'conf_entrega'], axis=1)
    df_new = remover_valores_negativos(df_new)
    df_new.dropna(inplace=True)
    df_new['Filial'] = pd.factorize(df_new['Filial'])[0]
    return df_new

def get_dataframe1(csv_filepath1):
    df1 = pd.read_csv(csv_filepath1, encoding='cp1252', delimiter=';')
    df1_new = df1.drop(['DsModelo', 'DsAnoFabricacao'], axis=1)
    df1_new = remover_valores_negativos(df1_new)
    df1_new.dropna(inplace=True)
    df1_new['DsTpVeiculo'] = pd.factorize(df1_new['DsTpVeiculo'])[0]
    df1_new['VlCusto'] = df1_new['VlCusto'].str.replace(',', '.').astype(float)
    df1_new['Lucro'] = df1_new['Lucro'].str.replace(',', '.').astype(float)
    return df1_new

def get_dataframe2(csv_filepath2):
    df2 = pd.read_csv(csv_filepath2, encoding='cp1252', delimiter=';')
    df2 = remover_valores_negativos(df2)
    df2.dropna(inplace=True)
    df2['DsLocal'] = pd.factorize(df2['DsLocal'])[0]
    df2['tp_ocor'] = pd.factorize(df2['tp_ocor'])[0]
    df2['Situacao'] = pd.factorize(df2['Situacao'])[0]
    df2['dsocorrencia'] = pd.factorize(df2['dsocorrencia'])[0]
    df2['CLIENTE'] = pd.factorize(df2['CLIENTE'])[0]
    df2['VlCusto'] = df2['VlCusto'].str.replace(',', '.').astype(float)
    return df2

def gerar_e_salvar_graficos(df, campos, nome_prefixo):
    with tqdm(total=len(campos), desc="Gerando gráficos parte 1") as pbar:
        for campo in campos:
            with plt.rc_context(rc={'figure.max_open_warning': 0}):
                try:
                    plt.figure(figsize=(10, 6))
                    if campo in ['peso_total', 'peso_entregue', 'frete_total', 'frete_entregue']:
                        plt.hist(df[campo], bins=30, color='blue', alpha=0.7)
                        plt.ylabel('Contagem', size=14)
                        plt.title(f'Histograma (plt.hist) de {campo}', size=18)
                    elif df[campo].dtype in ['int64', 'float64']:
                        sb.histplot(df[campo], kde=True, color='green')
                        plt.ylabel('Densidade', size=14)
                        plt.title(f'Histograma (sns.histplot) de {campo}', size=18)
                    else:
                        sb.countplot(x=campo, data=df)
                        plt.ylabel('Contagem', size=14)
                        plt.title(f'Distribuição de {campo}', size=18)

                    plt.xlabel(campo, size=14)
                    plt.xticks(rotation=45)
                    caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_{campo}.png')
                    plt.savefig(caminho_arquivo)
                    pbar.update(1)
                finally:
                    plt.close('all')

def gerar_e_salvar_graficos_pairplot(df, campos, nome_prefixo):
    with tqdm(total=1, desc="Gerando gráficos parte 2") as pbar:
        with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
                df_numeric = df[campos]
                sb.pairplot(df_numeric)
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_pairplot.png')
                plt.savefig(caminho_arquivo)
                pbar.update(1)
            finally:
                plt.close('all')

def gerar_e_salvar_graficos_heatmap(df, nome_prefixo):
    with tqdm(total=1, desc="Gerando gráficos parte 3") as pbar:
        with plt.rc_context(rc={'figure.max_open_warning': 0}):
            try:
                plt.figure(figsize=(20, 20))
                sb.heatmap(df.corr(), annot=True, square=True, cmap='RdBu', vmax=1, vmin=-1)
                plt.xticks(size=13)
                plt.yticks(size=13)
                plt.yticks(rotation=0)
                caminho_arquivo = os.path.join(graficos_dir, f'{nome_prefixo}_heatmap.png')
                plt.savefig(caminho_arquivo)
                pbar.update(1)
            finally:
                plt.close('all')

def perform_clustering_and_generate_graphs(df, n_clusters_range, nome_prefixo):
    df_std = StandardScaler().fit_transform(df)
    for n_clusters in n_clusters_range:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(df_std)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        silhouette_avg = silhouette_score(df_std, cluster_labels)
        sample_silhouette_values = silhouette_samples(df_std, cluster_labels)

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
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), color="red", fontweight='bold')
            y_lower = y_upper + 10

        ax1.set_title("The silhouette plot for various clusters")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(df_std[:, 0], df_std[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        centroids = clusterer.cluster_centers_
        ax2.scatter(centroids[:, 2], centroids[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
        for i, centroid in enumerate(centroids):
            ax2.scatter(centroid[2], centroid[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')

        plt.savefig(f'{graficos_dir}/{nome_prefixo}_silhouette_{n_clusters}.png')
        plt.close()

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
    plt.close()

def kmeans_scatterplot(data, nome_prefixo, n_clusters, **kwargs):
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('dim_reduction', PCA(n_components=2, random_state=0))
    ])
    pc = pipeline.fit_transform(data)
    kmeans_model = KMeans(n_clusters, **kwargs)
    y_cluster = kmeans_model.fit_predict(pc)

    fig, ax = plt.subplots(figsize=(8, 6))
    sb.scatterplot(x=pc[:,0], y=pc[:,1], hue=y_cluster, palette='bright', ax=ax)
    ax.set(xlabel="PC1", ylabel="PC2", title="KMeans Clustering - Dataset")
    ax.legend(title='Cluster')

    plt.savefig(f'{graficos_dir}/{nome_prefixo}_kmeans_scatterplot.png')  
    plt.close()

    new_data = data.copy()
    new_data['Cluster'] = y_cluster
    return new_data

@analise.route('/')
def index():
    return render_template('index.html')

@analise.route('/gerar_graficos')
def gerar_graficos():
    df = get_dataframe(csv_filepath)
    df1 = get_dataframe1(csv_filepath1)
    df2 = get_dataframe2(csv_filepath2)
    
    gerar_e_salvar_graficos(df, campos, 'df')
    gerar_e_salvar_graficos(df1, campos1, 'df1')
    gerar_e_salvar_graficos(df2, campos2, 'df2')
    gerar_e_salvar_graficos_pairplot(df, campos, 'df')
    gerar_e_salvar_graficos_pairplot(df1, campos1, 'df1')
    gerar_e_salvar_graficos_pairplot(df2, campos2, 'df2')
    gerar_e_salvar_graficos_heatmap(df, 'df')
    gerar_e_salvar_graficos_heatmap(df1, 'df1')
    gerar_e_salvar_graficos_heatmap(df2, 'df2')
    perform_clustering_and_generate_graphs(df, range(2, 11), 'df')
    perform_clustering_and_generate_graphs(df1, range(2, 11), 'df1')
    perform_clustering_and_generate_graphs(df2, range(2, 11), 'df2')
    kmeans_elbow_viz(df, 'df')
    kmeans_elbow_viz(df1, 'df1')
    kmeans_elbow_viz(df2, 'df2') 

    return "Gráficos gerados e salvos com sucesso!"

@analise.route('/dashboard_um')
def dashboard_um():
    df = get_dataframe(csv_filepath)
    df_old = pd.read_csv(csv_filepath, encoding='cp1252', delimiter=';')

    df.drop_duplicates(keep='first', inplace=True)
    combinacoes = list(itertools.combinations(campos, 2))

    buffer = StringIO()
    df.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    buffer_old = StringIO()
    df_old.info(buf=buffer_old)
    infos_variaveis_old = buffer_old.getvalue()

    correlacoes = {}
    for campo1 in campos:
        for campo2 in campos:
            if campo1 != campo2:
                correlacao = df.corr()[campo1][campo2]
                chave = f'{campo1} - {campo2}'
                correlacoes[chave] = correlacao

    df.dropna(inplace=True)       
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)
    df_std = pd.DataFrame(data=df_std, columns=df.columns)

    Soma_distancia_quadratica = []
    K = range(1,11)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42)
        km = km.fit(df_std)
        Soma_distancia_quadratica.append(km.inertia_)

    silhouette_scores = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=10)
        labels = kmeans.fit_predict(df_std)
        score = silhouette_score(df_std, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 11), Soma_distancia_quadratica, marker='o', linestyle='-.', color='red')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Soma_distancia_quadratica')
    plt.title('Elbow')
    caminho_arquivo = os.path.join(graficos_dir, 'df_cotovelo.png')
    plt.savefig(caminho_arquivo)

    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmeans.fit(df_std)
    df['Segment'] = kmeans.labels_
    df_segm_analysis = df.groupby(['Segment']).mean()

    new_data = kmeans_scatterplot(df, 'df', 3)
    html_data = new_data.head().to_html(classes='table')

    X = new_data.iloc[:, :-1]
    y = new_data.iloc[:, -1]
    uniformiza = MinMaxScaler()
    novo_X = uniformiza.fit_transform(X)
    tree = DecisionTreeClassifier()
    tree_para = {'criterion': ['entropy', 'gini'], 'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150], 'min_samples_leaf': [1, 2, 3, 4, 5]}
    grid = GridSearchCV(tree, tree_para, verbose=5, cv=10)
    grid.fit(novo_X, y)
    best_clf = grid.best_estimator_
    best = best_clf
    
    X_train, X_test, y_train, y_test = train_test_split(novo_X, y, test_size=0.3, random_state=100)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=40, min_samples_leaf=3)
    tree.fit(X_train, y_train)
    predictions_test = tree.predict(X_test)
    accuracy_test = accuracy_score(y_test, predictions_test) * 100
    report_test = classification_report(y_test, predictions_test)
    
    cf = confusion_matrix(y_test, predictions_test)
    lbl1 = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    lbl2 = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Test")
    plt.savefig(f'static/graficos/df_confusion_matrix_test.png')
    plt.close()

    predictions = cross_val_predict(tree, novo_X, y, cv=10)
    cf = confusion_matrix(y, predictions)
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Cross Validation")
    plt.savefig(f'static/graficos/df_confusion_matrix_cv.png')
    plt.close()

    report = classification_report(y, predictions)
    
    plt.figure(figsize=(100, 100))
    plot_tree(tree, filled=True, fontsize=7)
    plt.title("Decision Tree")
    plt.savefig(f'static/graficos/df_decision_tree.png')
    plt.close()

    dados_texto = {
        'colunas_old': df_old.columns.tolist(),
        'dados_originais_old': df_old.head(5).to_html(classes='table'),
        'infos_variaveis_old': infos_variaveis_old,
        'shape_old': df_old.shape,
        'describe_old': df_old.describe().to_html(classes='table'),
        'limpeza_old': df_old.isnull().sum(),
        'colunas': df.columns.tolist(),
        'dados_novos': df.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df.shape,
        'describe': df.describe().to_html(classes='table'),
        'limpeza': df.isnull().sum(),
        'correlacoes': correlacoes,
        'soma_quadratica': Soma_distancia_quadratica,
        'df_segm_analysis': df_segm_analysis.to_dict(),
        'best': best,
        'train': (X_train.shape, y_train.shape),
        'test': (X_test.shape, y_test.shape),
        'accuracy_test': accuracy_test,
        'report_test': report_test,
        'report': report
    }

    kmeans = KMeans(n_clusters=6, random_state=42)
    cluster_labels = kmeans.fit_predict(df)
    cluster_analysis_report = cluster_report(df, cluster_labels, min_samples_leaf=50, pruning_level=0.01)

    return render_template('dashboard_um.html', dados_texto=dados_texto, 
                           silhouette_scores=silhouette_scores, cluster_analysis_report=cluster_analysis_report, 
                           html_data=html_data)

@analise.route('/dashboard_dois')
def dashboard_dois():
    df1 = get_dataframe1(csv_filepath1)
    df1_old = pd.read_csv(csv_filepath1, encoding='cp1252', delimiter=';')

    df1.drop_duplicates(keep='first', inplace=True)
    combinacoes = list(itertools.combinations(campos1, 2))

    buffer = StringIO()
    df1.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    buffer_old1 = StringIO()
    df1_old.info(buf=buffer_old1)
    infos_variaveis_old1 = buffer_old1.getvalue()

    correlacoes = {}
    for campo1 in campos1:
        for campo2 in campos1:
            if campo1 != campo2:
                correlacao = df1.corr()[campo1][campo2]
                chave = f'{campo1} - {campo2}'
                correlacoes[chave] = correlacao

    df1.dropna(inplace=True)       
    scaler = StandardScaler()
    df1_std = scaler.fit_transform(df1)
    df1_std = pd.DataFrame(data=df1_std, columns=df1.columns)

    Soma_distancia_quadratica = []
    K = range(1,11)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42)
        km = km.fit(df1_std)
        Soma_distancia_quadratica.append(km.inertia_)

    silhouette_scores = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=10)
        labels = kmeans.fit_predict(df1_std)
        score = silhouette_score(df1_std, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 11), Soma_distancia_quadratica, marker='o', linestyle='-.', color='red')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Soma_distancia_quadratica')
    plt.title('Elbow')
    caminho_arquivo = os.path.join(graficos_dir, 'df1_cotovelo.png')
    plt.savefig(caminho_arquivo)

    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmeans.fit(df1_std)
    df1['Segment'] = kmeans.labels_
    df1_segm_analysis = df1.groupby(['Segment']).mean()

    new_data = kmeans_scatterplot(df1, 'df1', 3)
    html_data = new_data.head().to_html(classes='table')

    X = new_data.iloc[:, :-1]
    y = new_data.iloc[:, -1]
    uniformiza = MinMaxScaler()
    novo_X = uniformiza.fit_transform(X)
    tree = DecisionTreeClassifier()
    tree_para = {'criterion': ['entropy', 'gini'], 'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150], 'min_samples_leaf': [1, 2, 3, 4, 5]}
    grid = GridSearchCV(tree, tree_para, verbose=5, cv=10)
    grid.fit(novo_X, y)
    best_clf = grid.best_estimator_
    best = best_clf
    
    X_train, X_test, y_train, y_test = train_test_split(novo_X, y, test_size=0.3, random_state=100)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=40, min_samples_leaf=3)
    tree.fit(X_train, y_train)
    predictions_test = tree.predict(X_test)
    accuracy_test = accuracy_score(y_test, predictions_test) * 100
    report_test = classification_report(y_test, predictions_test)
    
    cf = confusion_matrix(y_test, predictions_test)
    lbl1 = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    lbl2 = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Test")
    plt.savefig(f'static/graficos/df1_confusion_matrix_test.png')
    plt.close()

    predictions = cross_val_predict(tree, novo_X, y, cv=10)
    cf = confusion_matrix(y, predictions)
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Cross Validation")
    plt.savefig(f'static/graficos/df1_confusion_matrix_cv.png')
    plt.close()

    report = classification_report(y, predictions)
    
    plt.figure(figsize=(100, 100))
    plot_tree(tree, filled=True, fontsize=7)
    plt.title("Decision Tree")
    plt.savefig(f'static/graficos/df1_decision_tree.png')
    plt.close()

    dados_texto = {
        'colunas_old': df1_old.columns.tolist(),
        'dados_originais_old': df1_old.head(5).to_html(classes='table'),
        'infos_variaveis_old': infos_variaveis_old1,
        'shape_old': df1_old.shape,
        'describe_old': df1_old.describe().to_html(classes='table'),
        'limpeza_old': df1_old.isnull().sum(),
        'colunas': df1.columns.tolist(),
        'dados_novos': df1.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df1.shape,
        'describe': df1.describe().to_html(classes='table'),
        'limpeza': df1.isnull().sum(),
        'correlacoes': correlacoes,
        'soma_quadratica': Soma_distancia_quadratica,
        'df_segm_analysis': df1_segm_analysis.to_dict(),
        'best': best,
        'train': (X_train.shape, y_train.shape),
        'test': (X_test.shape, y_test.shape),
        'accuracy_test': accuracy_test,
        'report_test': report_test,
        'report': report
    }

    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(df1)
    cluster_analysis_report = cluster_report(df1, cluster_labels, min_samples_leaf=50, pruning_level=0.01)

    return render_template('dashboard_dois.html', dados_texto=dados_texto, 
                           silhouette_scores=silhouette_scores, cluster_analysis_report=cluster_analysis_report, 
                           html_data=html_data)

@analise.route('/dashboard_tres')
def dashboard_tres():
    df2 = get_dataframe2(csv_filepath2)
    df2_old = pd.read_csv(csv_filepath2, encoding='cp1252', delimiter=';')

    df2.drop_duplicates(keep='first', inplace=True)
    combinacoes = list(itertools.combinations(campos2, 2))

    buffer = StringIO()
    df2.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    buffer_old2 = StringIO()
    df2_old.info(buf=buffer_old2)
    infos_variaveis_old2 = buffer_old2.getvalue()

    correlacoes = {}
    for campo1 in campos2:
        for campo2 in campos2:
            if campo1 != campo2:
                correlacao = df2.corr()[campo1][campo2]
                chave = f'{campo1} - {campo2}'
                correlacoes[chave] = correlacao

    df2.dropna(inplace=True)       
    scaler = StandardScaler()
    df2_std = scaler.fit_transform(df2)
    df2_std = pd.DataFrame(data=df2_std, columns=df2.columns)

    Soma_distancia_quadratica = []
    K = range(1,11)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42)
        km = km.fit(df2_std)
        Soma_distancia_quadratica.append(km.inertia_)

    silhouette_scores = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=10)
        labels = kmeans.fit_predict(df2_std)
        score = silhouette_score(df2_std, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 11), Soma_distancia_quadratica, marker='o', linestyle='-.', color='red')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Soma_distancia_quadratica')
    plt.title('Elbow')
    caminho_arquivo = os.path.join(graficos_dir, 'df2_cotovelo.png')
    plt.savefig(caminho_arquivo)

    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmeans.fit(df2_std)
    df2['Segment'] = kmeans.labels_
    df2_segm_analysis = df2.groupby(['Segment']).mean()

    new_data = kmeans_scatterplot(df2, 'df2', 3)
    html_data = new_data.head().to_html(classes='table')

    X = new_data.iloc[:, :-1]
    y = new_data.iloc[:, -1]
    uniformiza = MinMaxScaler()
    novo_X = uniformiza.fit_transform(X)
    tree = DecisionTreeClassifier()
    tree_para = {'criterion': ['entropy', 'gini'], 'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150], 'min_samples_leaf': [1, 2, 3, 4, 5]}
    grid = GridSearchCV(tree, tree_para, verbose=5, cv=10)
    grid.fit(novo_X, y)
    best_clf = grid.best_estimator_
    best = best_clf
    
    X_train, X_test, y_train, y_test = train_test_split(novo_X, y, test_size=0.3, random_state=100)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=40, min_samples_leaf=3)
    tree.fit(X_train, y_train)
    predictions_test = tree.predict(X_test)
    accuracy_test = accuracy_score(y_test, predictions_test) * 100
    report_test = classification_report(y_test, predictions_test)
    
    cf = confusion_matrix(y_test, predictions_test)
    lbl1 = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    lbl2 = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Test")
    plt.savefig(f'static/graficos/df2_confusion_matrix_test.png')
    plt.close()

    predictions = cross_val_predict(tree, novo_X, y, cv=10)
    cf = confusion_matrix(y, predictions)
    plt.figure(figsize=(10, 7))
    sb.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Cross Validation")
    plt.savefig(f'static/graficos/df2_confusion_matrix_cv.png')
    plt.close()

    report = classification_report(y, predictions)
    
    plt.figure(figsize=(100, 100))
    plot_tree(tree, filled=True, fontsize=7)
    plt.title("Decision Tree")
    plt.savefig(f'static/graficos/df2_decision_tree.png')
    plt.close()

    dados_texto = {
        'colunas_old': df2_old.columns.tolist(),
        'dados_originais_old': df2_old.head(5).to_html(classes='table'),
        'infos_variaveis_old': infos_variaveis_old2,
        'shape_old': df2_old.shape,
        'describe_old': df2_old.describe().to_html(classes='table'),
        'limpeza_old': df2_old.isnull().sum(),
        'colunas': df2.columns.tolist(),
        'dados_novos': df2.head(5).to_html(classes='table'),
        'infos_variaveis': infos_variaveis,
        'shape': df2.shape,
        'describe': df2.describe().to_html(classes='table'),
        'limpeza': df2.isnull().sum(),
        'correlacoes': correlacoes,
        'soma_quadratica': Soma_distancia_quadratica,
        'df_segm_analysis': df2_segm_analysis.to_dict(),
        'best': best,
        'train': (X_train.shape, y_train.shape),
        'test': (X_test.shape, y_test.shape),
        'accuracy_test': accuracy_test,
        'report_test': report_test,
        'report': report
    }

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(df2)
    cluster_analysis_report = cluster_report(df2, cluster_labels, min_samples_leaf=50, pruning_level=0.01)

    return render_template('dashboard_tres.html', dados_texto=dados_texto, 
                           silhouette_scores=silhouette_scores, cluster_analysis_report=cluster_analysis_report, 
                           html_data=html_data)

# Rota para exibir um gráfico específico
@analise.route('/grafico/<campo>')
def mostrar_grafico(campo):
    caminho_arquivo = os.path.join(graficos_dir, f'{campo}.png')
    return send_file(caminho_arquivo, mimetype='image/png')

if __name__ == '__main__':
    analise.run(debug=True)

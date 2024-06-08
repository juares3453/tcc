from flask import Flask, render_template, send_file
import matplotlib as mpl
import os
from io import StringIO
import itertools
from tqdm import tqdm
from time import time
import pandas as pd
import seaborn as sb
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree, _tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

mpl.use('Agg')
mpl.rcParams['figure.max_open_warning'] = 50
analise = Flask(__name__)

# Diretório para salvar os gráficos
graficos_dir = 'static/graficos'
os.makedirs(graficos_dir, exist_ok=True)


campos1 = ['Dia', 'Mes', 'Ano', 'DsTpVeiculo', 'VlCusto', 'km_rodado', 'VlCapacVeic',
       'NrAuxiliares', '%CapacidadeCarre', '%CapacidadeEntr', '%Entregas', '%VolumesEntr', '%PesoEntr', '%FreteCobrado', 'FreteEx',
       'Lucro', '%Lucro']

csv_filepath = os.path.join('df.csv')
csv_filepath1 = os.path.join('df1.csv')
csv_filepath2 = os.path.join('df2.csv')

def remover_valores_negativos(df):
    for coluna in df.columns:
        if pd.api.types.is_numeric_dtype(df[coluna]):
            df[coluna] = df[coluna].apply(lambda x: x if x >= 0 else np.nan)
    return df

def get_dataframe1(csv_filepath1):
    df1 = pd.read_csv(csv_filepath1, encoding='cp1252', delimiter=';')
    df1_new = df1.drop(['DsModelo', 'DsAnoFabricacao'], axis=1)
    df1_new = remover_valores_negativos(df1_new)
    df1_new.dropna(inplace=True)
    df1_new['DsTpVeiculo'] = pd.factorize(df1_new['DsTpVeiculo'])[0]
    df1_new['VlCusto'] = df1_new['VlCusto'].str.replace(',', '.').astype(float)
    df1_new['Lucro'] = df1_new['Lucro'].str.replace(',', '.').astype(float)
    return df1_new

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
                    sb.histplot(df1[campo1], kde=True, color='green')
                    plt.ylabel('Densidade', size=14)
                    plt.title(f'Histograma (sns.histplot) de {campo1}', size=18)
                else:
                    sb.countplot(x=campo1, data=df1)
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
                    sb.histplot(df2[campo2], kde=True, color='green')
                    plt.ylabel('Densidade', size=14)
                    plt.title(f'Histograma (sns.histplot) de {campo2}', size=18)
                else:
                    sb.countplot(x=campo2, data=df2)
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
                sb.pairplot(df_numeric)
             
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
            sb.scatterplot(x=campo1, y=campo2, color='r', data=df)
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
                sb.pairplot(df_numeric, 
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
                sb.heatmap(df.corr(),annot=True,square=True,
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
    sb.scatterplot(x=pc[:,0], y=pc[:,1], hue=y_cluster, palette='bright', ax=ax)
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

@analise.route('/dashboard_dois_console')
def dashboard_dois_console():
    df1 = get_dataframe1(csv_filepath1)

    csv_filepath_old1 = os.path.join('df1.csv')
    df1_old = pd.read_csv(csv_filepath_old1, encoding='cp1252', delimiter=';')

    # Removendo duplicatas
    df1.drop_duplicates(keep='first', inplace=True)

    combinacoes = list(itertools.combinations(campos1, 2))

    # Captura a saída de df.info()
    buffer = StringIO()
    df1.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    buffer_old1 = StringIO()
    df1.info(buf=buffer_old1)
    infos_variaveis_old1 = buffer_old1.getvalue()

    # Calculando correlações
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
    df1_std = pd.DataFrame(data=df1_std, columns=df1.columns)

    Soma_distancia_quadratica = []
    K = range(1, 11)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42)
        km = km.fit(df1_std)
        Soma_distancia_quadratica.append(km.inertia_)

    # Exibindo a soma das distâncias quadráticas no console
    print("Soma das Distâncias Quadráticas (Método do Cotovelo):")
    for i, s in enumerate(Soma_distancia_quadratica, 1):
        print(f"K={i}: {s}")

    # Silhouette scores
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=10)
        labels = kmeans.fit_predict(df1_std)
        score = silhouette_score(df1_std, labels)
        silhouette_scores.append(score)
        print(f'Para n_clusters={k}, Silhouette score é {score}')

    # Exibindo os scores no console
    print("Silhouette Scores:")
    for i, s in enumerate(silhouette_scores, 2):
        print(f"K={i}: {s}")

    # Performando o clustering
    kmeans = KMeans(n_clusters=6, random_state=42)
    cluster_labels = kmeans.fit_predict(df1_std)
    cluster_analysis_report = cluster_report(df1, cluster_labels, min_samples_leaf=50, pruning_level=0.01)

    # Exibindo o relatório de análise de clusters no console
    print("Relatório de Análise de Clusters:")
    print(cluster_analysis_report.to_string())

    # Exibindo informações do dataframe original e do processado
    print("\nInformações do DataFrame Original:")
    print(df1_old.info())
    print("\nDescrição do DataFrame Original:")
    print(df1_old.describe())
    print(df1_old.head())
    print("\nInformações do DataFrame Processado:")
    print(df1.info())
    print("\nDescrição do DataFrame Processado:")
    print(df1.describe())

    print("  ")
    print(df1_old.shape)

    print("  ")
    print(df1_old.isnull().sum())
    print("  ")  
    print(df1_old.dtypes)      

    # Dividindo os dados para treinamento e teste
    X = df1_std.iloc[:, :-1]
    y = df1_std.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # Treinando a árvore de decisão
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=70, min_samples_leaf=5)
    tree.fit(X_train, y_train)
    predictions_test = tree.predict(X_test)
    accuracy_test = accuracy_score(y_test, predictions_test) * 100
    report_test = classification_report(y_test, predictions_test)

    # Exibindo os resultados do teste
    print("\nAcurácia do Teste:", accuracy_test)
    print("\nRelatório de Classificação do Teste:\n", report_test)

    # Matriz de Confusão para Teste
    cf_test = confusion_matrix(y_test, predictions_test)
    print("\nMatriz de Confusão do Teste:\n", cf_test)

    # Cross Validation
    predictions_cv = cross_val_predict(tree, X, y, cv=10)
    accuracy_cv = accuracy_score(y, predictions_cv) * 100
    report_cv = classification_report(y, predictions_cv)

    # Exibindo os resultados da validação cruzada
    print("\nAcurácia da Validação Cruzada:", accuracy_cv)
    print("\nRelatório de Classificação da Validação Cruzada:\n", report_cv)

    # Matriz de Confusão para Validação Cruzada
    cf_cv = confusion_matrix(y, predictions_cv)
    print("\nMatriz de Confusão da Validação Cruzada:\n", cf_cv)

    return "Resultados exibidos no console"


if __name__ == '__main__':
    analise.run(debug=True)
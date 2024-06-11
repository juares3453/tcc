from flask import Flask, render_template, send_file
import matplotlib as mpl
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy import stats
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
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

campos1 = ['Dia', 'Mes', 'Ano', 'DsTpVeiculo', 'VlCusto', 'km_rodado', 'VlCapacVeic',
       'NrAuxiliares', '%CapacidadeCarre', '%CapacidadeEntr', '%Entregas', '%VolumesEntr', '%PesoEntr', '%FreteCobrado', 'FreteEx',
       'Lucro', '%Lucro']

csv_filepath1 = os.path.join('df1.csv')

def get_dataframe1(csv_filepath1):
    df1 = pd.read_csv(csv_filepath1, encoding='cp1252', delimiter=';')
    return df1

def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    def tree_dfs(node_id=0, current_rule=[]):
        split_feature = inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED:  # nó interno
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            tree_dfs(inner_tree.children_left[node_id], left_rule)
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            tree_dfs(inner_tree.children_right[node_id], right_rule)
        else:  # folha
            dist = inner_tree.value[node_id][0]
            dist = dist / dist.sum()
            max_idx = dist.argmax()
            rule_string = " and ".join(current_rule) if current_rule else "ALL"
            selected_class = classes[max_idx]
            class_probability = dist[max_idx]
            class_rules = class_rules_dict.get(selected_class, [])
            class_rules.append((rule_string, class_probability))
            class_rules_dict[selected_class] = class_rules

    tree_dfs()  # começa da raiz, node_id = 0
    return class_rules_dict

def cluster_report(data: pd.DataFrame, clusters, criterion='entropy', max_depth=4, min_samples_leaf=1):
    tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(data, clusters)

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
    return report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']]

def print_cluster_report(report_df):
    for index, row in report_df.iterrows():
        print(f"Cluster {row['class_name']} ({row['instance_count']} instâncias)")
        print("Regras:")
        rules = row['rule_list'].split("\\n\\n")
        for rule in rules:
            print(f"  - {rule}")
        print("\n")


@analise.route('/gerar_graficos')
def gerar_graficos():
    csv_filepath1 = os.path.join('df1.csv')
    df1 = get_dataframe1(csv_filepath1)

    # Substituição de vírgulas por pontos na coluna 'VlCusto' e 'Lucro'
    df1['VlCusto'] = df1['VlCusto'].str.replace(',', '.')
    df1['Lucro'] = df1['Lucro'].str.replace(',', '.')

    # Conversão de tipos de dados
    df1['VlCusto'] = pd.to_numeric(df1['VlCusto'], errors='coerce')
    df1['Lucro'] = pd.to_numeric(df1['Lucro'], errors='coerce')

    # Tratamento de Valores Nulos
    imputer = SimpleImputer(strategy='mean')
    df1[['VlCusto', 'Lucro', '%CapacidadeCarre']] = imputer.fit_transform(df1[['VlCusto', 'Lucro', '%CapacidadeCarre']])

    # Imputação com mediana
    imputer = SimpleImputer(strategy='median')
    df1[['FreteEx', '%Lucro']] = imputer.fit_transform(df1[['FreteEx', '%Lucro']])

    # Conversão de Tipos de Dados
    df1['Dia'] = df1['Dia'].astype(int)
    df1['Mes'] = df1['Mes'].astype(int)
    df1['Ano'] = df1['Ano'].astype(int)

    # Unificação de colunas de data
    df1['data'] = pd.to_datetime(df1[['Ano', 'Mes', 'Dia']].astype(str).agg('-'.join, axis=1), errors='coerce')

    # Remoção de Colunas Desnecessárias
    df1 = df1.drop(columns=['Ano', 'Mes', 'Dia', '%CapacidadeEntr', '%VolumesEntr', '%PesoEntr', '%Entregas', '%FreteCobrado']) 

    # Codificação de Variáveis Categóricas
    label_encoder = LabelEncoder()
    df1['DsTpVeiculo'] = label_encoder.fit_transform(df1['DsTpVeiculo'])
    df1['DsModelo'] = label_encoder.fit_transform(df1['DsModelo'])
    df1['DsAnoFabricacao'] = label_encoder.fit_transform(df1['DsAnoFabricacao'])
    df1['VlCapacVeic'] = label_encoder.fit_transform(df1['VlCapacVeic'])

    df1[['VlCusto', 'km_rodado', 'FreteEx', '%CapacidadeCarre']] = np.log1p(df1[['VlCusto', 'km_rodado', 'FreteEx', '%CapacidadeCarre']])

    # Normalização e Padronização
    scaler = MinMaxScaler()
    df1[['Lucro', '%Lucro', 'DsTpVeiculo', 'DsModelo', 'DsAnoFabricacao', 'VlCapacVeic', 'NrAuxiliares']] = scaler.fit_transform(df1[['Lucro', '%Lucro', 'DsTpVeiculo', 'DsModelo', 'DsAnoFabricacao', 'VlCapacVeic', 'NrAuxiliares']])
    
    # Tratamento de Outliers
    Q1 = df1[['km_rodado', 'FreteEx', 'Lucro', '%Lucro', '%CapacidadeCarre']].quantile(0.25)
    Q3 = df1[['km_rodado', 'FreteEx',  'Lucro', '%Lucro', '%CapacidadeCarre']].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[~((df1[['km_rodado', 'FreteEx', 'Lucro', '%Lucro', '%CapacidadeCarre']] < (Q1 - 1.5 * IQR)) | (df1[['km_rodado', 'FreteEx', 'Lucro', '%Lucro', '%CapacidadeCarre']] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Verificação de Duplicatas
    df1.drop_duplicates(inplace=True)

    # Informações após o tratamento
    print("Shape do DataFrame:")
    print(df1.shape)
    print(" ")
    print("Valores nulos por coluna:")
    print(df1.isnull().sum())
    print(" ")
    print("Tipos de dados:")
    print(df1.dtypes)
    print(" ")
    print("Primeiras linhas do DataFrame:")
    print(df1.head())

    primeiro_dia = df1['data'].min().strftime("%d %b %Y") 
    ultimo_dia = df1['data'].max().strftime("%d %b %Y") 
    total_dias = df1['data'].max() - df1['data'].min()

    print(f"Primeira registro do caso 2: {primeiro_dia}")
    print(f"Último registro do caso 2: {ultimo_dia}")
    print(f"Total de dias do caso 2: {total_dias}")

    print(df1.columns)
    # Heatmap de Correlação
    # plt.figure(figsize=(12, 8))
    # correlation_matrix = df1.corr()
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # plt.title('Heatmap de Correlação')
    # plt.savefig('static/graficos/new/dois/heatmap_correlacao_dois1.png')  # Salvar o heatmap como um arquivo de imagem
    # plt.show()

    # Histograma
    plt.figure(figsize=(12, 8))
    df1[['VlCusto', 'km_rodado', 'NrAuxiliares', '%CapacidadeCarre', 'FreteEx', 'Lucro', '%Lucro']].hist(bins=20, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('static/graficos/new/dois/histogramas_dois1.png')  # Salvar o histograma como um arquivo de imagem
    plt.show()

     # # Visualizações
    # plt.figure(figsize=(10, 6))
    # sns.histplot(df1['VlCusto'].dropna(), bins=50, kde=True)
    # plt.title('Distribuição de VlCusto')
    # plt.xlabel('VlCusto')
    # plt.ylabel('Frequência')
    # plt.savefig('static/graficos/new/dois/vlcusto_dois1.png') 
    # plt.close()

    # plt.figure(figsize=(10, 6))
    # sns.histplot(df1['Lucro'].dropna(), bins=50, kde=True)
    # plt.title('Distribuição de Lucro')
    # plt.xlabel('Lucro')
    # plt.ylabel('Frequência')
    # plt.savefig('static/graficos/new/dois/lucro_dois1.png') 
    # plt.close()

    # plt.figure(figsize=(14, 8))
    # sns.boxplot(data=df1, x='DsTpVeiculo', y='VlCusto')
    # plt.title('Dispersão de VlCusto por Tipo de Veículo')
    # plt.xlabel('Tipo de Veículo')
    # plt.ylabel('VlCusto')
    # plt.savefig('static/graficos/new/dois/vlcusto_tpveiculo_dois1.png') 
    # plt.close()

    # plt.figure(figsize=(14, 8))
    # sns.boxplot(data=df1, x='DsTpVeiculo', y='Lucro')
    # plt.title('Dispersão de Lucro por Tipo de Veículo')
    # plt.xlabel('Tipo de Veículo')
    # plt.ylabel('Lucro')
    # plt.savefig('static/graficos/new/dois/dstpveiculo_lucro_dois1.png') 
    # plt.close()

    # plt.figure(figsize=(14, 8))
    # df1['mes_ano'] = df1['data'].dt.to_period('M')
    # ocorrencias_por_mes = df1.groupby('mes_ano').size()
    # ocorrencias_por_mes.plot(kind='line')
    # plt.title('Número de Ocorrências ao Longo do Tempo')
    # plt.xlabel('Mês/Ano')
    # plt.ylabel('Número de Ocorrências')
    # plt.xticks(rotation=45)
    # plt.savefig('static/graficos/new/dois/entregas_dois1.png') 
    # plt.close()

    # Remoção de Colunas Desnecessárias
    df1 = df1.drop(columns=['data'])

    # plt.figure(figsize=(12, 10))
    # sns.heatmap(df1.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.title('Heatmap de Correlações')
    # plt.savefig('static/graficos/new/dois/corr_dois1.png') 
    # plt.close()

    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(data=df1, x='km_rodado', y='VlCusto')
    # plt.title('Relação entre Quilometragem Rodada e VlCusto')
    # plt.xlabel('Quilometragem Rodada')
    # plt.ylabel('VlCusto')
    # plt.savefig('static/graficos/new/dois/kmrodado_custo_dois1.png') 
    # plt.close()

    # plt.figure(figsize=(10, 6))
    # sns.countplot(data=df1, x='DsTpVeiculo')
    # plt.title('Distribuição de Tipos de Veículo')
    # plt.xlabel('Tipo de Veículo')
    # plt.ylabel('Frequência')
    # plt.savefig('static/graficos/new/dois/tpveiculo_dois1.png') 
    # plt.close()

    # # Scatter plot para visualizar concentração
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(data=df1, x='km_rodado', y='VlCusto', hue='DsTpVeiculo', palette='Set1')
    # plt.title('Relação entre Quilometragem Rodada e VlCusto por Tipo de Veículo')
    # plt.xlabel('Quilometragem Rodada')
    # plt.ylabel('VlCusto')
    # plt.legend(title='Tipo de Veículo')
    # plt.savefig('static/graficos/new/dois/tpveiculo_km_vlcusto_dois1.png') 
    # plt.close()

    # Aplicação do KMeans e Determinação do Número Ideal de Clusters
    X_scaled = df1.copy()
    silhouette_scores = []
    inertia = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        inertia.append(kmeans.inertia_)

    optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
    optimal_k_elbow = 4  # Finding the elbow points

    # Clustering com Silhouette
    kmeans_silhouette = KMeans(n_clusters=optimal_k_silhouette, random_state=42)
    kmeans_silhouette.fit(X_scaled)
    df1['Cluster_Silhouette'] = kmeans_silhouette.labels_

    # Clustering com Elbow
    kmeans_elbow = KMeans(n_clusters=optimal_k_elbow, random_state=42)
    kmeans_elbow.fit(X_scaled)
    df1['Cluster_Elbow'] = kmeans_elbow.labels_

    # PCA para visualização
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df1['PCA1'] = X_pca[:, 0]
    df1['PCA2'] = X_pca[:, 1]

    # Plotagem dos clusters com Silhouette
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_Silhouette', data=df1, palette='viridis', s=100, alpha=0.6, edgecolor='k')
    plt.title(f'Clusters com K={optimal_k_silhouette} (Silhouette)')
    plt.savefig('static/graficos/new/dois/clusters_silhouette.png')
    plt.close()

    # Plotagem dos clusters com Elbow
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_Elbow', data=df1, palette='viridis', s=100, alpha=0.6, edgecolor='k')
    plt.title(f'Clusters com K={optimal_k_elbow} (Elbow)')
    plt.savefig('static/graficos/new/dois/clusters_elbow.png')
    plt.close()

    # Plotagem do Gráfico de Silhouette
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.title('Silhouette Scores para Diferentes Valores de K')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.savefig('static/graficos/new/dois/silhouette_scores.png')
    plt.close()

    # Plotagem do Gráfico de Elbow
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Método Elbow para Determinar o Número de Clusters')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inertia')
    plt.savefig('static/graficos/new/dois/elbow_method.png')
    plt.close()

    silhouette_avg_silhouette = silhouette_score(X_scaled, kmeans_silhouette.labels_)
    silhouette_avg_elbow = silhouette_score(X_scaled, kmeans_elbow.labels_)
    davies_bouldin_silhouette = davies_bouldin_score(X_scaled, kmeans_silhouette.labels_)
    davies_bouldin_elbow = davies_bouldin_score(X_scaled, kmeans_elbow.labels_)

    print(f'Silhouette Score (Silhouette): {silhouette_avg_silhouette}')
    print(f'Silhouette Score (Elbow): {silhouette_avg_elbow}')
    print(f'Davies-Bouldin Score (Silhouette): {davies_bouldin_silhouette}')
    print(f'Davies-Bouldin Score (Elbow): {davies_bouldin_elbow}')

    # tree = DecisionTreeClassifier()
    # tree_para = {'criterion':['entropy','gini'],'max_depth':
    #              [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
    #              'min_samples_leaf':[1,2,3,4,5]}
    # grid = GridSearchCV(tree, tree_para,verbose=5, cv=10)
    # grid.fit(X_scaled,kmeans_silhouette.labels_)
    # best_clf = grid.best_estimator_
    # best = best_clf

    # # Exibir resultados do GridSearchCV
    # best_params = grid.best_params_
    # best_score = grid.best_score_
    
    # print(f'Best parameters found: {best_params}')
    # print(f'Best score found: {best_score}')

    # Criação de conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,kmeans_silhouette.labels_,test_size=0.3,random_state=100)
    train = (X_train.shape, y_train.shape) # shape - mostra quantas linhas e colunas for+am geradas
    test = (X_test.shape, y_test.shape)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=1)
    tree.fit(X_train,y_train)
    predictions_test = tree.predict(X_test)
    accuracy_test = accuracy_score(y_test,predictions_test)*100
    report_test = classification_report(y_test,predictions_test)

    print(f'Acurácia: {accuracy_test}')
    print(f'Report: {report_test}')

    #Test
    cf = confusion_matrix(y_test,predictions_test)
    lbl1 = ['high', 'low']
    lbl2 = ['high', 'low']
    plt.figure(figsize=(10, 7))
    sns.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Test")
    plt.savefig('static/graficos/new/dois/df_confusion_matrix_test.png')  # Salvando o gráfico
    plt.close()

    #Cross Validation
    predictions_test = cross_val_predict(tree,X_scaled,kmeans_silhouette.labels_,cv=10)
    accuracy_test1 = accuracy_score(kmeans_silhouette.labels_,predictions_test)*100

    print(f'Acurácia: {accuracy_test1}')

    predictions = cross_val_predict(tree,X_scaled,kmeans_silhouette.labels_,cv=10)
    cf = confusion_matrix(kmeans_silhouette.labels_,predictions)
    lbl1 = ['high', 'low']
    lbl2 = ['high', 'low']
    plt.figure(figsize=(10, 7))
    sns.heatmap(cf, annot=True, cmap="Greens", fmt="d", xticklabels=lbl1, yticklabels=lbl2)
    plt.title("Confusion Matrix - Cross Validation")
    plt.savefig('static/graficos/new/dois/df_confusion_matrix_cv.png')  # Salvando o gráfico
    plt.close()

    report = classification_report(kmeans_silhouette.labels_,predictions)
    print(f'Report: {report}')
    
    #Gera arvore de decisao
    plt.figure(figsize=(100, 100))
    plot_tree(tree, filled=True, fontsize=16, proportion=True)
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    plt.title("Decision Tree")
    plt.savefig('static/graficos/new/dois/df_decision_tree.png')  # Salvando o gráfico
    plt.close()

    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    trees = []
    for ccp_alpha in ccp_alphas:
        tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        tree.fit(X_train, y_train)
        trees.append(tree)

    train_scores = [tree.score(X_train, y_train) for tree in trees]
    test_scores = [tree.score(X_test, y_test) for tree in trees]

    best_alpha_index = np.argmax(test_scores)
    best_tree = trees[best_alpha_index]

    plt.figure(figsize=(40, 30))
    plot_tree(best_tree, filled=True, fontsize=12, proportion=True)
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    plt.savefig('static/graficos/new/dois/df_decision_tree_poda.png')  # Salvando o gráfico
    plt.close()

    # Relatório de clusters
    report_df = cluster_report(pd.DataFrame(X_scaled, columns=df1.columns), kmeans_silhouette.labels_)

    # Imprimir relatório no console
    print_cluster_report(report_df)

    return "Processamento concluído e informações exibidas no console."

if __name__ == '__main__':
    analise.run(debug=True)
from flask import Flask
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
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

csv_filepath2 = os.path.join('df2.csv')

# Função para carregar e preparar os dados
def get_dataframe(filepath):
    df = pd.read_csv(filepath, encoding='cp1252', delimiter=';')
    return df

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
    df2 = get_dataframe(csv_filepath2)

    # Substituição de vírgulas por pontos na coluna 'VlCusto'
    df2['VlCusto'] = df2['VlCusto'].str.replace(',', '.')

    # Conversão de colunas numéricas para tipos numéricos, tratando erros
    colunas_numericas = ['dtcte', 'mescte', 'anocte', 'dtemissao', 'mesemissao', 'anoemissao', 'dtocor', 'mesocor', 
                         'anoocor', 'dtbaixa', 'mesbaixa', 'anobaixa', 'diasemissao', 'diasresolucao', 'NrBo', 'VlCusto']
    for coluna in colunas_numericas:
        df2[coluna] = pd.to_numeric(df2[coluna], errors='coerce')

    # Tratamento de Valores Nulos
    imputer_num = SimpleImputer(strategy='mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df2[colunas_numericas] = imputer_num.fit_transform(df2[colunas_numericas])
    colunas_categoricas = ['Resp', 'CLIENTE', 'DsLocal', 'tp_ocor', 'Situacao', 'dsocorrencia']
    

    # Codificação de Variáveis Categóricas
    label_encoder = LabelEncoder()
    for coluna in colunas_categoricas:
        df2[coluna] = label_encoder.fit_transform(df2[coluna].astype(str))

    df2[colunas_categoricas] = imputer_cat.fit_transform(df2[colunas_categoricas])

     # Conversão de Tipos de Dados
    df2['anocte'] = df2['anocte'].astype(int)
    df2['mescte'] = df2['mescte'].astype(int)
    df2['dtcte'] = df2['dtcte'].astype(int)
    df2['anoemissao'] = df2['anoemissao'].astype(int)
    df2['mesemissao'] = df2['mesemissao'].astype(int)
    df2['dtemissao'] = df2['dtemissao'].astype(int)
    df2['anoocor'] = df2['anoocor'].astype(int)
    df2['mesocor'] = df2['mesocor'].astype(int)
    df2['dtocor'] = df2['dtocor'].astype(int)
    df2['anobaixa'] = df2['anobaixa'].astype(int)
    df2['mesbaixa'] = df2['mesbaixa'].astype(int)
    df2['dtbaixa'] = df2['dtbaixa'].astype(int)

    # Combinação de colunas de datas
    df2['data_cte'] = pd.to_datetime(df2[['anocte', 'mescte', 'dtcte']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d', errors='coerce')
    df2['data_emissao_bo'] = pd.to_datetime(df2[['anoemissao', 'mesemissao', 'dtemissao']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d', errors='coerce')
    df2['data_ocor'] = pd.to_datetime(df2[['anoocor', 'mesocor', 'dtocor']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d', errors='coerce')
    df2['data_baixa'] = pd.to_datetime(df2[['anobaixa', 'mesbaixa', 'dtbaixa']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d', errors='coerce')

    # Exclusão de colunas de dia, mês e ano originais
    df2 = df2.drop(columns=[ 'dtemissao', 'dtocor', 'mesocor', 'anoocor', 
                            'dtbaixa', 'mesbaixa', 'anobaixa', 'dtcte', 'mescte', 'anocte', 'data_cte', 'data_emissao_bo', 'data_ocor', 'data_baixa', 'mesemissao', 'anoemissao', 'diasemissao'])

    # Remoção de duplicatas
    df2.drop_duplicates(inplace=True)

    # Tratamento de Outliers usando Z-score
    z_scores = np.abs(stats.zscore(df2[['diasresolucao', 'NrBo', 'dsocorrencia', 'CLIENTE', 'DsLocal', 'tp_ocor', 'VlCusto', 'Situacao', 'Resp']]))
    df2 = df2[(z_scores < 3).all(axis=1)]

    # Normalização dos dados
    scaler = MinMaxScaler()
    colunas_para_normalizar = ['diasresolucao', 'NrBo', 'dsocorrencia', 'CLIENTE', 'DsLocal', 'tp_ocor', 'VlCusto', 'Situacao', 'Resp']
    df2[colunas_para_normalizar] = scaler.fit_transform(df2[colunas_para_normalizar])


    # # Análise e tratamento de outliers
    # Q1 = df2['VlCusto'].quantile(0.25)
    # Q3 = df2['VlCusto'].quantile(0.75)
    # IQR = Q3 - Q1
    # df2 = df2[~((df2['VlCusto'] < (Q1 - 1.5 * IQR)) | (df2['VlCusto'] > (Q3 + 1.5 * IQR)))]

    # Impressão das informações no console
    print("Shape do DataFrame:")
    print(df2.shape)
    print(" ")
    print("Valores nulos por coluna:")
    print(df2.isnull().sum())
    print(" ")
    print("Tipos de dados:")
    print(df2.dtypes)
    print(" ")
    print("Primeiras linhas do DataFrame:")
    print(df2.head())
    
    # primeiro_dia = df2['data_emissao_bo'].min().strftime("%d %b %Y") 
    # ultimo_dia = df2['data_emissao_bo'].max().strftime("%d %b %Y") 
    # total_dias = df2['data_emissao_bo'].max() - df2['data_emissao_bo'].min()

    # print(f"Primeira registro do caso 3: {primeiro_dia}")
    # print(f"Último registro do caso 3: {ultimo_dia}")
    # print(f"Total de dias do caso 3: {total_dias}")

    # Heatmap de Correlação
    plt.figure(figsize=(12, 8))
    correlation_matrix = df2.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap de Correlação')
    plt.savefig('static/graficos/new/tres/heatmap_correlacao_tres1.png')  # Salvar o heatmap como um arquivo de imagem
    plt.show()


    # # Histograma
    plt.figure(figsize=(12, 8))
    df2[['diasresolucao', 'NrBo', 'dsocorrencia', 'CLIENTE', 'DsLocal', 'tp_ocor', 'VlCusto', 'Situacao', 'Resp']].hist(bins=20, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('static/graficos/new/tres/histogramas_tres1.png')  # Salvar o histograma como um arquivo de imagem
    plt.show()

    # # Histograma de VlCusto 
    # plt.figure(figsize=(10, 6)) 
    # sns.histplot(df2['VlCusto'], bins=50, kde=True) 
    # plt.title('Distribuição de VlCusto') 
    # plt.xlabel('VlCusto') 
    # plt.ylabel('Frequência') 
    # plt.savefig('static/graficos/new/tres/histograma_vlcusto_tres1.png')  # Salvar o histograma como um arquivo de imagem
    # plt.show()

    # # Gráfico de barras de DsLocal
    # plt.figure(figsize=(14, 8))
    # sns.countplot(data=df2, x='DsLocal', order=df2['DsLocal'].value_counts().index)
    # plt.title('Contagem de Ocorrências por Local')
    # plt.xlabel('Local')
    # plt.ylabel('Contagem')
    # plt.xticks(rotation=90)
    # plt.savefig('static/graficos/new/tres/contagem_DsLocal_tres1.png')  # Salvar o histograma como um arquivo de imagem
    # plt.show()

    # # Boxplot de VlCusto por tp_ocor
    # plt.figure(figsize=(14, 8))
    # sns.boxplot(data=df2, x='tp_ocor', y='VlCusto')
    # plt.title('Dispersão de VlCusto por Tipo de Ocorrência')
    # plt.xlabel('Tipo de Ocorrência')
    # plt.ylabel('VlCusto')
    # plt.savefig('static/graficos/new/tres/ocor_vlcusto_tres1.png')
    # plt.show()
   
    # # Boxplot
    # plt.figure(figsize=(12, 6))
    # sns.boxplot(data=df2[['VlCusto', 'NrBo']])
    # plt.title('Boxplot de VlCusto e NrBo')
    # plt.savefig('static/graficos/new/tres/boxplot_tres1.png')  # Salvar o boxplot como um arquivo de imagem
    # plt.show()

    # # Boxplot de VlCusto por tp_ocor
    # plt.figure(figsize=(14, 8))
    # sns.boxplot(data=df2, x='tp_ocor', y='VlCusto')
    # plt.title('Dispersão de VlCusto por Tipo de Ocorrência')
    # plt.xlabel('Tipo de Ocorrência')
    # plt.ylabel('VlCusto')
    # plt.savefig('static/graficos/new/tres/boxplot_vlcusto_tpocor_tres1.png')  # Salvar o boxplot como um arquivo de imagem
    # plt.show()

    # # Gráfico de linha de número de ocorrências ao longo do tempo
    # df2['mes_ano'] = df2['data_cte'].dt.to_period('M')
    # ocorrencias_por_mes = df2.groupby('mes_ano').size()

    # plt.figure(figsize=(14, 8))
    # ocorrencias_por_mes.plot(kind='line')
    # plt.title('Número de Ocorrências ao Longo do Tempo')
    # plt.xlabel('Mês/Ano')
    # plt.ylabel('Número de Ocorrências')
    # plt.xticks(rotation=45)
    # plt.savefig('static/graficos/new/tres/ocorrencias_mes_tres1.png')  # Salvar o boxplot como um arquivo de imagem
    # plt.show()

    # # Boxplot de VlCusto por mesemissao
    # plt.figure(figsize=(14, 8))
    # sns.boxplot(data=df2, x='mesemissao', y='VlCusto')
    # plt.title('Dispersão de VlCusto por Mês de Emissão')
    # plt.xlabel('Mês de Emissão')
    # plt.ylabel('VlCusto')
    # plt.savefig('static/graficos/new/tres/vlcusto_mes_tres1.png')  # Salvar o boxplot como um arquivo de imagem
    # plt.show()

    # # Boxplot de VlCusto por anoemissao
    # plt.figure(figsize=(14, 8))
    # sns.boxplot(data=df2, x='anoemissao', y='VlCusto')
    # plt.title('Dispersão de VlCusto por Ano de Emissão')
    # plt.xlabel('Ano de Emissão')
    # plt.ylabel('VlCusto')
    # plt.savefig('static/graficos/new/tres/vlcusto_ano_tres1.png')  # Salvar o boxplot como um arquivo de imagem
    # plt.show()

    # # Remoção de Colunas Desnecessárias
    # df2 = df2.drop(columns=['mes_ano'])

    # # Heatmap de Correlação
    # plt.figure(figsize=(12, 8))
    # correlation_matrix = df2.corr()
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # plt.title('Heatmap de Correlação')
    # plt.savefig('static/graficos/new/tres/heatmap_correlacao_tres1.png')  # Salvar o heatmap como um arquivo de imagem
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # df2['tp_ocor'].value_counts().plot(kind='bar')
    # plt.title('Frequência de tp_ocor')
    # plt.savefig('static/graficos/new/tres/barras_tp_ocor_tres1.png')  # Salvar o gráfico de barras como um arquivo de imagem
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # df2['Situacao'].value_counts().plot(kind='bar')
    # plt.title('Frequência de Situacao')
    # plt.savefig('static/graficos/new/tres/barras_situacao_tres1.png')  # Salvar o gráfico de barras como um arquivo de imagem
    # plt.show()

    # # Scatter plot entre diasresolucao e VlCusto
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(data=df2, x='diasresolucao', y='VlCusto')
    # plt.title('Relação entre Dias de Resolução e VlCusto')
    # plt.xlabel('Dias de Resolução')
    # plt.ylabel('VlCusto')
    # plt.savefig('static/graficos/new/tres/vlcusto_diasresol_tres1.png')
    # plt.show()

    # Aplicação do KMeans e Determinação do Número Ideal de Clusters
    X_scaled = df2.copy()
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
    optimal_k_elbow = K_range[np.argmin(np.diff(inertia, 2)) + 1]  # Finding the elbow points

    # Clustering com Silhouette
    kmeans_silhouette = KMeans(n_clusters=optimal_k_silhouette, random_state=42)
    kmeans_silhouette.fit(X_scaled)
    df2['Cluster_Silhouette'] = kmeans_silhouette.labels_

    # Clustering com Elbow
    kmeans_elbow = KMeans(n_clusters=optimal_k_elbow, random_state=42)
    kmeans_elbow.fit(X_scaled)
    df2['Cluster_Elbow'] = kmeans_elbow.labels_

    # PCA para visualização
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df2['PCA1'] = X_pca[:, 0]
    df2['PCA2'] = X_pca[:, 1]

    # Plotagem dos clusters com Silhouette
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_Silhouette', data=df2, palette='viridis', s=100, alpha=0.6, edgecolor='k')
    plt.title(f'Clusters com K={optimal_k_silhouette} (Silhouette)')
    plt.savefig('static/graficos/new/tres/clusters_silhouette.png')
    plt.close()

    # Plotagem dos clusters com Elbow
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_Elbow', data=df2, palette='viridis', s=100, alpha=0.6, edgecolor='k')
    plt.title(f'Clusters com K={optimal_k_elbow} (Elbow)')
    plt.savefig('static/graficos/new/tres/clusters_elbow.png')
    plt.close()

    # Plotagem do Gráfico de Silhouette
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.title('Silhouette Scores para Diferentes Valores de K')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.savefig('static/graficos/new/tres/silhouette_scores.png')
    plt.close()

    # Plotagem do Gráfico de Elbow
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Método Elbow para Determinar o Número de Clusters')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inertia')
    plt.savefig('static/graficos/new/tres/elbow_method.png')
    plt.close()

    silhouette_avg_silhouette = silhouette_score(X_scaled, kmeans_silhouette.labels_)
    silhouette_avg_elbow = silhouette_score(X_scaled, kmeans_elbow.labels_)
    davies_bouldin_silhouette = davies_bouldin_score(X_scaled, kmeans_silhouette.labels_)
    davies_bouldin_elbow = davies_bouldin_score(X_scaled, kmeans_elbow.labels_)

    print(f'Silhouette Score (Silhouette): {silhouette_avg_silhouette}')
    print(f'Silhouette Score (Elbow): {silhouette_avg_elbow}')
    print(f'Davies-Bouldin Score (Silhouette): {davies_bouldin_silhouette}')
    print(f'Davies-Bouldin Score (Elbow): {davies_bouldin_elbow}')
    
    tree = DecisionTreeClassifier()
    tree_para = {'criterion':['entropy','gini'],'max_depth':
                 [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
                 'min_samples_leaf':[1,2,3,4,5]}
    grid = GridSearchCV(tree, tree_para,verbose=5, cv=10)
    grid.fit(X_scaled,kmeans_silhouette.labels_)
    best_clf = grid.best_estimator_
    best = best_clf

    # Exibir resultados do GridSearchCV
    best_params = grid.best_params_
    best_score = grid.best_score_
    
    print(f'Best parameters found: {best_params}')
    print(f'Best score found: {best_score}')

    # Criação de conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,kmeans_silhouette.labels_,test_size=0.3,random_state=100)
    train = (X_train.shape, y_train.shape) # shape - mostra quantas linhas e colunas for+am geradas
    test = (X_test.shape, y_test.shape)
    tree = DecisionTreeClassifier(criterion='gini', max_depth=5, max_features='sqrt', min_samples_leaf=4, min_samples_split=2, ccp_alpha=0.0)
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
    plt.savefig('static/graficos/new/tres/df_confusion_matrix_test.png')  # Salvando o gráfico
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
    plt.savefig('static/graficos/new/tres/df_confusion_matrix_cv.png')  # Salvando o gráfico
    plt.close()

    report = classification_report(kmeans_silhouette.labels_,predictions)
    print(f'Report: {report}')
    
    #Gera arvore de decisao
    plt.figure(figsize=(100, 100))
    plot_tree(tree, filled=True, fontsize=16, proportion=True)
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    plt.title("Decision Tree")
    plt.savefig('static/graficos/new/tres/df_decision_tree.png')  # Salvando o gráfico
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
    plt.savefig('static/graficos/new/tres/df_decision_tree_poda.png')  # Salvando o gráfico
    plt.close()

    # Relatório de clusters
    report_df = cluster_report(pd.DataFrame(X_scaled, columns=df2.columns), kmeans_silhouette.labels_)

    # Imprimir relatório no console
    print_cluster_report(report_df)

    return "Processamento concluído e informações exibidas no console."

if __name__ == '__main__':
    analise.run(debug=True)
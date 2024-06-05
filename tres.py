from flask import Flask
import matplotlib as mpl
import os
from io import StringIO
import itertools
from tqdm import tqdm
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

mpl.use('Agg')
mpl.rcParams['figure.max_open_warning'] = 50
analise = Flask(__name__)

# Diretório para salvar os gráficos
graficos_dir = 'static/graficos'
os.makedirs(graficos_dir, exist_ok=True)

# Lista de campos
campos2 = ['Resp', 'CLIENTE', 'dtcte','mescte','anocte','dtemissao','mesemissao','anoemissao','dtocor','mesocor','anoocor','dtbaixa','mesbaixa',
 'anobaixa','diasemissao','diasresolucao','DsLocal', 'tp_ocor', 'Situacao','NrBo','dsocorrencia','VlCusto']

csv_filepath2 = os.path.join('df2.csv')

def remover_valores_negativos(df):
    for coluna in df.columns:
        if pd.api.types.is_numeric_dtype(df[coluna]):
            df[coluna] = df[coluna].apply(lambda x: x if x >= 0 else np.nan)
    return df

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

def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    def tree_dfs(node_id=0, current_rule=[]):
        split_feature = inner_tree.feature[node_id]
        if split_feature != -2:  # internal node
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            tree_dfs(inner_tree.children_left[node_id], left_rule)
            # right child
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            tree_dfs(inner_tree.children_right[node_id], right_rule)
        else:  # leaf
            dist = inner_tree.value[node_id][0]
            dist = dist / dist.sum()
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

    tree_dfs()  # start from root, node_id = 0
    return class_rules_dict

def cluster_report(data: pd.DataFrame, clusters, min_samples_leaf=50, pruning_level=0.01):
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level)
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

    print(report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']])
    return report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']]

@analise.route('/')
def index():
    return "Bem-vindo à análise de dados!"

@analise.route('/gerar_graficos')
def gerar_graficos():
    df1 = get_dataframe2(csv_filepath2)

    csv_filepath_old1 = os.path.join('df2.csv')
    df2_old = pd.read_csv(csv_filepath_old1, encoding='cp1252', delimiter=';')

    # Removendo duplicatas
    df1.drop_duplicates(keep='first', inplace=True)

    # Captura a saída de df.info()
    buffer = StringIO()
    df1.info(buf=buffer)
    infos_variaveis = buffer.getvalue()

    # Calculando correlações
    correlacoes = {}
    for campo1 in campos2:
        for campo2 in campos2:
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

    # Exibindo informações do dataframe processado
    print("\nInformações do DataFrame Processado:")
    buffer = StringIO()
    df1.info(buf=buffer)
    print(buffer.getvalue())
    print("\nDescrição do DataFrame Processado:")
    print(df1.describe())
    print(df1.head())





    print("  ")
    print(df2_old.shape)

    print("  ")
    print(df2_old.isnull().sum())
    print("  ")  
    print(df2_old.dtypes)








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
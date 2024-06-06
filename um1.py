from flask import Flask, render_template, send_file
import matplotlib as mpl
import os
from io import StringIO
import itertools
from tqdm import tqdm
from time import time
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from IPython.display import HTML, display
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree, _tree

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

csv_filepath = os.path.join('df.csv')

def get_dataframe(csv_filepath):
    df = pd.read_csv(csv_filepath, encoding='cp1252', delimiter=';')
    return df

@analise.route('/gerar_graficos')
def dashboard_um_console():
    df = get_dataframe(csv_filepath)
    df_old = pd.read_csv(csv_filepath, encoding='cp1252', delimiter=';')

    # Informações após o tratamento
    print("Shape do DataFrame:")
    print(df.shape)
    print(" ")
    print("Valores nulos por coluna:")
    print(df.isnull().sum())
    print(" ")
    print("Tipos de dados:")
    print(df.dtypes)
    print(" ")
    print("Primeiras linhas do DataFrame:")
    print(df.head())

    return "RESULTADO"

if __name__ == '__main__':
    analise.run(debug=True)
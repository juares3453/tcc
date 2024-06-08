from flask import Flask
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

mpl.use('Agg')
mpl.rcParams['figure.max_open_warning'] = 50
analise = Flask(__name__)

# Diretório para salvar os gráficos
graficos_dir = 'static/graficos'
os.makedirs(graficos_dir, exist_ok=True)

csv_filepath2 = os.path.join('df2.csv')

# Função para carregar e preparar os dados
def get_dataframe(filepath):
    df = pd.read_csv(filepath, encoding='cp1252', delimiter=';')
    return df

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
    df2[colunas_categoricas] = imputer_cat.fit_transform(df2[colunas_categoricas])

    # Codificação de Variáveis Categóricas
    label_encoder = LabelEncoder()
    for coluna in colunas_categoricas:
        df2[coluna] = label_encoder.fit_transform(df2[coluna].astype(str))

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
    df2 = df2.drop(columns=[ 'dtemissao', 'mesemissao', 'anoemissao', 'dtocor', 'mesocor', 'anoocor', 
                            'dtbaixa', 'mesbaixa', 'anobaixa', 'dtcte', 'mescte', 'anocte'])

    # Remoção de duplicatas
    df2.drop_duplicates(inplace=True)

    # Normalização dos dados
    scaler = MinMaxScaler()
    colunas_para_normalizar = ['diasresolucao', 'diasemissao', 'NrBo', 'dsocorrencia', 'CLIENTE', 'DsLocal', 'tp_ocor', 'VlCusto', 'Situacao']
    df2[colunas_para_normalizar] = scaler.fit_transform(df2[colunas_para_normalizar])

    # Análise e tratamento de outliers
    Q1 = df2['VlCusto'].quantile(0.25)
    Q3 = df2['VlCusto'].quantile(0.75)
    IQR = Q3 - Q1
    df2 = df2[~((df2['VlCusto'] < (Q1 - 1.5 * IQR)) | (df2['VlCusto'] > (Q3 + 1.5 * IQR)))]

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
    
    primeiro_dia = df2['data_emissao_bo'].min().strftime("%d %b %Y") 
    ultimo_dia = df2['data_emissao_bo'].max().strftime("%d %b %Y") 
    total_dias = df2['data_emissao_bo'].max() - df2['data_emissao_bo'].min()

    print(f"Primeira registro do caso 3: {primeiro_dia}")
    print(f"Último registro do caso 3: {ultimo_dia}")
    print(f"Total de dias do caso 3: {total_dias}")

    # Histograma
    plt.figure(figsize=(12, 8))
    df2[['VlCusto', 'NrBo', 'diasresolucao', 'diasemissao']].hist(bins=20, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('static/graficos/new/histogramas_tres1.png')  # Salvar o histograma como um arquivo de imagem
    plt.show()

    # Histograma de VlCusto 
    plt.figure(figsize=(10, 6)) 
    sns.histplot(df2['VlCusto'], bins=50, kde=True) 
    plt.title('Distribuição de VlCusto') 
    plt.xlabel('VlCusto') 
    plt.ylabel('Frequência') 
    plt.savefig('static/graficos/new/histograma_vlcusto_tres1.png')  # Salvar o histograma como um arquivo de imagem
    plt.show()

    # Gráfico de barras de DsLocal
    plt.figure(figsize=(14, 8))
    sns.countplot(data=df2, x='DsLocal', order=df2['DsLocal'].value_counts().index)
    plt.title('Contagem de Ocorrências por Local')
    plt.xlabel('Local')
    plt.ylabel('Contagem')
    plt.xticks(rotation=90)
    plt.savefig('static/graficos/new/contagem_DsLocal_tres1.png')  # Salvar o histograma como um arquivo de imagem
    plt.show()

    # Boxplot de VlCusto por tp_ocor
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df2, x='tp_ocor', y='VlCusto')
    plt.title('Dispersão de VlCusto por Tipo de Ocorrência')
    plt.xlabel('Tipo de Ocorrência')
    plt.ylabel('VlCusto')
    plt.show()
   
    # Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df2[['VlCusto', 'NrBo']])
    plt.title('Boxplot de VlCusto e NrBo')
    plt.savefig('static/graficos/new/boxplot_tres1.png')  # Salvar o boxplot como um arquivo de imagem
    plt.show()

    # Boxplot de VlCusto por tp_ocor
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df2, x='tp_ocor', y='VlCusto')
    plt.title('Dispersão de VlCusto por Tipo de Ocorrência')
    plt.xlabel('Tipo de Ocorrência')
    plt.ylabel('VlCusto')
    plt.savefig('static/graficos/new/boxplot_vlcusto_tpocor_tres1.png')  # Salvar o boxplot como um arquivo de imagem
    plt.show()

    # Gráfico de linha de número de ocorrências ao longo do tempo
    df2['mes_ano'] = df2['data_cte'].dt.to_period('M')
    ocorrencias_por_mes = df2.groupby('mes_ano').size()

    plt.figure(figsize=(14, 8))
    ocorrencias_por_mes.plot(kind='line')
    plt.title('Número de Ocorrências ao Longo do Tempo')
    plt.xlabel('Mês/Ano')
    plt.ylabel('Número de Ocorrências')
    plt.xticks(rotation=45)
    plt.savefig('static/graficos/new/ocorrencias_mes_tres1.png')  # Salvar o boxplot como um arquivo de imagem
    plt.show()

    # # Heatmap de Correlação
    # plt.figure(figsize=(12, 8))
    # correlation_matrix = df2.corr()
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # plt.title('Heatmap de Correlação')
    # plt.savefig('static/graficos/new/heatmap_correlacao_tres1.png')  # Salvar o heatmap como um arquivo de imagem
    # plt.show()

    plt.figure(figsize=(12, 6))
    df2['tp_ocor'].value_counts().plot(kind='bar')
    plt.title('Frequência de tp_ocor')
    plt.savefig('static/graficos/new/barras_tp_ocor_tres1.png')  # Salvar o gráfico de barras como um arquivo de imagem
    plt.show()

    plt.figure(figsize=(12, 6))
    df2['Situacao'].value_counts().plot(kind='bar')
    plt.title('Frequência de Situacao')
    plt.savefig('static/graficos/new/barras_situacao_tres1.png')  # Salvar o gráfico de barras como um arquivo de imagem
    plt.show()

    # Scatter plot entre diasresolucao e VlCusto
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df2, x='diasresolucao', y='VlCusto')
    plt.title('Relação entre Dias de Resolução e VlCusto')
    plt.xlabel('Dias de Resolução')
    plt.ylabel('VlCusto')
    plt.savefig('static/graficos/new/vlcusto_diasresol_tres1.png')
    plt.show()

    return "Processamento concluído e informações exibidas no console."

if __name__ == '__main__':
    analise.run(debug=True)
#import pyodbc
#import numpy as np # linear algebra
#import pandas as pd # data manipulation and analysis
#import matplotlib.pyplot as plt # data visualization
#import seaborn as sns # data visualization
#from time import time
#from sklearn import metrics
#from sklearn.cluster import KMeans
#from sklearn import preprocessing
#from sklearn.preprocessing import scale
#from sklearn.decomposition import PCA
#from sqlalchemy import create_engine

#from flask import Flask, render_template
#app = Flask(__name__)


# Informações da conexão
#server = '10.0.0.14'  # Exemplo: 'localhost'
#database = 'softran_rasador'  
#username = 'softran'  
#password = 'sof1209'  

# String de conexão usando SQLAlchemy
#connection_str = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=SQL+Server'

# Criando o motor de conexão
#engine = create_engine(connection_str)

# Comandos SQL
# Função para ler o conteúdo SQL do arquivo
#def ler_sql_do_arquivo(nome_do_arquivo):
#    with open(nome_do_arquivo, 'r') as arquivo:
#        return arquivo.read()

# Lendo os comandos SQL dos arquivos
#sql_comando1 = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC.sql")
#sql_comando2 = ler_sql_do_arquivo("C:\\Users\\juare\\Desktop\\TCC\\Dados TCC Plus.sql")

# Executando os comandos e lendo os dados para DataFrames
#df1 = pd.read_sql(sql_comando1, engine)
#df2 = pd.read_sql(sql_comando2, engine)
#df1_copia = pd.read_sql(sql_comando1, engine)
#df2_copia = pd.read_sql(sql_comando2, engine)

# Fechando a conexão
#engine.dispose()

# Exibindo os primeiros registros para verificar
# print("Resultado 1:")
# print(df1.head())
# print("\nResultado 2:")
# print(df2.head())

#def index_of_dic(dic, key):
#    return dic[key]

#def StrList_to_UniqueIndexList(lista):
#    group = set(lista)
#    print(group)

 #   dic = {}
 #   i = 0
 #   for g in group:
 #       if g not in dic:
 #           dic[g] = i
 #           i += 1

  #  print(dic)
  #  return [index_of_dic(dic, p) for p in lista]

# Supondo que 'df1' é o seu DataFrame
#df1['Filial'] = StrList_to_UniqueIndexList(df1['Filial'])

# Exibindo as primeiras linhas dos DataFrames#
#print("\nDados (Colunas):")
#print(df1.columns)
#print("Dados (original):")
#print(df1_copia.head(5))  
#print("\nDados (modificado):")
#print(df1.head(5))
#print("\nDados (Infos Variáveis):")
#print(df1.info())
#print("\nDados (Shape):")
#print(df1.shape)
#print("\nDados (Describe):")
#print(df1.describe())
#print("\nDados (Describe = include 0):")
#print(df1.describe(include='O'))

#Data Cleaning
#print("\nDados (Limpeza):")
#print(df1.isnull().sum())

#Data outliers
#df1[df1.duplicated(keep='first')]
#df1.drop_duplicates(keep='first',inplace=True)

#Filial versus conferencia de carregamento
#Charges
#plt.figure(figsize=(10,6))
#sns.distplot(df1.Filial,color='r')
#plt.title('Distribuição Filial - Carregamento',size=18)
#plt.xlabel('Filial',size=14)
#plt.ylabel('conf_carregamento',size=14)
#plt.show()

#Age
#plt.figure(figsize=(10,6))
#sns.histplot(df1.Filial)
#plt.title('Distribuição Filial - Carregamento',size=18)
#plt.xlabel('Filial',size=14)
#plt.ylabel('conf_carregamento',size=14)
#plt.show()

#BMI
#plt.figure(figsize=(10,6))
#plt.hist(df1.Filial,color='y')
#plt.title('Distribuição Filial',size=18)
#plt.show()

#BMI
#plt.figure(figsize=(10,6))
#plt.hist(df1.conf_carregamento,color='y')
#plt.title('Distribuição conf_carregamento',size=18)
#plt.show()

#Boxplot
#plt.figure(figsize = (10,6))
#sns.boxplot(df1.Filial)
#plt.title('Distribution Filial',size=18)
#plt.show()
#Q1 = df1['Filial'].quantile(0.25)
#Q3 = df1['Filial'].quantile(0.75)
#IQR = Q3 - Q1
#print(IQR)
#df1[(df1['Filial']< Q1-1.5* IQR) | (df1['Filial']> Q3+1.5* IQR)]

#Boxplot
#plt.figure(figsize = (10,6))
#sns.boxplot(df1.conf_carregamento)
#plt.title('Distribution conf_carregamento',size=18)
#plt.show()
#Q1 = df1['conf_carregamento'].quantile(0.25)
#Q3 = df1['conf_carregamento'].quantile(0.75)
#IQR = Q3 - Q1
#print(IQR)
#df1[(df1['conf_carregamento']< Q1-1.5* IQR) | (df1['conf_carregamento']> Q3+1.5* IQR)]

#Gender
#plt.figure(figsize=(10,6))
#sns.countplot(x = 'conf_carregamento', data = df1)
#plt.title('Distribution conf_carregamento',size=18)
#plt.xlabel('conf_carregamento',size=14)
#plt.show()

#Gender
#plt.figure(figsize=(10,6))
#sns.countplot(x = 'Filial', data = df1)
#plt.title('Distribution Filial',size=18)
#plt.xlabel('Filial',size=14)
#plt.show()

#Count
#plt.figure(figsize = (10,6))
#sns.countplot(df1.conf_carregamento)
#plt.title('Distribution conf_carregamento',size=18)
#plt.xlabel('conf_carregamento',size=14)
#plt.ylabel('Count',size=14)
#plt.show()

#Count
#plt.figure(figsize = (10,6))
#sns.countplot(df1.Filial)
#plt.title('Distribution Filial',size=18)
#plt.xlabel('Filial',size=14)
#plt.ylabel('Count',size=14)
#plt.show()

#df1.Filial.value_counts()

#Scatter
#plt.figure(figsize = (10,6))
#sns.scatterplot(x='Filial',y='conf_carregamento',color='r',data=df1)
##plt.title('Filial vs Conf_carregamento',size=18)
#p#lt.xlabel('Filial',size=14)
#plt.ylabel('conf_carregamento',size=14)
#plt.show()

print('Correlation between Filial and Conf_carregamento is : {}'.format(round(df1.corr()['Filial']['Conf_carregamento'],3)))

#plt.figure(figsize = (10,6))
#sns.set_style('darkgrid')
#sns.boxplot(x='Filial',y='conf_carregamento',data=df)
#plt.title('Filial vs Conf_carregamento',size=18);

#Correlation Map
#plt.figure(figsize = (10,6))
#sns.heatmap(df1.corr(),annot=True,square=True,
 #           cmap='RdBu',
 #           vmax=1,
 #           vmin=-1)
#plt.title('Correlations Between Variables',size=18);
#plt.xticks(size=13)
#plt.yticks(size=13)
#plt.show()

#Using Pairplot for Numerical Values
#sns.pairplot(df1, 
#                 markers="+",
#                 diag_kind="kde",
#                 kind='reg',
#                 plot_kws={'line_kws':{'color':'#aec6cf'}, 
#                           'scatter_kws': {'alpha': 0.7, 
#                                           'color': 'red'}},
#                 corner=True);


#Filial versus conferencia de entrega
#Charges
#plt.figure(figsize=(10,6))
#sns.distplot(df1.Filial,color='r')
#plt.title('Distribuição Filial - Entrega',size=18)
#plt.xlabel('Filial',size=14)
#plt.ylabel('conf_entrega',size=14)
#plt.show()

#Age
#plt.figure(figsize=(10,6))
#sns.histplot(df1.Filial)
#plt.title('Distribuição Filial - Entrega',size=18)
#plt.xlabel('Filial',size=14)
#plt.ylabel('conf_entrega',size=14)
#plt.show()

#BMI
#plt.figure(figsize=(10,6))
#sns.histplot(df1.Filial)
#plt.title('Distribuição Filial - Entrega',size=18)
#plt.xlabel('Filial',size=14)
#plt.ylabel('conf_entrega',size=14)
#plt.show()

#Boxplot
#plt.figure(figsize = (10,6))
#sns.boxplot(df1.Filial)
#plt.title('Distribution conf_entrega',size=18)
##plt.show()
#Q1 = df1['conf_entrega'].quantile(0.25)
#Q3 = df1['conf_entrega'].quantile(0.75)
#IQR = Q3 - Q1
#print(IQR)
#df1[(df1['conf_entrega']< Q1-1.5* IQR) | (df1['conf_entrega']> Q3+1.5* IQR)]

#Gender
#plt.figure(figsize=(10,6))
#sns.countplot(x = 'Filial', data = df1)
#plt.title('Distribution Conferencia entrega',size=18)
##plt.xlabel('conf_entrega',size=14)
#plt.show()

#Count
plt.figure(figsize = (10,6))
sns.countplot(df1.Filial)
plt.title('Distribution Conferencia entrega',size=18)
plt.xlabel('conf_entrega',size=14)
plt.ylabel('Count',size=14)
plt.show()

df1.Filial.value_counts()

#Scatter
plt.figure(figsize = (10,6))
sns.scatterplot(x='Filial',y='conf_entrega',color='r',data=df1)
plt.title('Filial vs conf_entrega',size=18)
plt.xlabel('Filial',size=14)
plt.ylabel('conf_entrega',size=14)
plt.show()

print('Correlation between Filial and conf_entrega is : {}'.format(round(df1.corr()['Filial']['conf_entrega'],3)))

plt.figure(figsize = (10,6))
sns.set_style('darkgrid')
sns.boxplot(x='Filial',y='conf_entrega',data=df)
plt.title('Filial vs conf_entrega',size=18);

#Correlation Map
plt.figure(figsize = (10,6))
sns.heatmap(df1.corr(),annot=True,square=True,
            cmap='RdBu',
            vmax=1,
            vmin=-1)
plt.title('Correlations Between Variables',size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()

#Using Pairplot for Numerical Values
sns.pairplot(df1, 
                 markers="+",
                 diag_kind="kde",
                 kind='reg',
                 plot_kws={'line_kws':{'color':'#aec6cf'}, 
                           'scatter_kws': {'alpha': 0.7, 
                                           'color': 'red'}},
                 corner=True);


#def index_of_dic1(dic1, key1):
 #     return dic1[key1]

#def StrList_to_UniqueIndexList1(lista):
      #group = set(lista)
      #print(group)#

      #dic1 = {}
      #i = 0
      #for g in group:
     #     if g not in dic1:
    #          dic1[g] = i
   #           i += 1

  #    print(dic1)
 #     return [index_of_dic1(dic1, p) for p in lista]

# Supondo que 'df1' é o seu DataFrame
#df2['DsTpVeiculo'] = StrList_to_UniqueIndexList1(df2['DsTpVeiculo'])

#def index_of_dic2(dic2, key2):
 #    return dic2[key2]

#def StrList_to_UniqueIndexList2(lista):
     #group = set(lista)
     #print(group)

    # dic2 = {}
    # i = 0
    # for g in group:
     #    if g not in dic2:
   #          dic2[g] = i
    #         i += 1

  #   print(dic2)
 #    return [index_of_dic2(dic2, p) for p in lista]

# Supondo que 'df1' é o seu DataFrame
#df2['DsModelo'] = StrList_to_UniqueIndexList2(df2['DsModelo'])

# Exibindo as primeiras linhas dos DataFrames
#print("\nDados (Colunas):")
#print(df2.columns)
##print("\nDados (original):")
#print(df2_copia.head(5))  
#print("\nDados (modificado):")
#print(df2.head(5))
#print("\nDados (Infos Variáveis):")
#print(df2.info())
#print("\nDados (Shape):")
#print(df2.shape)
#print("\nDados (Describe):")
#print(df2.describe())
#print("\nDados (Tipos):")
#print(df2.dtypes)
# print("\nDados (Describe = include 0):")
# print(df2.describe(include='O'))
# Removi os dois acima pois nao temos colunas dos tipos object

#Data Cleaning
##rint("\nDados (Limpeza):")
#print(df2.isnull().sum())

#Data outliers
#df2[df2.duplicated(keep='first')]
#df2.drop_duplicates(keep='first',inplace=True)

# # Acessando a 20ª linha do DataFrame
# linha_20 = df1.values[19,:]
# print("Linha 20 do DataFrame:")
# print(linha_20)

# # Acessando todos os valores da quarta coluna do DataFrame
# coluna_4 = df1.values[:,3]
# print("\nValores da quarta coluna do DataFrame:")
# print(coluna_4)

# Soma_distancia_quadratica = []
# K = range(1,10)
# for k in K:
#     km = KMeans(n_clusters=k)
#     km = km.fit(df1)
#     Soma_distancia_quadratica.append(km.inertia_)
# print('Distâncias totais:')
# print(Soma_distancia_quadratica)
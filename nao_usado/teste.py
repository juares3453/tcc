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

csv_filepath = 'C:\\Users\\juare\\Desktop\\TCC\\df.csv'
csv_filepath1 = 'C:\\Users\\juare\\Desktop\\TCC\\df1.csv'
csv_filepath2 = 'C:\\Users\\juare\\Desktop\\TCC\\df2.csv'

df = pd.read_csv(csv_filepath, encoding='cp1252', delimiter=';')
print(df.columns)
import os
import re
import sys
import pickle
import shutil
import chardet  # Detecta encode do arquivo
import numpy as np
from math import *
import pandas as pd
import seaborn as sns
from scipy.stats import *
import matplotlib.cm as cm
import plotly.express as PX
import statsmodels.api as sm
from sklearn.metrics import *
from unidecode import unidecode  # Útil para retirar cacteres especiais (como acentos) das palavras
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


""" # Adicionar esse trecho no arquivo .py e descomentar se quiser executar diretamente do terminal
if len(sys.argv) > 1:
    if os.path.isfile(sys.argv[1]):
        file = sys.argv[1]
    elif os.path.isfile(os.path.join('Bases', sys.argv[1])):
        file = os.path.join('Bases', sys.argv[1])
    else:
        raise Exception("Nome de arquivo não válido")
else:
    raise Exception("Sem nome de arquivo")
"""


# Check o encode do arquivo/base de dados
def read_data(file_t):
    pd.set_option('display.max_columns', None) # Mostrar todas as colunas
    pd.set_option('display.max_rows', 10) # Mostrar apenas 10 linhas

    with open(file_t, 'rb') as check_raw:
        raw_data = check_raw.readline()
        encode = chardet.detect(raw_data).get('encoding')
        if encode == 'ascii':
            encode = 'utf-8'

    data = pd.read_csv(file_t, encoding=encode, sep=';', on_bad_lines='skip')
    print('---------------------------------------')
    print('Base: {}'.format(file_t[6:-4]))
    print("Tamanho da base: {}".format(len(data)))

    del encode
    return data

# Padronizar o nome dos sintomas, tirando acentos e espaços.
def standarding_features(raw_data_t, *features, **kwargs):
    coluna = kwargs['coluna']
    features_list = []
    if not raw_data_t[coluna].strip():
        for i in features:
            raw_data_t[i] = 0 # Remover aqui se for para não preencher com 0 em valores faltosos
        return raw_data_t
    else:
        for i in features:
            if re.search(i, unidecode(raw_data_t[coluna]).strip().lower()):# Retira todos os acentos e caracteres especiais
                features_list.append(i)
                raw_data_t[i] = 1
            else:
                raw_data_t[i] = 0 # Remover aqui se for para não preencher com 0 em valores faltosos
    raw_data_t[coluna] = ','.join(features_list)
    return raw_data_t

# Útil para salvar informações de agrupamento de uma coluna em arquivo txt
def save_value_counts_file(data_t, column_name_t):
    file_name = '{}.txt'.format(column_name_t)
    with open(file_name, 'w') as file_value:
        file_value.write(data_t[column_name_t].value_counts().to_string())

# Útil para adicionar colunas, função usada para adicionar as colunas de cada sintoma
def add_column(data_t, columns_t):
    for i, column in enumerate(columns_t):
        data_t.insert((i + 3), column, '', True)

# Útil para arredondar números, ainda mais em escalas inteiras
def round_up(number_t, decimals_t=0):
    multiplier = 10 ** decimals_t
    return ceil(number_t * multiplier) / multiplier

# Salvando arquivo
def saving_data(data_t, file_t):
    data_t.to_csv(file_t, sep=';', encoding='utf-8', index=False)
    print("Salvando arquivo")
    print("Tamanho da base: {}".format(len(data_t)))

# Salvando o modelo da regressão
def saving_model(model_t, name_model_t):
    model_file = 'Modelos/{}.pkl'.format(name_model_t)

    with open(model_file, 'wb') as file:
        pickle.dump(model_t, file)

# Carregando o modelo da regressão
def loading_model(name_model_t):
    model_file = 'Modelos/{}.pkl'.format(name_model_t)

    with open(model_file, 'rb') as file:
        model_loaded = pickle.load(file)

    return model_loaded

# Balanceamento dos dados
def dataset_balancing(data_t, features_t, kind_t):
    if kind_t == 'Unbalanced':
        return data_t
    else:
        data_positives = data_t[data_t['resultadoTeste'] == 1]
        data_negatives = data_t[data_t['resultadoTeste'] == 0]

    if kind_t == 'Symptoms_Based':
        data_positives = data_positives[(data_positives['olfativos'] == 1) | (data_positives['gustativos'] == 1)]
    elif kind_t == 'Symptoms_Amount':
        data_positives = data_positives[data_positives[features_t].sum(axis=1) > 3]
        data_positives = data_positives[(data_positives['olfativos'] == 1) | (data_positives['gustativos'] == 1)]

    if len(data_positives) > len(data_negatives):
        data_majority = data_positives
        data_minority = data_negatives
    else:
        data_majority = data_negatives
        data_minority = data_positives

    data_size = round((len(data_minority) * 2) * 0.60)
    data_reduced = resample(data_majority, replace=True, n_samples=data_size, random_state=123)
    data_balanced = pd.concat([data_reduced, data_minority])

    return data_balanced

# Cria diretório para guardar dados
def directory_create(directory_t):
    if not os.path.exists(directory_t):
        os.mkdir(directory_t)

def grouping_datasets(datasets_t):
    data = read_data('Bases/{}'.format(datasets_t[0]))

    if len(datasets_t) == 1:
        return datasets_t
    else:
        for i in range(1, len(datasets_t)):
            data_temp = read_data('Bases/{}'.format(datasets_t[i]))
            data = pd.concat([data, data_temp], ignore_index=True)
            print('Tamanho da base concatenada: {}'.format(len(data)))

        for dataset in datasets_t:
            os.remove('Bases/{}'.format(dataset))

        saving_data(data, 'Bases/base_concatenada.csv')

        datasets_t = os.listdir('Bases/')
        datasets_t = [name for name in datasets_t if name[-3:] == 'csv']

        return datasets_t

# Limpa todas as amostras que não possuem nenhuma comorbidade
def clean_without_conditions(data_t):
    data_t.drop(data_t[(data_t['cardiacas'] == 0) & (data_t['diabetes'] == 0) & (data_t['respiratorias'] == 0) &
                       (data_t['renais'] == 0) & (data_t['imunologica'] == 0) & (data_t['obesidade'] == 0) &
                       (data_t['imunossupressao'] == 0)].index, inplace=True)

    return data_t
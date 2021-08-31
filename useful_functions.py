import os
import re
import sys
import chardet  # Detecta encode do arquivo
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from unidecode import unidecode  # Útil para retirar cacteres especiais (como acentos) das palavras


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
def read_data(file_t='dados-ce-1.csv'):
    pd.set_option('display.max_columns', None) # Mostrar todas as colunas
    pd.set_option('display.max_rows', 10) # Mostrar apenas 10 linhas
    file = 'Bases/{}'.format(file_t)

    with open(file, 'rb') as check_raw:
        raw_data = check_raw.readline()
        encode = chardet.detect(raw_data).get('encoding')
        if encode == 'ascii':
            encode = 'utf-8'

    data = pd.read_csv(file, encoding=encode, sep=';')
    del encode
    return data


# Remover amostras com dados importantes como faltosos, as entradas são colunas para analizar
def delete_rows_NA_values(data_t, columns_analyze_t):
    data_t.dropna(how='any', subset=columns_analyze_t, inplace=True)


# Deletando algumas colunas irrelevantes
def delete_columns(data_t, columns_list_t):
    data_t.drop(columns_list_t, axis=1, inplace=True)


# Deleta amostras (rows) com valores diferentes (ou iguais) ao do parâmetro 'value', se baseando na columa 'column'
def delete_rows_by_value(data_t, column_t, value_t, diff_t=True):
    if diff_t:
        data_t.drop(data_t[data_t[column_t] != value_t].index, inplace=True)
    else:
        data_t.drop(data_t[data_t[column_t] == value_t].index, inplace=True)


# Alterar valores de colunas, essencialmente ajudar a tirar caracteres especiais ou palavras mal formadas
def change_column_values(data_t, column_t, regex_exp_t, new_value_t, use_regex_t=True):
    #print(data[column].value_counts())
    data_t[column_t].replace(to_replace=regex_exp_t, value=new_value_t, regex=use_regex_t, inplace=True)
    #print(data[column].value_counts())


# Formatando datas, colocando no padrão yyyy-mm-dd
def formating_dates(raw_date_t):
    return re.sub(r'(\d{4})-(\d{2})-(\d{2})(.*)', r'\1-\2-\3', raw_date_t)


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

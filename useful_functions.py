import os
import re
import pickle
import chardet  # Detecta encode do arquivo
from math import *
import pandas as pd
from unidecode import unidecode  # Útil para retirar cacteres especiais (como acentos) das palavras
from sklearn.utils import resample


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
    pd.set_option('display.max_columns', None)  # Mostrar todas as colunas
    pd.set_option('display.max_rows', 10)  # Mostrar apenas 10 linhas

    with open(file_t, 'rb') as check_raw:
        raw_data = check_raw.readline()
        encode = chardet.detect(raw_data).get('encoding')
        if encode == 'ascii':
            encode = 'utf-8'

    data = pd.read_csv(file_t, encoding=encode, sep=None, on_bad_lines='skip', engine='python')
    print('---------------------------------------')
    print('Base: {}'.format(file_t[6:-4]))
    print("Tamanho da base: {}".format(len(data)))

    del encode
    return data


def dates_filling(raw_df_t, *columns):
    if pd.isnull(raw_df_t.loc[columns[0]]) & pd.notnull(raw_df_t.loc[columns[1]]):
        raw_df_t[columns[0]] = raw_df_t[columns[1]]
    elif pd.notnull(raw_df_t.loc[columns[0]]) & pd.isnull(raw_df_t.loc[columns[1]]):
        raw_df_t[columns[1]] = raw_df_t[columns[0]]
    elif pd.isnull(raw_df_t.loc[columns[0]]) & pd.isnull(raw_df_t.loc[columns[1]]):
        raw_df_t[columns[0]] = pd.to_datetime('today', format='%Y-%m-%d')
        raw_df_t[columns[1]] = pd.to_datetime('today', format='%Y-%m-%d')

    return raw_df_t


# Padronizar o nome dos sintomas, tirando acentos e espaços.
def standarding_features(raw_df_t, *features, **kwargs):
    column = kwargs['key']

    if not raw_df_t[column].strip():
        for i in features:
            raw_df_t[i] = 0  # Remover aqui se for para não preencher com 0 em valores faltosos
        return raw_df_t
    else:
        for i in features:
            if re.search(i, unidecode(raw_df_t[column]).strip().lower()):  # Retira todos os caracteres especiais
                raw_df_t[i] = 1
            else:
                raw_df_t[i] = 0  # Remover aqui se for para não preencher com 0 em valores faltosos

    return raw_df_t


# ffdsfsdfsdfsd
def standarding_features_with_dict(raw_df_t, *column, **kwargs):
    kwargs = kwargs['key']

    if not raw_df_t[column[0]].strip():
        for key in kwargs.keys():
            if not raw_df_t[key]:
                raw_df_t[key] = 0  # Remover aqui se for para não preencher com 0 em valores faltosos
        return raw_df_t
    else:
        for key in kwargs.keys():
            for value in kwargs[key]:
                if raw_df_t[key] == 1:
                    break
                elif re.search(value, unidecode(str(raw_df_t[column[0]])).strip().lower()):  # Retira caract. especiais
                    raw_df_t[key] = 1
                    break
                else:
                    raw_df_t[key] = 0  # Remover aqui se for para não preencher com 0 em valores faltosos

    return raw_df_t


# Útil para salvar informações de agrupamento de uma coluna em arquivo txt
def save_value_counts_file(df_t, column_name_t):
    file_name = '{}.txt'.format(column_name_t)

    with open(file_name, 'w') as file_value:
        file_value.write(df_t[column_name_t].value_counts().to_string())


# Útil para adicionar colunas, função usada para adicionar as colunas de cada sintoma
def add_column(df_t, columns_t):
    for i, column in enumerate(columns_t):
        df_t.insert((i + 1), column, '', True)


# Útil para arredondar números, ainda mais em escalas inteiras
def round_up(number_t):
    decimal = (len(str(number_t)) * -1) + 1

    if decimal == 0:
        decimal = -1

    multiplier = 10 ** decimal
    final_number = ceil(number_t * multiplier) / multiplier

    return round(final_number)


# Salvando arquivo
def saving_data(df_t, file_t):
    df_t.to_csv(file_t, sep=';', encoding='utf-8', index=False)
    print("Salvando arquivo")
    print("Tamanho da base: {}".format(len(df_t)))


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
def dataset_balancing(df_t, features_t, kind_t):
    if kind_t == 'Unbalanced':
        return df_t
    else:
        df_positives = df_t[df_t['resultadoTeste'] == 1]
        df_negatives = df_t[df_t['resultadoTeste'] == 0]

    if kind_t == 'Symptoms_Amount':
        df_positives = df_positives[df_positives[features_t].sum(axis=1) > 3]
        df_positives = df_positives[(df_positives['anosmia'] == 1) | (df_positives['hipogeusia'] == 1)]

    if len(df_positives) > len(df_negatives):
        df_majority = df_positives
        df_minority = df_negatives
    else:
        df_majority = df_negatives
        df_minority = df_positives

    df_size = round((len(df_minority) * 2) * 0.60)
    df_reduced = resample(df_majority, replace=True, n_samples=df_size, random_state=123)
    df_balanced = pd.concat([df_reduced, df_minority])

    return df_balanced


def dataset_conditions_balancing(df_t, cases_t):
    df_t = df_t[df_t['resultadoTeste'] == 1]

    df_case_1 = df_t[df_t['evolucaoCaso_{}'.format(cases_t[1])] == 0]
    df_case_2 = df_t[df_t['evolucaoCaso_{}'.format(cases_t[1])] == 1]

    if len(df_case_1) > len(df_case_2):
        df_majority = df_case_1
        df_minority = df_case_2
    else:
        df_majority = df_case_2
        df_minority = df_case_1

    df_size = round((len(df_minority) * 2) * 0.60)
    df_reduced = resample(df_majority, replace=True, n_samples=df_size, random_state=123)
    df_balanced = pd.concat([df_reduced, df_minority])

    return df_balanced


def get_files_names(base_dir_t, grouping=False):
    files = os.listdir(base_dir_t)
    files = [name for name in files if name[-3:] == 'csv']

    if grouping:
        files = grouping_datasets(files)

    return files


# Cria diretório para guardar dados
def directory_create(directory_list_t):
    for directory in directory_list_t:
        if not os.path.exists(directory):
            os.mkdir(directory)


def grouping_datasets(df_files_t):
    df = read_data('Bases/{}'.format(df_files_t[0]))

    if len(df_files_t) == 1:
        return df_files_t
    else:
        for i in range(1, len(df_files_t)):
            df_temp = read_data('Bases/{}'.format(df_files_t[i]))
            df = pd.concat([df, df_temp], ignore_index=True)
            print('Tamanho da base concatenada: {}'.format(len(df)))

        name_temp = df_files_t[0][:3]

        for dataset in df_files_t:
            os.remove('Bases/{}'.format(dataset))

        saving_data(df, 'Bases/{}ALL.csv'.format(name_temp))

        df_files_t = os.listdir('Bases/')
        df_files_t = [name for name in df_files_t if name[-3:] == 'csv']

        return df_files_t


def clean_features(df_t, datas_t, symptoms_t=False, conditions_t=False):
    if symptoms_t:
        df_t.drop(df_t[(df_t[datas_t['symptoms']] == 0).all(axis=1)].index, inplace=True)

    if conditions_t:
        df_t.drop(df_t[(df_t[datas_t['conditions']] == 0).all(axis=1)].index, inplace=True)


def convert_to_categorical(features_t):
    ages = [12, 18, 30, 45, 60, 75, 90, 121]
    days = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']

    for i, age in enumerate(ages):
        if features_t['idade'] < age:
            features_t['idade'] = categories[i]
            break

    for i, days in enumerate(days):
        if features_t['diasSintomas'] <= days:
            features_t['diasSintomas'] = categories[i]
            break

    return features_t

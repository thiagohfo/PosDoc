"""
Arquivo utilizado com bases atualizadas até o abril de 2022.
Não válido para as bases atualizadas em maio de 2022.
"""
import os
import numpy as np
import pandas as pd
from useful_functions import dates_filling
from useful_functions import read_data, add_column, standarding_features, standarding_features_with_dict, saving_data


# Pré-processamento dos datasets
def preprocessing(file_t):
    df = read_data(file_t)

    # Colunas para analisar os dados e deletar as amostras (rows)
    columns_analyze = ['evolucaoCaso', 'sintomas', 'tipoTeste', 'resultadoTeste', 'classificacaoFinal']

    # Colunas para deletar
    columns_del = ['id', 'estadoNotificacao', 'estadoNotificacaoIBGE', 'municipioNotificacao',
                   'municipioNotificacaoIBGE', 'profissionalSaude', 'profissionalSeguranca', 'cbo', 'sexo', 'racaCor',
                   'estado', 'estadoIBGE', 'municipio', 'municipioIBGE', 'dataNotificacao', 'dataInicioSintomas',
                   'estadoTeste', 'dataTeste', 'dataEncerramento', 'cnes']

    # Sintomas e condições para virarem colunas
    datas = {'other_symptoms': {'mialgia': ['mialgia', 'corpo', 'muscular'],
                                'cabeca': ['cefaleia', 'cabeca'],
                                'fadiga': ['fadiga', 'cansaco', 'fraqueza', 'indisposicao', 'adinamia', 'moleza',
                                           'astenia'],
                                'nauseas': ['nausea', 'nauseas', 'tontura', 'mal estar', 'enjoo'],
                                'coriza': ['coriza'],
                                'anosmia': ['olfato', 'anosmia', 'hiposmia'],
                                'hipogeusia': ['paladar', 'hipogeusia', 'disgeusia'],
                                'diarreia': ['diarreia'], 'febre': ['febre', 'febril'],
                                'tosse': ['tosse']
                                },
             'symptoms': ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'hipogeusia', 'anosmia',
                          'fadiga', 'nauseas', 'mialgia', 'diarreia'],
             'conditions': ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade',
                            'imunossupressao', 'gestante', 'puerpera']
             }

    # Deletando alguns rows com base nos valores nulos ou diferentes do necessário para se trabalhar
    df.dropna(how='any', subset=columns_analyze, inplace=True)
    df['tipoTeste'].replace(to_replace='TESTE RÁPIDO - ANTÍGENO', value='RT-PCR', inplace=True)
    df.drop(df[df['tipoTeste'] != 'RT-PCR'].index, inplace=True)
    df.drop(df[~df['classificacaoFinal'].str.contains(r'Confirmado')].index, inplace=True)
    df.drop(df[df['resultadoTeste'].str.contains(r'^[In].*')].index, inplace=True)
    df.drop(df[(df['evolucaoCaso'] == 'Cancelado') | (df['evolucaoCaso'] == 'Ignorado')].index, inplace=True)
    del columns_analyze

    # Correção, muda valores de null para empty strings.
    df['condicoes'].fillna('', inplace=True)
    df['outrosSintomas'].fillna('', inplace=True)

    # Consertando um probleminha de datas com valores ausentes para poder aplicar funções
    df = df.apply(dates_filling, args=['dataNotificacao', 'dataInicioSintomas'], axis=1)
    data_notificacao = pd.to_datetime(df['dataNotificacao'], errors='coerce', format='%Y-%m-%d')
    data_inicio_sintomas = pd.to_datetime(df['dataInicioSintomas'], errors='coerce', format='%Y-%m-%d')
    df.insert(0, "diasSintomas", (data_notificacao - data_inicio_sintomas).dt.days, True)
    df.loc[df['diasSintomas'] < 0, 'diasSintomas'] = df['diasSintomas'] * -1
    df.drop(df[df['diasSintomas'] > 90].index, inplace=True)

    # Deletando atributos irrelevantes para análise
    df.drop(columns_del, axis=1, inplace=True)
    del columns_del
    del data_inicio_sintomas, data_notificacao

    # Adiciona os sintomas e condicoes como colunas no dataframe, cada sintoma/condição vira 1 coluna
    add_column(df, datas['symptoms'] + [col for col in datas['other_symptoms'].keys() if col not in datas['symptoms']] +
               datas['conditions'])

    # Preenche as colunas dos sintomas e condicoes com 1
    for feature, column in zip([datas['symptoms'], datas['conditions']], ['sintomas', 'condicoes']):
        df = df.apply(standarding_features, axis=1, args=feature, key=column)

    df = df.apply(standarding_features_with_dict, axis=1, args=['outrosSintomas'], key=datas['other_symptoms'])
    df.drop(['sintomas', 'condicoes', 'outrosSintomas'], axis=1, inplace=True)
    df.drop(df[(df[datas['symptoms']] == 0).all(axis=1)].index, inplace=True)

    # Trocar alguns valores
    cols_change = ['resultadoTeste', 'evolucaoCaso']  # Índice 0 no zip
    old_values = [['Negativo', 'Positivo'], [r'.*tratamento.*', r'Internado.*']]  # Índice 1 no zip
    new_values = [[0, 1], ['Cura', 'Internado']]  # Índice 2 no zip
    use_regex = [False, True]  # Índice 3 no zip

    for values in zip(cols_change, old_values, new_values, use_regex):
        df[values[0]].replace(to_replace=values[1][0], value=values[2][0], regex=values[3], inplace=True)
        df[values[0]].replace(to_replace=values[1][1], value=values[2][1], regex=values[3], inplace=True)

    del cols_change, old_values, new_values, use_regex

    for col in ['idade', 'diasSintomas']:
        df.drop(df[df[col].isin([np.nan, np.inf, -np.inf])].index, inplace=True)
        df[col] = df[col].astype(int)

    # Salvando o arquivo
    saving_data(df, file_t)

    return df


def preprocessing_all():
    files = os.listdir('Bases/')
    files = [name for name in files if name[-3:] == 'csv']

    for name in files:
        preprocessing('Bases/{}'.format(name))

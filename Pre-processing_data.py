#!/opt/anaconda3/envs/PosDoc/bin/python
import os
import re
import sys
import chardet  #Detecta encode do arquivo
import pandas as pd
from useful_functions import *
from collections import Counter
from unidecode import unidecode  #Útil para retirar cacteres especiais (como acentos) das palavras


# Leitura dos dados
file = 'Bases/dados-ce-1.csv'
data = read_data()
print(len(data))


# Colunas para analisar os dados e deletar as amostras
columns_analyze = ['evolucaoCaso', 'sintomas', 'profissionalSaude', 'tipoTeste', 'resultadoTeste']


# Colunas para deletar
columns_del = ['ÿid', 'dataNascimento', 'estadoTeste', 'paisOrigem', 'municipioIBGE', 'origem', 'estadoNotificacao',
                'estadoNotificacaoIBGE', 'municipioNotificacao', 'estado', 'estadoIBGE', 'excluido', 'validado',
                'dataEncerramento', 'dataTeste', 'municipio', 'municipioNotificacaoIBGE', 'cbo', 'dataNotificacao',
                'dataInicioSintomas', 'classificacaoFinal']


# Sintomas e condições para virarem colunas
features = [['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao'],
            ['assintomatico', 'outros', 'tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'gustativos', 'olfativos']]
colunas = ['condicoes', 'sintomas']


delete_rows_NA_values(data, columns_analyze)
delete_rows_by_value(data, 'tipoTeste', 'RT-PCR')
del columns_analyze


# Consertando um probleminha de datas com valores ausentes para poder aplicar funções
data.loc[data.dataNotificacao.isnull(), 'dataNotificacao'] = data['dataInicioSintomas']
data.loc[data.dataInicioSintomas.isnull(), 'dataInicioSintomas'] = data['dataNotificacao']


# Correção, muda valores de null para empty strings.
data['condicoes'] = data['condicoes'].fillna('')


# Chamando função para consertar as datas e aplicar nas colunas correspondentes
data['dataNotificacao'] = data['dataNotificacao'].apply(formating_dates)
data['dataInicioSintomas'] = data['dataInicioSintomas'].apply(formating_dates)


# Realizar operação entre as datas, subtraindo dataInicioSintomas de dataNotificacao e cria a coluna diasSintomas
dataNotificacao = pd.to_datetime(data['dataNotificacao'], format='%Y-%m-%d')
dataInicioSintomas = pd.to_datetime(data['dataInicioSintomas'], format='%Y-%m-%d')
data.insert(2, "diasSintomas", (dataNotificacao - dataInicioSintomas).dt.days, True)


# Deletando atributos irrelevantes para análise
delete_columns(data, columns_del)
del columns_del


# Adiciona os sintomas e condicoes como colunas no dataset, cada sintoma/condição vira 1 coluna
add_column(data, features[1] + features[0])


# Preenche as colunas dos sintomas e condicoes com 1
for feature, coluna in zip(features, colunas):
    data = data.apply(standarding_features, axis=1, args=feature, coluna=coluna)


# Mais algumas colunas e linhas para serem limpas, só podem serem feitas depois de algumas limpezas prévias
delete_rows_by_value(data, 'evolucaoCaso', 'Cancelado', False)
delete_rows_by_value(data, 'evolucaoCaso', 'Ignorado', False)
delete_rows_by_value(data, 'sintomas', 'assintomatico', False)
delete_rows_by_value(data, 'sintomas', 'outros', False)
features_columns = ['sintomas', 'assintomatico', 'condicoes', 'gustativos', 'olfativos', 'outros']
delete_columns(data, features_columns)


# Trocar alguns valores
change_column_values(data, 'profissionalSaude', 'Não', 0, False)
change_column_values(data, 'profissionalSaude', 'Sim', 1, False)
change_column_values(data, 'resultadoTeste', r'^[Ne|In].*', 0, True)
change_column_values(data, 'resultadoTeste', 'Positivo', 1, False)
change_column_values(data, 'sexo', 'Feminino', 0, False)
change_column_values(data, 'sexo', 'Masculino', 1, False)


#Converte colunas selecionadas para int
columns = data.columns.to_list()
columns.remove('evolucaoCaso')
columns.remove('tipoTeste')
data[columns] = data[columns].astype(int)
print(data.dtypes)

data.to_csv(file, sep=';', encoding='utf-8', index=False)

print(len(data))

exit(0)
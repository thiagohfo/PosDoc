#!/opt/anaconda3/envs/PosDoc/bin/python
import os
import re
import sys
import chardet  #Detecta encode do arquivo
import pandas as pd
from collections import Counter
from unidecode import unidecode  #Útil para retirar cacteres especiais (como acentos) das palavras
"""
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

#Mostrar todas as colunas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
file = 'Bases/dados-ce-1.csv'


#Check o encode do arquivo/base de dados
with open(file, 'rb') as check_raw:
    raw_data = check_raw.readline()
    encode = chardet.detect(raw_data).get('encoding')
    if encode == 'ascii':
        encode = 'utf-8'

data = pd.read_csv(file, encoding=encode, sep=';')
del encode


#Remover amostras com dados importantes como faltosos, as entradas são colunas para analizar
def delete_rows_NA_values(columns_analyze):
    columnsToAnalyze = columns_analyze
    data.dropna(how='any', subset=columnsToAnalyze, inplace=True)


#Deletando algumas colunas irrelevantes
def delete_columns(columns_list):
    columnsToDel = columns_list
    data.drop(columnsToDel, axis=1, inplace=True)


#Deleta amostras (rows) com valores diferentes (ou iguais) ao do parâmetro 'value', se baseando na columa 'column'
def delete_rows_by_value(column, value, diff=True):
    if diff:
        indexes = data[data[column] != value].index
    else:
        indexes = data[data[column] == value].index
    data.drop(indexes, inplace=True)


#Alterar valores de colunas, essencialmente ajudar a tirar caracteres especiais ou palavras mal formadas
def change_column_values(column, regex_exp, new_value, use_regex=True):
    #print(data[column].value_counts())
    data[column].replace(to_replace=regex_exp, value=new_value, regex=use_regex, inplace=True)
    #print(data[column].value_counts())


#Formatando datas, colocando no padrão yyyy-mm-dd
def formating_dates(raw_date):
    return re.sub(r'(\d{4})-(\d{2})-(\d{2})(.*)', r'\1-\2-\3', raw_date)


#Padronizar o nome dos sintomas, tirando acentos e espaços.
def standarding_features(raw_data, *features, **coluna):
    coluna = coluna['kwarg']
    features_list = []
    if not raw_data[coluna].strip():
        for i in features:
            raw_data[i] = 0
        return raw_data
    else:
        for i in features:
            if re.search(i, unidecode(raw_data[coluna]).strip().lower()):#Retira todos os acentos e caracteres especiais
                features_list.append(i)
                raw_data[i] = 1
            else:
                raw_data[i] = 0
    raw_data[coluna] = ','.join(features_list)
    return raw_data


#Útil para salvar informações de agrupamento de uma coluna em arquivo txt
def save_value_counts_file(column_name):
    file_name = '{}.txt'.format(column_name)
    with open(file_name, 'w') as file_value:
        file_value.write(data[column_name].value_counts().to_string())


#Útil para adicionar colunas, função usada para adicionar as colunas de cada sintoma
def add_column(columns):
    for i, column in enumerate(columns):
        data.insert((i + 3), column, '', True)

#Colunas para analisar os dados e deletar as amostras
columnsToAnalyze = ['evolucaoCaso', 'sintomas', 'profissionalSaude', 'tipoTeste', 'resultadoTeste']


#Colunas para deletar
columnsToDel = ['ÿid', 'dataNascimento', 'estadoTeste', 'paisOrigem', 'municipioIBGE', 'origem', 'estadoNotificacao',
                'estadoNotificacaoIBGE', 'municipioNotificacao', 'estado', 'estadoIBGE', 'excluido', 'validado',
                'dataEncerramento', 'dataTeste', 'municipio', 'municipioNotificacaoIBGE', 'cbo', 'dataNotificacao',
                'dataInicioSintomas', 'classificacaoFinal']


#Sintomas e condições para virarem colunas
features = [['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao'],
            ['assintomatico', 'outros', 'tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'gustativos', 'olfativos']]
colunas = ['condicoes', 'sintomas']


delete_rows_NA_values(columnsToAnalyze)
delete_rows_by_value('tipoTeste', 'RT-PCR')
del columnsToAnalyze


#Consertando um probleminha de datas com valores ausentes para poder aplicar funções
data.loc[data.dataNotificacao.isnull(), 'dataNotificacao'] = data['dataInicioSintomas']
data.loc[data.dataInicioSintomas.isnull(), 'dataInicioSintomas'] = data['dataNotificacao']
#Correção, muda valores de null para empty strings.
data['condicoes'] = data['condicoes'].fillna('')


#Chamando função para consertar as datas e aplicar nas colunas correspondentes
data['dataNotificacao'] = data['dataNotificacao'].apply(formating_dates)
data['dataInicioSintomas'] = data['dataInicioSintomas'].apply(formating_dates)


#Realizar operação entre as datas, subtraindo dataInicioSintomas de dataNotificacao e cria a coluna diasSintomas
dataNotificacao = pd.to_datetime(data['dataNotificacao'], format='%Y-%m-%d')
dataInicioSintomas = pd.to_datetime(data['dataInicioSintomas'], format='%Y-%m-%d')
data.insert(2, "diasSintomas", (dataNotificacao - dataInicioSintomas).dt.days, True)

delete_columns(columnsToDel)
del columnsToDel


#Adiciona os sintomas e condicoes como coluna no dataset
add_column(features[1] + features[0])


#Preenche as colunas dos sintomas e condicoes com 1
for feature, coluna in zip(features, colunas):
    data = data.apply(standarding_features, axis=1, args=feature, kwarg=coluna)


#Mais algumas colunas e linhas para serem limpas, só podem serem feitas depois de algumas limpezas
delete_rows_by_value('evolucaoCaso', 'Cancelado', False)
delete_rows_by_value('evolucaoCaso', 'Ignorado', False)
delete_rows_by_value('sintomas', 'assintomatico', False)
delete_rows_by_value('sintomas', 'outros', False)
features_columns = ['sintomas', 'assintomatico', 'condicoes', 'gustativos', 'olfativos', 'outros']
delete_columns(features_columns)


#Trocar alguns valores
change_column_values('profissionalSaude', 'Não', 0, False)
change_column_values('profissionalSaude', 'Sim', 1, False)
change_column_values('resultadoTeste', r'^[Ne|In].*', 0, True)
change_column_values('resultadoTeste', 'Positivo', 1, False)
change_column_values('sexo', 'Feminino', 0, False)
change_column_values('sexo', 'Masculino', 1, False)


#Converte colunas selecionadas para int
columns = data.columns.to_list()
columns.remove('evolucaoCaso')
columns.remove('tipoTeste')
data[columns] = data[columns].astype(int)
print(data.dtypes)

data.to_csv(file, sep=';', encoding='utf-8', index=False)

print(len(data))

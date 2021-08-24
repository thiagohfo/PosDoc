import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

#Mostrar todas as colunas e linhas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
file = 'Bases/dados-ce-1.csv'


# Lendo a instância
data = pd.read_csv(file, sep=';')
print(len(data))


# Deletando algumas colunas irrelevantes
def delete_columns(columns_del):
    data.drop(columns_del, axis=1, inplace=True)


# Remover amostras com dados importantes com zeros, a entrada é coluna para analizar
def delete_rows_by_values(columns_analyze, zeros=True):
    if zeros:
        data.drop(data[data[columns_analyze] == 0].index, inplace=True)
    else:
        data.drop(data[data[columns_analyze] == 1].index, inplace=True)
    print(data[columns_analyze].value_counts())


# Predição
def prediction(X, Y, test_size_t=0.3, random_state_t=0):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size_t, random_state=random_state_t)
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    return lm.predict(x_test)

# Variáveis
#conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
conditions = ['cardiacas', 'diabetes', 'renais', 'imunossupressao']
symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']


# Escolhendo o grupo para testes SÓ NEGATIVOS ou SÓ POSITIVOS
delete_rows_by_values('resultadoTeste')


data_size = len(data)


# Analisando a presença de cada sintoma no total de positivos
for i in (symptoms + conditions):
    print(i)
    value = data[i].value_counts(normalize=True).to_string()
    value = [x.split() for x in value.split('\n')]
    for j in range(len(value)):
        print("{} - {:.2f}%".format(value[j][0], float(value[j][1]) * 100))


#
X = data[conditions + symptoms].astype(float)
Y = data['resultadoTeste'].astype(float)


pred = prediction(X, Y)
X = sm.add_constant(X)
ols = sm.OLS(endog=Y, exog=X).fit()
print(ols.summary())

exit(0)
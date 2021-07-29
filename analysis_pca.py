import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
pd.set_option('display.max_rows', None)
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


# Scree Plot
def scree_plot(pca):
    plt.plot(pca.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Influência dos Componentes')
    plt.xlabel('Componente Principal')
    plt.ylabel('Autovalor')
    plt.show()


# Matriz de correlação
def correlation_matrix():
    data_corr = data.loc[:, data.columns[:-1]]
    sns.heatmap(data_corr.corr(), xticklabels=data.columns[:-1], yticklabels=data.columns[:-1], annot=True, fmt='.1f', linewidths=.6, cmap='YlGnBu')
    plt.show()


# Executando o PCA
def pca_apply(n_componentes):
    X = data.loc[:, data.columns[:-1]].values # Passando os dados para o PCA trabalhar
    Y = data.loc[:, ['resultadoTeste']].values # Passando os dados para o PCA trabalhar
    X = StandardScaler().fit_transform(X) # Padronizando valores
    # Aplicando o PCA
    pca = PCA(n_components=n_componentes)
    pca.fit(X)
    pca_transformed = pca.transform(X)
    pca_data = pd.DataFrame(data=pca_transformed, columns=['PCA{}'.format(i) for i in range(1, n_componentes+1)])
    pca_data = pd.concat([pca_data, pd.DataFrame(data=Y, columns=['Target'])], axis=1)
    # Printando dados de cada componente
    print('Value by component: {}'.format(['PCA{}={}'.format(i+1, np.round(value,2)) for i, value in enumerate(pca.explained_variance_ratio_)]))
    print('Cumulative value: {}'.format(['{}={}'.format(i+1, np.round(value,2)) for i, value in enumerate(pca.explained_variance_ratio_.cumsum())]))
    # Plots
    scree_plot(pca)
    correlation_matrix()
    # Deletando as variáveis, só por motivo de clareza
    del pca
    del pca_transformed
    del pca_data


# Variáveis
conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']


# Escolhendo o grupo para testes SÓ NEGATIVOS ou SÓ POSITIVOS
delete_rows_by_values('resultadoTeste')


# Removendo algumas colunas
list_del_cols = conditions + others
for i in list_del_cols:
    delete_columns(i)


# Deletando alguns rows com valores zerados
columns_analyze = [] # Apontar colunas com valores zero para remover row
for i in columns_analyze:
    delete_rows_by_values(columns_analyze)
    columns_analyze.pop()


# Usando o PCA
max = len(data.columns[:-1])
min = 2
for i in range(max, min, -1):
    pca_apply(i)
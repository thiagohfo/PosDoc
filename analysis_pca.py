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


#Lendo a instância
data = pd.read_csv(file, sep=';')
print(len(data))


#Deletando algumas colunas irrelevantes
def delete_columns(columns_del):
    data.drop(columns_del, axis=1, inplace=True)


#Remover amostras com dados importantes como faltosos ou zeros, as entradas são colunas para analizar
def delete_rows_by_values(columns_analyze, all=True):
    if all:
        data[(data[columns_analyze] != 0).all(axis=1)]
    else:
        data[(data[columns_analyze] != 0).any(axis=1)]


#Executando o PCA
def pca_apply(n_componentes, X, Y):
    pca = PCA(n_components=n_componentes)
    pca.fit(X)
    pca_transformed = pca.transform(X)
    pca_data = pd.DataFrame(data=pca_transformed, columns=['PCA{}'.format(i) for i in range(1, n_componentes+1)])
    pca_data = pd.concat([pca_data, pd.DataFrame(data=Y, columns=['Target'])], axis=1)
    print('Value by component: {}'.format(['PCA{}={}'.format(i+1, np.round(value,2)) for i, value in enumerate(pca.explained_variance_ratio_)]))
    print('Cumulative value: {}'.format(['{}={}'.format(i+1, np.round(value,2)) for i, value in enumerate(pca.explained_variance_ratio_.cumsum())]))


conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']


#Escolhendo o grupo para testes SÓ NEGATIVOS ou SÓ POSITIVOS
delete_rows_by_values(['resultadoTeste'])


#Removendo algumas colunas
list_del_cols = conditions + others
for i in list_del_cols:
    delete_columns(i)


#Passando os dados para o PCA trabalhar
X = data.loc[:, data.columns[:-1]].values
Y = data.loc[:, ['resultadoTeste']].values


#Padronizando valores
X = StandardScaler().fit_transform(X)


#Usando o PCA
pca_apply(3, X, Y)


#Deletando alguns rows com valores zerados
columns_analyze = []
for i in columns_analyze:
    delete_rows_by_values(columns_analyze)
    columns_analyze.pop()








#Matriz de correlação
#data_corr = data.loc[:, data.columns[:-1]]
#sns.heatmap(data_corr.corr(), xticklabels=data.columns[:-1], yticklabels=data.columns[:-1], annot=True, fmt='.1f', linewidths=.6, cmap='YlGnBu')
#plt.show()


#pca.fit(X)
"""
plt.plot(pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Influência dos Componentes')
plt.xlabel('Componente Principal')
plt.ylabel('Autovalor')
plt.show()
"""
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
def delete_columns(data, columns_list):
    columnsToDel = columns_list
    return data.drop(columnsToDel, axis=1, inplace=True)


#Remover amostras com dados importantes como faltosos ou zeros, as entradas são colunas para analizar
def delete_rows_by_values(data, columns_analyze, all=True):
    if all:
        return data[(data[columns_analyze] != 0).all(axis=1)]
    else:
        return data[(data[columns_analyze] != 0).any(axis=1)]


conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']


#Escolhendo o grupo para testes SÓ NEGATIVOS ou SÓ POSITIVOS
data = delete_rows_by_values(data, ['resultadoTeste'])


#Removendo algumas colunas
list_del_cols = []
for i in list_del_cols:
    data = delete_columns(data, i)


#Deletando alguns rows com valores zerados
columns_analyze = symptoms
for i in columns_analyze:
    data = delete_rows_by_values(data, columns_analyze)
    columns_analyze.pop()


#Passando os dados para trabalhar
X = data.loc[:, data.columns[:-1]].values
Y = data.loc[:, ['resultadoTeste']].values


#Padronizando valores
X = StandardScaler().fit_transform(X)


#Matriz de correlação
#data_corr = data.loc[:, data.columns[:-1]]
#sns.heatmap(data_corr.corr(), xticklabels=data.columns[:-1], yticklabels=data.columns[:-1], annot=True, fmt='.1f', linewidths=.6, cmap='YlGnBu')
#plt.show()


#Usando o PCA
pca = PCA(n_components=3)
#pca.fit(X)
"""
plt.plot(pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Influência dos Componentes')
plt.xlabel('Componente Principal')
plt.ylabel('Autovalor')
plt.show()
"""


#pca1 = pca.transform(X)[:,0]
#pca2 = pca.transform(X)[:,1]
#pca3 = pca.transform(X)[:,2]


#data['PCA1'] = pca1
#data['PCA2'] = pca2
#data['PCA3'] = pca3


pca_transformed = pca.fit_transform(X)
pca_data = pd.DataFrame(data=pca_transformed, columns=['PCA1', 'PCA2', 'PCA3'])
pca_data_label = pd.DataFrame(data=Y, columns=['Target'])
pca_data_final = pd.concat([pca_data, pca_data_label['Target']], axis=1)

print(np.round(pca.explained_variance_ratio_, 2))
print(pca.explained_variance_ratio_.cumsum())
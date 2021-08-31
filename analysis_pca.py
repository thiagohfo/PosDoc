from plot_functions import *
from useful_functions import *


# Leitura dos dados
file = 'Bases/dados-ce-1.csv'
data = read_data()
print(len(data))


# Executando o PCA
def pca_apply(n_components_t, columns_t):
    X = data[columns_t].values # Passando os dados para o PCA trabalhar
    Y = data['resultadoTeste'].values # Passando os dados para o PCA trabalhar
    #X = StandardScaler().fit_transform(X) # Padronizando valores
    # Aplicando o PCA
    pca = PCA(n_components=n_components_t)
    pca.fit(X)
    X_transformed = pca.transform(X)
    pca_data = pd.DataFrame(data=X_transformed, columns=['PCA{}'.format(i) for i in range(1, n_components_t + 1)])
    pca_data = pd.concat([pca_data, pd.DataFrame(data=Y, columns=['Target'])], axis=1)

    # Printando dados de cada componente
    print('Value by component: {}'.format(['PCA{}={}'.format(i+1, np.round(value,2)) for i, value in enumerate(pca.explained_variance_ratio_)]))
    print('Cumulative value: {}'.format(['{}={}'.format(i+1, np.round(value,2)) for i, value in enumerate(pca.explained_variance_ratio_.cumsum())]))

    # Plots
    for i in pca_data.columns[1:-1]:
        scatter_plot(pca_data, 'PCA1', i)
    scree_plot(pca)

    # Deletando as variáveis, só por motivo de clareza
    del pca
    del X_transformed
    del pca_data


# Variáveis
conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']


# Escolhendo o grupo para testes SÓ NEGATIVOS ou SÓ POSITIVOS
delete_rows_by_value(data, 'resultadoTeste', 0, False)


# Usando o PCA
max = len(symptoms)
min = 2
for i in range(max, min, -1):
    pca_apply(i, (symptoms))


exit(0)
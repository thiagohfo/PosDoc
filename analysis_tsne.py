from plot_functions import *
from useful_functions import *


# Leitura dos dados
file = 'Bases/dados-ce-1.csv'
data = read_data()
print(len(data))


# Executando o t-SNE
def tsne_apply(n_components_t, columns_t):
    X = data[columns_t].values # Passando os dados para o t-SNE trabalhar
    Y = data['resultadoTeste'].values # Passando os dados para o t-SNE trabalhar
    #X = StandardScaler().fit_transform(X) # Padronizando valores
    # Aplicando o t-SNE
    tsne = TSNE(n_components=n_components_t, random_state=0)
    X_transformed = tsne.fit_transform(X)
    tsne_data = pd.DataFrame(data=X_transformed, columns=['t-SNE{}'.format(i) for i in range(1, n_components_t + 1)])
    tsne_data = pd.concat([tsne_data, pd.DataFrame(data=Y, columns=['Target'])], axis=1)


    # Plots
    for i in tsne_data.columns[1:-1]:
        scatter_plot(tsne_data, 't-SNE1', i)


    # Deletando as variáveis, só por motivo de clareza
    del tsne
    del X_transformed
    del tsne_data


# Variáveis
conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']


# Escolhendo o grupo para testes SÓ NEGATIVOS ou SÓ POSITIVOS
delete_rows_by_value(data, 'resultadoTeste', 0, False)


# Usando o t-SNE
max = 3 #len(symptoms)
min = 1
for i in range(max, min, -1):
    tsne_apply(i, (symptoms))


exit(0)
from plot_functions import *
from useful_functions import *


# Executando o t-SNE
def tsne_apply(data_t_x, data_t_y, n_components_t):
    for components in range(n_components_t, 1, -1):
        X = data_t_x.values # Passando os dados para o t-SNE trabalhar
        Y = data_t_y.values # Passando os dados para o t-SNE trabalhar
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
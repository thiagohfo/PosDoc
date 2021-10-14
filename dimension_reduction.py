from plot_functions import *
from useful_functions import *


# Executando o PCA
def pca_apply(data_t_x, data_t_y, n_components_t=0):
    if n_components_t == 0:
        n_components_t = len(data_t_x.columns)

    for components in range(n_components_t, 1, -1):
        X = data_t_x.values # Passando os dados para o PCA trabalhar
        Y = data_t_y.values # Passando os dados para o PCA trabalhar
        #X = StandardScaler().fit_transform(X) # Padronizando valores


        # Aplicando o PCA
        pca = PCA(n_components=components)
        pca.fit(X)
        X_transformed = pca.transform(X)
        pca_data = pd.DataFrame(data=X_transformed, columns=['PCA{}'.format(i) for i in range(1, components + 1)])
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


# Executando t-SNe
def tsne_apply(data_t_x, data_t_y, n_components_t=3):
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
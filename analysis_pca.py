from plot_functions import *
from useful_functions import *


# Executando o PCA
def pca_apply(data_t_x, data_t_y, n_components_t):
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
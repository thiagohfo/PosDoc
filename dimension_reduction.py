import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
from plot_functions import scatter_plot, scree_plot


# Executando o PCA
def pca_apply(df_t_x, df_t_y, n_components_t=0):
    if n_components_t == 0:
        n_components_t = len(df_t_x.columns)

    for components in range(n_components_t, 1, -1):
        x = df_t_x.values  # Passando os dados para o PCA trabalhar
        y = df_t_y.values  # Passando os dados para o PCA trabalhar
        # x = StandardScaler().fit_transform(x) # Padronizando valores

        # Aplicando o PCA
        pca = PCA(n_components=components)
        pca.fit(x)
        x_transformed = pca.transform(x)
        pca_data = pd.DataFrame(data=x_transformed, columns=['PCA{}'.format(i) for i in range(1, components + 1)])
        pca_data = pd.concat([pca_data, pd.DataFrame(data=y, columns=['Target'])], axis=1)

        # Printando dados de cada componente
        print('Value by component: {}'.format(['PCA{}={}'.format(i+1, np.round(value, 2)) for i, value in
                                               enumerate(pca.explained_variance_ratio_)]))
        print('Cumulative value: {}'.format(['{}={}'.format(i+1, np.round(value, 2)) for i, value in
                                             enumerate(pca.explained_variance_ratio_.cumsum())]))

        # Plots
        for i in pca_data.columns[1:-1]:
            scatter_plot(pca_data, 'PCA1', i)
        scree_plot(pca)

        # Deletando as variáveis, só por motivo de clareza
        del pca
        del x_transformed
        del pca_data


# Executando t-SNe
def tsne_apply(df_t_x, df_t_y, n_components_t=3):
    for components in range(n_components_t, 1, -1):
        x = df_t_x.values  # Passando os dados para o t-SNE trabalhar
        y = df_t_y.values  # Passando os dados para o t-SNE trabalhar
        # x = StandardScaler().fit_transform(x) # Padronizando valores

        # Aplicando o t-SNE
        tsne = TSNE(n_components=n_components_t, random_state=0)
        x_transformed = tsne.fit_transform(x)
        tsne_data = pd.DataFrame(data=x_transformed, columns=['t-SNE{}'.format(i) for i in
                                                              range(1, n_components_t + 1)])
        tsne_data = pd.concat([tsne_data, pd.DataFrame(data=y, columns=['Target'])], axis=1)

        # Plots
        for i in tsne_data.columns[1:-1]:
            scatter_plot(tsne_data, 't-SNE1', i)

        # Deletando as variáveis, só por motivo de clareza
        del tsne
        del x_transformed
        del tsne_data

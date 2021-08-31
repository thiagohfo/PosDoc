import plotly.express as PX
from useful_functions import *


# Gráfico de barra. As entradas são os índices e os valores.
def bar_plot(indexes_t, values_t):
    #plt.figure().add_axes([0, 0, 1, 1])
    plt.figure(figsize=(10, 6))
    plt.bar(indexes_t, values_t, width=0.4)

    for index, value in enumerate(values_t):
        plt.text(index - 0.20, value + 0.02, str(value))
    plt.show()


# Heatmapa de correlação. As entradas são a base e o método padrão é o de Pearson
def correlation_heatmap(data_t, pearson_t=True):
    if pearson_t:
        correlation_method = 'pearson'
    else:
        correlation_method = 'spearman'

    sns.heatmap(data_t.corr(method=correlation_method), xticklabels=data_t.columns, yticklabels=data_t.columns, annot=True, fmt='.1f', linewidths=.6, cmap='YlGnBu')
    plt.show()


# Scree Plot (gráfico de "joelho") - Importância de cada componente
def scree_plot(pca_t):
    plt.plot(pca_t.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Influência dos Componentes')
    plt.xlabel('Componente Principal')
    plt.ylabel('Autovalor')
    plt.show()


# Scatter plot (gráfico de dispersão)
def scatter_plot(data_t, X, Y):
    plt.figure(figsize=(10, 6))
    #plt.scatter(data_t[X], data_t[Y], c=data_t[Y], cmap='viridis')
    plt.scatter(data_t[X], data_t[Y], c=data_t[Y], cmap=cm.get_cmap('Spectral', 2))
    plt.xlabel(X)
    plt.ylabel(Y)
    #plt.colorbar()
    plt.show()


# Plot de multiplos gráficos via Scatter
def scatter_plot_PX(data_t, features_t):
    fig = PX.scatter_matrix(data_t, features_t)
    fig.update_traces(diagonal_visible=False)
    fig.show()
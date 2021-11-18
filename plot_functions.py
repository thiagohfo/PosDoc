from useful_functions import *


# Gráfico de barra. As entradas são os índices e os valores.
def bar_plot(indexes_t, values_t, name_fig_t, ylim=0, xlim=0):
    #plt.figure().add_axes([0, 0, 1, 1])
    plt.figure(figsize=(10, 6))
    plt.bar(indexes_t, values_t, width=0.4)

    for index, value in enumerate(values_t):
        plt.text(index - 0.20, value + 0.02, str(value))

    if ylim != 0:
        plt.ylim(0, ylim)

    if xlim != 0:
        plt.xlim(0, xlim)

    plt.savefig(name_fig_t)
    #plt.show()
    plt.close()

# Heatmap de correlação. As entradas são a base e o método padrão é o de Pearson
def correlation_heatmap(data_t, pearson_t=True):
    if pearson_t:
        correlation_method = 'pearson'
    else:
        correlation_method = 'spearman'

    sns.heatmap(data_t.corr(method=correlation_method), xticklabels=data_t.columns, yticklabels=data_t.columns, annot=True, fmt='.1f', linewidths=.6, cmap='YlGnBu')
    plt.show()

# Heatmap de correlação para Chi-Square Test com coeficiente de Pearson
def correlation_heatmap_chi_square(data_t):
    data_cross = pd.DataFrame(index=data_t.columns, columns=data_t.columns)

    for i in data_t.columns:
        for j in data_t.columns:
            if i == j:
                data_cross.at[i, j] = 1
            else:
                CrosstabResult = pd.crosstab(index=data_t[i], columns=data_t[j])
                data_cross.at[i, j], _, _, _ = chi2_contingency(CrosstabResult)

    data_cross = data_cross.astype(float)
    sns.heatmap(data_cross, xticklabels=data_cross.columns, yticklabels=data_cross.columns, annot=True, fmt='.1f', linewidths=.6, cmap='YlGnBu')
    plt.show()

# Scree Plot (gráfico de "joelho") - Importância de cada componente
def scree_plot(pca_t):
    print(pca_t)
    plt.plot(['{}'.format(i + 1) for i in range(pca_t.n_components)], pca_t.explained_variance_ratio_, 'ro-', linewidth=2)
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

# Radar/Polar/Spider plot
def radar_chart(values_t, groups_t, name_fig_t):
    largest_number = int(round_up(max(values_t), -(len(str(max(values_t))) - 1)))

    angles = [n / float(len(groups_t)) * 2 * pi for n in range(len(groups_t))]
    angles += angles[:1]

    multiplier_number = 10**(len(str(largest_number)) - 1)

    if multiplier_number == largest_number:
        multiplier_number = multiplier_number // 10

    number = 0
    labels = []

    while ((number + multiplier_number) < largest_number):
        number += multiplier_number
        labels.append(number)

    plt.figure(figsize=(10, 6))
    plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], groups_t, color='grey', size=8)
    plt.yticks(labels, list(map(str, labels)), color="grey", size=7)
    plt.ylim(0, largest_number)

    values_t += values_t[:1]
    plt.plot(angles, values_t, linewidth=1, linestyle='solid')
    plt.fill(angles, values_t, 'b', alpha=0.1)
    plt.savefig(name_fig_t)
    #plt.show()
    plt.close()

# Curva ROC
def roc_curve_plot(y_t, y_probs_t, name_fig_t):
    fpr, tpr, thresholds = roc_curve(y_t, y_probs_t)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', label='Curva ROC (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig('{}curva_ROC'.format(name_fig_t))
    #plt.show()
    plt.close()

# Box Plot
def box_plot(data_t, name_fig_t):
    ax = plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Base', y='Values', data=data_t)
    ax = sns.stripplot(x='Base', y='Values', data=data_t, color="orange", jitter=0.2, size=3.5)
    plt.savefig(name_fig_t)
    plt.close()
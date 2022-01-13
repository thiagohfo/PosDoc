#!/opt/anaconda3/envs/PosDoc/bin/python
from plot_functions import *
from useful_functions import *


# Informações dos dados
def mean_information(data_t):
    print("Média de sintomas: {}".format(data_t.mean(axis=1).mean(axis=0)))
    print("Desvio padrão: {}".format(data_t.mean(axis=1).std()))
    print("Tamanho da base: {}".format(len(data_t)))

# Mostra as porcentagens de cada características
def percents_features(data_t):
    for i in data_t.columns.to_list():
        print(i)
        value = data_t[i].value_counts(normalize=True).to_string()
        value = [x.split() for x in value.split('\n')]

        for j in range(len(value)):
            print("{} - {:.2f}%".format(value[j][0], float(value[j][1]) * 100))

# Gerar gráficos com base em cada uma das características (sintomas e condições)
def main_feature(data_t):
    for i in data_t.columns.to_list():
        #print(data_t[i].value_counts())
        temp_data = data_t.copy()
        temp_data.drop(temp_data[temp_data[i] != 1].index, inplace=True)

        values = []
        for j in data_t.columns.to_list():
            values.append(temp_data[j].sum(axis=0))

        bar_plot(data_t.columns.to_list(), values, i)

# Features by data type
def features_count(data_t, name_fig_t):
    values = []

    for i in data_t.columns.to_list():
        values.append(len(data_t.loc[data_t[i] == 1]))

    bar_plot(data_t.columns.to_list(), values, name_fig_t, len(data_t))

# Features by group('Internado', 'Curado', 'Óbito')
def features_by_group(data_t, features_t, group_t, name_fig_t):
    values = []
    values_mean = []

    for group in group_t:
        data_temp = data_t.copy()
        data_temp.drop(data_temp[data_temp['evolucaoCaso'] != group].index, inplace=True)
        list_temp = data_temp[features_t].sum(axis=1)
        values_mean.append(round(np.sum(list_temp) / np.count_nonzero(list_temp), 2))
        values.append(np.count_nonzero(list_temp))

    groups = '{}_Grupos'.format((name_fig_t))
    bar_plot(group_t, values, groups)
    groups_mean = '{}_Grupos_Media'.format((name_fig_t))
    bar_plot(group_t, values_mean, groups_mean)

# Gerar sumarização dos dados
def summarizing(data_t, qtd_groups_t, name_fig_t, print_t=True, radar_t=True):
    features = data_t.columns.to_list()
    data_grouped = data_t.groupby(features)
    all_lists = []

    for key, item in data_grouped:
        list_names = []

        for i, feature in enumerate(features):
            if key[i] == 0:
                continue
            else:
                list_names.append(feature.upper())

        if len(list_names) > 0:
            all_lists.append(list_names)
            list_names.append(len(data_grouped.get_group(key)))

    all_lists.sort(key=lambda x: x[-1], reverse=True)
    all_lists = all_lists[:qtd_groups_t]

    if print_t:
        for list in all_lists:
            print(list)

    if radar_t:
        values = [x[-1] for x in all_lists]
        groups = []

        for i, list in enumerate(all_lists):
            all_lists[i].pop()
            groups.append("-".join(x[:2] for x in list))

        if max(values) > 1:
            radar_chart(values, groups, name_fig_t)

# Correlação
def correlation(data_t, type_t):
    if len(data_t.loc[data_t['resultadoTeste'] == 1]) > 0:
        data_t.at[data_t['resultadoTeste'].ne(1).idxmin(), 'resultadoTeste'] = 0
    else:
        data_t.at[data_t['resultadoTeste'].ne(0).idxmin(), 'resultadoTeste'] = 1

    if type_t == 'Pearson':
        correlation_heatmap(data_t)
    elif type_t == 'Spearman':
        correlation_heatmap(data_t, pearson_t=False)
    elif type_t == 'ChiSquare':
        correlation_heatmap_chi_square(data_t)

# Informações da base por grupos (Curados, Internados e Óbitos) e sendo completa (com positivos e negativos) e depois só com positivos
def base_information(data_t, features_t, folder_t):
    groups_qty = 10

    for result in [0, 1]:
        data_temp = data_t.copy()
        name_temp = '{}{}/'.format(folder_t, features_t[0])
        directory_create(name_temp)

        if result == 1:  # Base somente com positivos
            name_temp += 'Positivos'
            data_temp.drop(data_temp[data_temp['resultadoTeste'] != result].index, inplace=True)
        else:
            data_negatives = data_t.copy()
            data_negatives.drop(data_negatives[data_negatives['resultadoTeste'] != result].index, inplace=True)
            features_count(data_negatives[features_t[1]], '{}Negativos'.format(name_temp))
            del data_negatives

            name_temp += 'Completa'

        features_count(data_temp[features_t[1]], name_temp) # Todos os sintomas de acordo com o tipo de balanceamento

        if features_t[0] == 'Conditions':
            features_by_group(data_temp, features_t[1], ['Cura', 'Internado', 'Óbito'], name_temp)

        for group in ['Cura', 'Internado', 'Óbito']:
            name_temp_2 = '{}_{}'.format(name_temp, group)
            data_temp_2 = data_temp.copy()
            data_temp_2.drop(data_temp_2[data_temp_2['evolucaoCaso'] == group].index, inplace=True)

            if features_t[0] == 'Conditions':
                features_count(data_temp_2[features_t[1]], name_temp_2)

            if len(data_temp_2) > 10: # O grupo deve ter pelo menos 10 amostras/registros
                summarizing(data_temp_2[features_t[1]], groups_qty, name_temp_2, False, True)  # Se usar os sintomas de falta de olfato e paladar
            else:
                continue
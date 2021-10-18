#!/opt/anaconda3/envs/PosDoc/bin/python
from plot_functions import *
from useful_functions import *


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
        print(data_t[i].value_counts())
        temp_data = data_t.copy()
        delete_rows_by_value(temp_data, i, 1)

        values = []
        for j in data_t.columns.to_list():
            values.append(temp_data[j].sum(axis=0))

        bar_plot(data_t.columns.to_list(), values)

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
def base_information(data_t, folder_t):
    for result in [0, 1]:
        data_temp = data_t.copy()
        name_temp = folder_t

        if result == 1:  # Base somente com positivos
            name_temp += 'Positivos'
            delete_rows_by_value(data_temp, 'resultadoTeste', result)
        else:
            name_temp += 'Completa'

        for group in ['Cura', 'Internado', 'Óbito']:
            name_temp_2 = '{}_{}'.format(name_temp, group)
            data_temp_2 = data_temp.copy()
            delete_rows_by_value(data_temp_2, 'evolucaoCaso', group)

            if len(data_temp_2) > 10:
                summarizing(data_temp_2.iloc[:, 2:8], 10, name_temp_2, False, True)
            else:
                continue
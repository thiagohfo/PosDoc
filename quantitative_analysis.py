#!/opt/anaconda3/envs/PosDoc/bin/python
import numpy as np
from useful_functions import directory_create
from plot_functions import bar_plot, radar_chart, correlation_heatmap, correlation_heatmap_chi_square


# Informações dos dados
def mean_information(df_t):
    print("Média de sintomas: {}".format(df_t.mean(axis=1).mean(axis=0)))
    print("Desvio padrão: {}".format(df_t.mean(axis=1).std()))
    print("Tamanho da base: {}".format(len(df_t)))


# Mostra as porcentagens de cada características
def percents_features(df_t):
    for i in df_t.columns.to_list():
        print(i)
        value = df_t[i].value_counts(normalize=True).to_string()
        value = [x.split() for x in value.split('\n')]

        for j in range(len(value)):
            print("{} - {:.2f}%".format(value[j][0], float(value[j][1]) * 100))


# Gerar gráficos com base em cada uma das características (sintomas e condições)
def main_feature(df_t):
    for i in df_t.columns.to_list():
        # print(data_t[i].value_counts())
        temp_data = df_t.copy()
        temp_data.drop(temp_data[temp_data[i] != 1].index, inplace=True)

        values = []
        for j in df_t.columns.to_list():
            values.append(temp_data[j].sum(axis=0))

        bar_plot(df_t.columns.to_list(), values, i)


# Features by data type
def features_count(df_t, name_fig_t):
    values = []

    for i in df_t.columns.to_list():
        values.append((df_t[i] == 1).sum())

    bar_plot(df_t.columns.to_list(), values, name_fig_t, len(df_t))


# Features by group('Internado', 'Curado', 'Óbito')
def features_by_group(df_t, features_t, group_t, name_fig_t):
    values = []
    values_mean = []

    for group in group_t:
        data_temp = df_t.copy()
        data_temp.drop(data_temp[data_temp['evolucaoCaso'] != group].index, inplace=True)
        list_temp = data_temp[features_t].sum(axis=1)
        values_mean.append(round(np.sum(list_temp) / np.count_nonzero(list_temp), 2))
        values.append(np.count_nonzero(list_temp))

    groups = '{}_Grupos'.format(name_fig_t)
    bar_plot(group_t, values, groups)
    groups_mean = '{}_Grupos_Media'.format(name_fig_t)
    bar_plot(group_t, values_mean, groups_mean)


# Correlação
def correlation(df_t, type_t):
    if len(df_t.loc[df_t['resultadoTeste'] == 1]) > 0:
        df_t.at[df_t['resultadoTeste'].ne(1).idxmin(), 'resultadoTeste'] = 0
    else:
        df_t.at[df_t['resultadoTeste'].ne(0).idxmin(), 'resultadoTeste'] = 1

    if type_t == 'Pearson':
        correlation_heatmap(df_t)
    elif type_t == 'Spearman':
        correlation_heatmap(df_t, pearson_t=False)
    elif type_t == 'ChiSquare':
        correlation_heatmap_chi_square(df_t)


# Gerar sumarização dos dados
def summarizing(df_t, qtd_groups_t, name_fig_t, radar_t=True):
    features = df_t.columns.to_list()
    data_grouped = df_t.groupby(features)
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

    if radar_t:
        values = [x[-1] for x in all_lists]
        groups = []

        for i, current_list in enumerate(all_lists):
            all_lists[i].pop()
            groups.append("-".join(x[:3] for x in current_list))

        if max(values) > 1:
            radar_chart(values, groups, name_fig_t)


# Informações da base por grupos (Curados, Int. e Óbitos) na completa (com positivos e negativos) e depois com positivos
def base_information(df_t, features_t, folder_t):
    groups_qty = 10

    for result in [0, 1]:
        data_temp = df_t.copy()
        name_temp = '{}{}/'.format(folder_t, features_t[0])
        directory_create([name_temp])

        if result == 1:  # Base somente com positivos
            name_temp += 'Positivos'
            data_temp.drop(data_temp[data_temp['resultadoTeste'] != result].index, inplace=True)
        else:
            data_negatives = df_t.copy()
            data_negatives.drop(data_negatives[data_negatives['resultadoTeste'] != result].index, inplace=True)
            features_count(data_negatives[features_t[1]], '{}Negativos'.format(name_temp))
            del data_negatives

            name_temp += 'Completa'

        features_count(data_temp[features_t[1]], name_temp)  # Todos os sintomas de acordo com o tipo de balanceamento

        if features_t[0] == 'Conditions':
            features_by_group(data_temp, features_t[1], ['Cura', 'Internado', 'Óbito'], name_temp)

        for group in ['Cura', 'Internado', 'Óbito']:
            name_temp_2 = '{}_{}'.format(name_temp, group)
            data_temp_2 = data_temp.copy()
            data_temp_2.drop(data_temp_2[data_temp_2['evolucaoCaso'] == group].index, inplace=True)

            if features_t[0] == 'Conditions':
                features_count(data_temp_2[features_t[1]], name_temp_2)

            if len(data_temp_2) > 50:  # O grupo deve ter pelo menos 10 amostras/registros
                summarizing(data_temp_2[features_t[1]], groups_qty, name_temp_2, True)
            else:
                continue

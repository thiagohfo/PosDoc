#!/opt/anaconda3/envs/PosDoc/bin/python
from plot_functions import *
from useful_functions import *
from analysis_pca import pca_apply
from analysis_regressions import lr_prediction, rr_prediction, lasso_prediction, logistic_prediction


# Leitura dos dados
file = 'Bases/dados-ce-1.csv'
data = read_data(file)


# Mostra as porcentagens de cada características
def percents_features(features_t):
    for i in features_t:
        print(i)
        value = data[i].value_counts(normalize=True).to_string()
        value = [x.split() for x in value.split('\n')]

        for j in range(len(value)):
            print("{} - {:.2f}%".format(value[j][0], float(value[j][1]) * 100))


# Gerar gráficos com base em cada uma das características (sintomas e condições)
def main_feature(features_t):
    for i in features_t:
        print(data[i].value_counts())
        temp_data = data.copy()
        delete_rows_by_value(temp_data, i, 1)

        values = []
        for j in features_t:
            values.append(temp_data[j].sum(axis=0))

        bar_plot(features_t, values)


# Gerar sumarização dos dados
def summarizing(data_t, qtd_groups_t, print_t=True, radar_t=True):
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

        radar_chart(values, groups)


# Columns
conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']
all = symptoms + conditions + ['resultadoTeste']


# Pega somente rows com testes positivos ou negativos
#delete_rows_by_value(data, 'resultadoTeste', 1)


# Montando o perfil de acordo com alguns tipos de evolução
#delete_rows_by_value(data, 'evolucaoCaso', 'Cura') # Deixa somente os curados
#delete_rows_by_value(data, 'evolucaoCaso', 'Internado') # Deixa somente os internados
#delete_rows_by_value(data, 'evolucaoCaso', 'Óbito') # Deixa somente os óbitos


# Funções para apresentação dos dados
#percents_features(symptoms + conditions)
#main_feature(symptoms)
#summarizing(data[symptoms], 10)


# Análises
#pca_apply(data[symptoms], data['resultadoTeste'], 6)
#tsne_apply(data[symptoms], data['resultadoTeste'], 3)
#lr_prediction(data[symptoms], data['resultadoTeste'])
#lasso_prediction(data[symptoms], data['resultadoTeste'])
#rr_prediction(data[symptoms], data['resultadoTeste'])
logistic_prediction(data[symptoms], data['resultadoTeste'])


# Linha responsável por modificar apenas 1 entrada de resultadoTeste, pois se ficarem todos com o mesmo valor, não rolará correlação
#data.at[data['resultadoTeste'].ne(1).idxmin(), 'resultadoTeste'] = 0
#data.at[data['resultadoTeste'].ne(0).idxmin(), 'resultadoTeste'] = 1
# Correlação com base em Pearson, Spearman e Pearson Chi-Square
#correlation_heatmap(data[symptoms + ['resultadoTeste']]) # Pearson
#correlation_heatmap(data[symptoms + ['resultadoTeste']], pearson_t=False) # Spearman
#correlation_heatmap_chi_square(data[symptoms], symptoms) # Chi-Square Test


exit(0)
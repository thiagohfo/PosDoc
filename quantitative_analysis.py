#!/opt/anaconda3/envs/PosDoc/bin/python
from plot_functions import *
from useful_functions import *
import scipy


# Leitura dos dados
file = 'Bases/dados-ce-1.csv'
data = read_data()
print("Tamanha da base: {}".format(len(data)))


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
def summarizing(data_t, features_t):
    data_grouped = data_t.groupby(features_t)

    for key, item in data_grouped:
        data_grouped.get_group(key)

        list_names = []

        for i, feature in enumerate(features_t):
            if key[i] == 0:
                list_names.append(feature)
            else:
                list_names.append(feature.upper())

        # print(list_names)
        for name in list_names:
            if name.isupper():
                print("\033[1m" + name + "\033[0m", end=' ')

        print()
        print(len(data_grouped.get_group(key)))


# Columns
conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
#conditions = ['cardiacas', 'diabetes', 'renais', 'imunossupressao']
symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']
all = symptoms + conditions + ['resultadoTeste']


# Pega somente rows com testes positivos ou negativos
delete_rows_by_value(data, 'resultadoTeste', 1)


# Montando o perfil de acordo com alguns tipos de evolução
#delete_rows_by_value(data, 'evolucaoCaso', 'Cura', False) # Deleta todos os curados
#delete_rows_by_value(data, 'evolucaoCaso', 'Em tratamento domiciliar', False) # Deleta todos em tratamento
#delete_rows_by_value(data, 'evolucaoCaso', 'Cura') # Deixa somente os curados


# Linha responsável por modificar apenas 1 entrada de resultadoTeste, pois se ficarem todos com o mesmo valor, não rolará correlação
#data.at[1, 'resultadoTeste'] = 1 # Para casos de curados
data.at[16, 'resultadoTeste'] = 0 # Para casos de óbitos
#print(data['evolucaoCaso'].value_counts())
#print("Tamanha da base: {}".format(len(data)))


#percents_features(symptoms + conditions)
#main_feature(symptoms)
#summarizing(data, symptoms)


# Correlação com base em Pearson e Spearman
correlation_heatmap(data[symptoms + ['resultadoTeste']])
correlation_heatmap(data[symptoms + ['resultadoTeste']], pearson_t=False)


# Média de sintomas por pessoa (Considerando só casos positivos)
print("Média de sintomas: {}".format(data[symptoms].mean(axis=1).mean(axis=0)))
print("Desvio padrão: {}".format(data[symptoms].mean(axis=1).std()))
print("Tamanha da base: {}".format(len(data)))

exit(0)
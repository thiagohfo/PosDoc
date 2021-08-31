#!/opt/anaconda3/envs/PosDoc/bin/python
from plot_functions import *
from useful_functions import *
import scipy


# Leitura dos dados
file = 'Bases/dados-ce-1.csv'
data = read_data()
print(len(data))


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


# Columns
conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
#conditions = ['cardiacas', 'diabetes', 'renais', 'imunossupressao']
symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']
all = symptoms + conditions + ['resultadoTeste']


# Pega somente rows com testes positivos ou negativos
delete_rows_by_value(data, 'resultadoTeste', 0, False)


percents_features(symptoms + conditions)

#main_feature(symptoms)


# Linha responsável por modificar apenas 1 entrada de resultadoTeste, pois se ficarem todos com o mesmo valor, não rolará correlação
data.at[2, 'resultadoTeste'] = 0

#correlation_heatmap(data[all])
#correlation_heatmap(data[all], pearson_t=False)

# Média de sintomas por pessoa (Considerando só casos positivos)
print(data[symptoms].mean(axis=1).mean(axis=0))
print(data[symptoms].mean(axis=1).std())


exit(0)
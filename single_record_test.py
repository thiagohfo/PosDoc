#!/opt/anaconda3/envs/PosDoc/bin/python
'''Arquivo para realizar testes para um único registro e poder visualizar porcentagens de chances negativa e positiva'''
import pandas as pd

from useful_functions import *

data = read_data('Bases/dados-sp-1.csv')
model = loading_model('logistic_model_Symptoms_Amount')

symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'olfativos', 'gustativos']
inputs = []

for i in symptoms:
    inputs.append(int(input('{}: '.format(i))))

test_case = pd.DataFrame(data=np.array([inputs]), columns=symptoms)

chances = model.predict_proba(test_case)

print('Chance de ser negativo: {:.2f}%\nChance de ser positivo: {:.2f}%'.format(chances[0][0] * 100, chances[0][1] * 100))


data_positives = data[data['resultadoTeste'] == 1]
data_negatives = data[data['resultadoTeste'] == 0]
data_positives = data_positives[data_positives[symptoms].sum(axis=1) > 3]
data_positives = data_positives[(data_positives['olfativos'] == 1) | (data_positives['gustativos'] == 1)]
data_modified = dataset_balancing(data_positives, data_negatives)

total_size_modified = len(data_modified)
data_modified = data_modified[(data_modified['tosse'] == inputs[0]) & (data_modified['febre'] == inputs[1]) &
                              (data_modified['garganta'] == inputs[2]) & (data_modified['coriza'] == inputs[5]) &
                              (data_modified['cabeca'] == inputs[4]) & (data_modified['dispneia'] == inputs[3]) &
                              (data_modified['olfativos'] == inputs[6]) & (data_modified['gustativos'] == inputs[7])]
parcial_size_modified = len(data_modified)

print('Corresponde a {:.2f}% da base balanceada'.format((parcial_size_modified/total_size_modified) * 100))

total_size = len(data)
data = data[(data['tosse'] == inputs[0]) & (data['febre'] == inputs[1]) & (data['garganta'] == inputs[2]) &
            (data['coriza'] == inputs[5]) & (data['cabeca'] == inputs[4]) & (data['dispneia'] == inputs[3]) &
            (data['olfativos'] == inputs[6]) & (data['gustativos'] == inputs[7])]
parcial_size = len(data)

print('Corresponde a {:.2f}% da base original'.format((parcial_size/total_size) * 100))

exit(0)
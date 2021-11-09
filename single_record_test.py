#!/opt/anaconda3/envs/PosDoc/bin/python
'''Arquivo para realizar testes para um único registro e poder visualizar porcentagens de chances negativa e positiva'''
import pandas as pd

from useful_functions import *

model = loading_model('logistic_model_Symptoms_Based_Balanced')

symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'olfativos', 'gustativos']
inputs = []

for i in symptoms:
    inputs.append(int(input('{}: '.format(i))))

test_case = pd.DataFrame(data=np.array([inputs]), columns=symptoms)

chances = model.predict_proba(test_case)

print('Chance de ser negativo: {:.2f}%\nChance de ser positivo: {:.2f}%'.format(chances[0][0] * 100, chances[0][1] * 100))

exit(0)
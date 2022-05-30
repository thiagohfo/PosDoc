#!/opt/anaconda3/envs/PosDoc/bin/python
# Arquivo para realizar testes para um único registro e poder visualizar porcentagens de chances negativa e positiva
import numpy as np
import pandas as pd
from useful_functions import loading_model, convert_to_categorical


def test_symptoms(datas_t):
    inputs = []
    model = loading_model('logistic_model_Symptoms_Amount')

    for i in datas_t['symptoms']:
        inputs.append(int(input('{}: '.format(i))))

    test_case = pd.DataFrame(data=np.array([inputs]), columns=datas_t['symptoms'])
    chances = model.predict_proba(test_case)

    print('Chance de ser negativo: {:.2f}%\nChance de ser positivo: {:.2f}%'.format(chances[0][0] * 100,
                                                                                    chances[0][1] * 100))


def test_conditions(datas_t):
    inputs = []
    for i in datas_t['conditions_model']:
        inputs.append(int(input('{}: '.format(i))))

    test_case = pd.DataFrame(data=np.array([inputs]), columns=datas_t['conditions_model'])
    test_case = test_case.apply(convert_to_categorical, axis=1)
    internar = False
    msg = ''

    for case in [['Cura', 'Óbito'], ['Cura', 'Internado']]:
        model = loading_model('logistic_model_{}_{}'.format(case[0], case[1]))
        chances = model.predict_proba(test_case)

        if case[1] == 'Óbito':
            if (chances[0][0] * 100) < 32:
                msg = 'PCovid 3 com chance de óbito: {:.2f}%'.format(chances[0][1] * 100)
                internar = False
            else:
                msg = 'PCovid 1 com chance de cura: {:.2f}%'.format(chances[0][0] * 100)
                internar = True
        elif (case[1] == 'Internado') & internar:
            if (chances[0][1] * 100) > 55:
                msg = 'PCovid 2 com chance de precisar de internação: {:.2f}%'.format(chances[0][1] * 100)
                internar = False

    print(msg)


if __name__ == '__main__':
    # file = 'Bases/AC.csv'
    # df = read_data(file)
    datas = {'other_symptoms': {'mialgia': ['mialgia', 'corpo', 'muscular'],
                                'cabeca': ['cefaleia', 'cabeca'],
                                'fadiga': ['fadiga', 'cansaco', 'fraqueza', 'indisposicao', 'adinamia', 'moleza',
                                           'astenia'],
                                'nauseas': ['nausea', 'nauseas', 'tontura', 'mal estar', 'enjoo'],
                                'coriza': ['coriza'],
                                'anosmia': ['olfato', 'anosmia', 'hiposmia'],
                                'hipogeusia': ['paladar', 'hipogeusia', 'disgeusia'],
                                'diarreia': ['diarreia'], 'febre': ['febre', 'febril'],
                                'tosse': ['tosse']
                                },
             'symptoms': ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'hipogeusia', 'anosmia',
                          'fadiga', 'nauseas', 'mialgia', 'diarreia'],
             'conditions': ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade',
                            'imunossupressao'],
             'conditions_model': ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade',
                                  'imunossupressao', 'diasSintomas', 'idade']
             }

    # test_symptoms(datas)
    test_conditions(datas)

exit(0)

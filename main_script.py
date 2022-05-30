import warnings
import pandas as pd
from plot_functions import box_plot
from metrics_script import metrics_calc
from summary_report import model_summary
from preprocessing_data import preprocessing
from data_info_save import basic_informations
from quantitative_analysis import base_information
from analysis_regressions import logistic_prediction
from training_conditions import conditions_training
from algorithms import naive_bayes, decision_tree
from useful_functions import clean_features
from useful_functions import read_data, dataset_balancing, loading_model, directory_create, get_files_names


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
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
                                  'imunossupressao', 'diasSintomas', 'idade'],
             'coefs_report_symptoms': pd.DataFrame(columns=['Base', 'Values']),
             'coefs_report_conditions': pd.DataFrame(columns=['Base', 'Values'])
             }
    read = True  # Se falso, as bases serão pré-processadas. Se verdadeiro, serão apenas lidas.
    train = False  # Se verdadeiro, modelos serão construídos. Se falso, usará os modelos na pasta Modelos.
    metrics_conditions = True
    train_conditions = False

    files = get_files_names('Bases/', False)  # Se True, bases são agrupadas.

    # Loop principal com todas as bases
    for name in files:
        if read:  # Leitura ou pré-processamento
            df = read_data('Bases/{}'.format(name))
        else:
            df = preprocessing('Bases/{}'.format(name))

        clean_features(df, datas, symptoms_t=False, conditions_t=False)
        coefs_report = pd.DataFrame(columns=['Base', 'Values'])
        backup_data = df.copy()  # Backup do dataset para garantir que o processo de balanceamento pega sem alterações

        for kind in ['Unbalanced', 'Balanced', 'Symptoms_Amount']:
            df = backup_data.copy()  # Utiliza o backup para realizar operações em dataset normal
            folder = 'Bases/{}/{}/'.format(name[:-4], kind)  # Extração do nome
            file_fullname = '{}{}'.format(folder, name[:-4])

            df = dataset_balancing(df, datas['symptoms'], kind)

            directory_create(['Bases/{}/'.format(name[:-4]), folder])  # Cria diretório base e subdiretórios

            for feature in dict({'Symptoms': datas['symptoms'], 'Conditions': datas['conditions']}).items():
                base_information(df, feature, folder)  # Informações dos grupos de sintomas/condições

            if (kind == 'Balanced') & (train_conditions | metrics_conditions):
                conditions_training(df, datas, folder, train_conditions)

            if train:
                logistic_prediction(df[datas['symptoms']], df['resultadoTeste'], kind)
                # naive_bayes(df[datas['symptoms']], df['resultadoTeste'], kind)
                # decision_tree(df[datas['symptoms']], df['resultadoTeste'], kind)

            model = loading_model('logistic_model_{}'.format(kind))  # Carregando o modelo
            # model = loading_model('naive_bayes_model_{}'.format(kind))  # Carregando o modelo
            # model = loading_model('decision_tree_model_{}'.format(kind))  # Carregando o modelo
            metrics_calc(df[datas['symptoms']], df['resultadoTeste'], model, folder, kind)
            basic_informations(df, folder)

            datas['coefs_report_symptoms'] = model_summary(df, datas, folder, kind, 'symptoms',
                                                           datas['coefs_report_symptoms'])

            box_plot(datas['coefs_report_symptoms'], 'Bases/{}/boxplot_symptoms'.format(name[:-4]), 'Symptoms')

    exit(0)

import pandas as pd
from data_info_save import *
from metrics_script import *
from summary_report import *
from useful_functions import *
from dimension_reduction import *
from analysis_regressions import *
from quantitative_analysis import *
from preprocessing_data import preprocessing

if __name__ == '__main__':
    # Columns
    conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
    symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'olfativos', 'gustativos']
    others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']
    all = symptoms + conditions + ['resultadoTeste']

    # Pegando todas as bases
    files = os.listdir('Bases/')
    files = [name for name in files if name[-3:] == 'csv']
    read = True # Se falso, as bases serão pré-processadas. Se verdadeiro, serão apenas lidas.
    train = False # Se verdadeiro, modelos serão construídos. Se falso, usará os modelos na pasta Modelos.

    # Loop principal com todas as bases
    for name in files:
        # Leitura ou pré-processamento
        if read:
            data = read_data('Bases/{}'.format(name))
        else:
            data = preprocessing('Bases/{}'.format(name))

        coefs_report = pd.DataFrame(columns=['Base', 'Values'])
        backup_data = data.copy() # Backup do dataset para garantir que o processo de balanceamento pega um dataset sem alterações

        for kind in ['Unbalanced', 'Balanced', 'Symptoms_Based', 'Symptoms_Amount']:
            data = backup_data.copy() # Utiliza o backup para realizar operações em dataset normal
            folder = 'Bases/{}/{}/'.format(name[:-4], kind) # Extração do nome
            file_fullname = '{}{}'.format(folder, name[:-4])

            data_positives = data[data['resultadoTeste'] == 1]
            data_negatives = data[data['resultadoTeste'] == 0]

            if kind == 'Balanced':
                data = dataset_balancing(data_positives, data_negatives)
            elif kind == 'Symptoms_Based':
                data_positives = data_positives[(data_positives['olfativos'] == 1) | (data_positives['gustativos'] == 1)]
                data = dataset_balancing(data_positives, data_negatives)
            elif kind == 'Symptoms_Amount':
                data_positives = data_positives[data_positives[symptoms].sum(axis=1) > 3]
                data_positives = data_positives[(data_positives['olfativos'] == 1) | (data_positives['gustativos'] == 1)]
                data = dataset_balancing(data_positives, data_negatives)

            if train:
                logistic_prediction(data[symptoms], data['resultadoTeste'], kind)

            model = loading_model('logistic_model_{}'.format(kind))  # Carregando o modelo

            directory_create('Bases/{}/'.format(name[:-4])) # Criação do diretório para cada base

            directory_create(folder) # Criação do diretório para cada base e seu tipo (Balanced, Unbalanced)

            base_information(data, symptoms, folder)  # Informações dos grupos de sintomas

            metrics_calc(data[symptoms], data['resultadoTeste'], model, folder, file_fullname)

            coefs_report = model_summary(sm.Logit(data['resultadoTeste'], data[symptoms]), folder, kind, coefs_report)

            basic_informations(data, file_fullname)

        #shutil.move('Bases/{}'.format(name), 'Bases/{}'.format(name[:-4])) # Movendo os arquivos correspondente para dentro das pastas

        box_plot(coefs_report, 'Bases/{}/boxplot'.format(name[:-4]))

    exit(0)
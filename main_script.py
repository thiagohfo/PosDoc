import pandas as pd

from data_info_save import *
from metrics_script import *
from useful_functions import *
from dimension_reduction import *
from analysis_regressions import *
from quantitative_analysis import *
from preprocessing_data import preprocessing


# Balanceamento da base de dados
def dataset_balancing(data_t):
    data_majority = data_t[data_t['resultadoTeste'] == 1]
    data_minority = data_t[data_t['resultadoTeste'] == 0]

    data_size = round(len(data_minority) + (len(data_minority) * 0.60))
    data_reduced = resample(data_majority, replace=False, n_samples=data_size, random_state=123)
    data_balanced = pd.concat([data_reduced, data_minority])

    #print(data_balanced['resultadoTeste'].value_counts())

    return data_balanced

if __name__ == '__main__':
    # Columns
    conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
    symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
    others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']
    all = symptoms + conditions + ['resultadoTeste']

    # Correlações
    #correlation(data, 'tipo') # Trocar o tipo por um dos predefinidos são: Pearson, Spearman e ChiSquare

    # Pegando todas as bases
    files = os.listdir('Bases/')
    files = [name for name in files if name[-3:] == 'csv']
    read = True

    # Loop principal com todas as bases
    all_bases = ''
    for name in files:
        # Leitura ou pré-processamento
        if read:
            data = read_data('Bases/{}'.format(name))
        else:
            data = preprocessing('Bases/{}'.format(name))

        folder = 'Bases/{}/'.format(name[:-4]) # Extração do nome
        file_fullname = '{}{}'.format(folder, name[:-4])

        data = dataset_balancing(data)

        if not os.path.exists(folder): # Criação do diretório para cada base
            os.mkdir(folder)

        model = loading_model('logistic_model_balanced') # Carregando o modelo

        base_information(data, folder)  # Informações dos grupos
        metrics_calc(data[symptoms], data['resultadoTeste'], model, folder, file_fullname)
        basic_informations(data, file_fullname)

        shutil.move('Bases/{}'.format(name), 'Bases/{}'.format(name[:-4])) # Movendo os arquivos correspondente para dentro das pastas

    exit(0)
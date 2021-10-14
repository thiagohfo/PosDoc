from data_info_save import *
from useful_functions import *
from dimension_reduction import *
from analysis_regressions import *
from quantitative_analysis import *
from preprocessing_data import preprocessing


# Criando o modelo
def model_create(data_t):
    logistic_prediction(data_t[symptoms], data_t['resultadoTeste'])


if __name__ == '__main__':
    # Columns
    conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
    symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
    others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']
    all = symptoms + conditions + ['resultadoTeste']


    # Pega somente rows com testes positivos ou negativos
    #delete_rows_by_value(data, 'resultadoTeste', 1)


    # Correlações
    #correlation(data, 'tipo') # Trocar o tipo por um dos predefinidos são: Pearson, Spearman e ChiSquare


    # Regressões ou PCA/tSNe
    #funcao(variaveis, variavel_preditora) # No arquivo analysis_regressions


    # Montando o perfil de acordo com alguns tipos de evolução
    #delete_rows_by_value(data, 'evolucaoCaso', 'grupo') # Deixa somente o grupo escolhido, as opções são: Cura, Internado, Óbito


    # Pegando todas as bases
    files = os.listdir('Bases/')
    files = [name for name in files if name[-3:] == 'csv']


    # Loop principal com todas as bases
    all_bases = ''
    for data_name in files:
        print(data_name)
        #data = read_data('Bases/{}'.format(data_name))
        data = preprocessing('Bases/{}'.format(data_name))
        model = loading_model('logistic_model')
        information = 'Accuracy: {:.2f}%'.format(100 * model.score(data[symptoms], data['resultadoTeste']))
        single_information(information, data_name)
        basic_informations(data, data_name)
        all_bases += 'Base {} and {}\n'.format(data_name, information)

    single_information(all_bases, 'TODAS AS BASES.   ')


    exit(0)
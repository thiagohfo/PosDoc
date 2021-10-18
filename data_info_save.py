#!/opt/anaconda3/envs/PosDoc/bin/python
from useful_functions import *


# Função para salvar informações gerais da base em um arquivo TXT
def basic_informations(data_t, file_t, cleaned_dataset=False):
    pd.set_option('display.max_columns', None) # Mostrar todas as colunas
    pd.set_option('display.max_rows', None) # Mostrar apenas 10 linhas

    # Criando o arquivo
    file_to_write = open('{}.txt'.format(file_t), 'a')

    # Pegando a quantidade de colunas
    if cleaned_dataset == False:
        cols_name = ['profissionalSaude', 'tipoTeste', 'resultadoTeste', 'sexo', 'evolucaoCaso', 'classificacaoFinal']
    else:
        cols_name = ['profissionalSaude', 'tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao', 'tipoTeste', 'resultadoTeste', 'sexo', 'idade', 'evolucaoCaso', 'classificacaoFinal']

    # Escrevendo no arquivo
    file_to_write.write("Tamanho da base: {}\n".format(len(data_t)))
    file_to_write.write('--------------------------------------------------------------------------------\n\n')
    file_to_write.write('---------------------------------------Tipos das Colunas---------------------------------------\n')
    file_to_write.write(data_t.dtypes.to_string())
    file_to_write.write('\n--------------------------------------------------------------------------------\n\n')

    for col in cols_name:
        file_to_write.write('---------------------------------------{}---------------------------------------\n'.format(col))
        file_to_write.write('{}\n'.format(data_t[col].value_counts().to_string()))
        file_to_write.write('--------------------------------------------------------------------------------\n\n')

    # Fechando o arquivo
    file_to_write.close()

# Salva informação única
def single_information(information_t, file_t):
    # Criando o arquivo
    file_to_write = open('{}.txt'.format(file_t), 'a')

    # Escrevendo no arquivo
    file_to_write.write('--------------------------------------------------------------------------------\n')
    file_to_write.write('{}'.format(information_t))
    file_to_write.write('--------------------------------------------------------------------------------\n')

    # Fechando o arquivo
    file_to_write.close()
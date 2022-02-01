#!/opt/anaconda3/envs/PosDoc/bin/python
import pandas as pd


# Função para salvar informações gerais da base em um arquivo TXT
def basic_informations(data_t, file_t, cleaned_dataset=False):
    pd.set_option('display.max_columns', None)  # Mostrar todas as colunas
    pd.set_option('display.max_rows', None)  # Mostrar apenas 10 linhas

    # Criando o arquivo
    file_to_write = open('{}Métricas_e_Resumo_Dados.txt'.format(file_t), 'a')

    # Pegando a quantidade de colunas
    if cleaned_dataset:
        cols_name = ['tipoTeste', 'resultadoTeste', 'evolucaoCaso', 'classificacaoFinal']
    else:
        cols_name = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'hipogeusia', 'anosmia', 'fadiga',
                     'nauseas', 'mialgia', 'diarreia', 'cardiacas', 'diabetes', 'respiratorias', 'renais', 'obesidade',
                     'imunologica', 'imunossupressao', 'gestante', 'puerpera', 'tipoTeste', 'resultadoTeste',
                     'evolucaoCaso', 'classificacaoFinal']

    # Escrevendo no arquivo
    file_to_write.write("Tamanho da base: {}\n".format(len(data_t)))
    file_to_write.write('--------------------------------------------------------------------------------\n\n')

    for col in cols_name:
        file_to_write.write('------------------------------------{}------------------------------------\n'.format(col))
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

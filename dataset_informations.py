#!/opt/anaconda3/envs/PosDoc/bin/python
from useful_functions import *


# Leitura dos dados
file = 'Bases/dados-ce-1.csv'
data = read_data(file)
pd.set_option('display.max_columns', None) # Mostrar todas as colunas
pd.set_option('display.max_rows', None) # Mostrar apenas 10 linhas
print("Tamanha da base: {}".format(len(data)))


# Criando o arquivo
file_to_write = open('{}txt'.format(file[:-3]), 'w')


# Pegando a quantidade de colunas
cleaned_dataset = False
if cleaned_dataset == False:
    cols_name = ['profissionalSaude', 'tipoTeste', 'resultadoTeste', 'sexo', 'idade', 'evolucaoCaso', 'classificacaoFinal']
else:
    cols_name = ['profissionalSaude', 'tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza', 'cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao', 'tipoTeste', 'resultadoTeste', 'sexo', 'idade', 'evolucaoCaso', 'classificacaoFinal']


# Escrevendo no arquivo
file_to_write.write("Tamanha da base: {}\n\n".format(len(data)))
file_to_write.write(data.dtypes.to_string())
file_to_write.write('\n\n\n')
for col in cols_name:
    file_to_write.write('-------------------------------{}---------------------------------\n'.format(col))
    file_to_write.write(data[col].value_counts().to_string())
    file_to_write.write('\n')
    file_to_write.write('--------------------------------------------------------------------')
    file_to_write.write('\n\n')


# Fechando o arquivo
file_to_write.close()


exit(0)
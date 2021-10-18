#!/opt/anaconda3/envs/PosDoc/bin/python
from useful_functions import *

data = read_data('Bases/dados-rj-2.csv')
pd.set_option('display.max_columns', None) # Mostrar todas as colunas
pd.set_option('display.max_rows', None) # Mostrar apenas 10 linhas
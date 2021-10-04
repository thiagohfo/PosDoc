#!/opt/anaconda3/envs/PosDoc/bin/python
from plot_functions import *
from useful_functions import *


# Leitura dos dados
file = 'Bases/dados-ce-1.csv'
data = read_data(file)
print(len(data))


features_columns = ['tosse', 'febre', 'garganta', 'coriza', 'cabeca', 'dispneia']


data_grouped = data.groupby(features_columns)


for key, item in data_grouped:
    data_grouped.get_group(key)

    list_names = []

    for i, feature in enumerate(features_columns):
        if key[i] == 0:
            list_names.append(feature)
        else:
            list_names.append(feature.upper())

    #print(list_names)
    for name in list_names:
        if name.isupper():
            print("\033[1m" + name + "\033[0m", end=' ')

    print()
    print(len(data_grouped.get_group(key)))

print(len(data))


exit(0)
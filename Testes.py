#!/opt/anaconda3/envs/PosDoc/bin/python
from plot_functions import *
from useful_functions import *


# Leitura dos dados
file = 'Bases/dados-ce-1.csv'
data = read_data(file)
print(len(data))

import matplotlib.pyplot as plt
import pandas as pd
from math import pi



#print(int(round_up(len(data), -(len(str(len(data))) - 1))))



exit(0)
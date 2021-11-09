from data_info_save import *
from useful_functions import *

def model_summary(model_t, folder_t):
    report = model_t.fit()
    single_information(report.summary(), '{}Resumo'.format(folder_t))
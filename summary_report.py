from data_info_save import *
from useful_functions import *

def model_summary(model_t, folder_t, kind_t, coefs_report_t):
    report = model_t.fit()
    single_information(report.summary(), '{}Resumo'.format(folder_t))
    temp = pd.DataFrame({ 'Symptoms': report.params.index.values, 'Base': np.repeat(kind_t, 8), 'Values': report.params.values })
    coefs_report_t = coefs_report_t.append(temp)

    return coefs_report_t
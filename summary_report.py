import numpy as np
import pandas as pd
import statsmodels.api as sm
from plot_functions import diagn_res_fit
from data_info_save import single_information


def model_summary(df_t, datas_t, folder_t, kind_t, type_t, coefs_report_t, case_t=None):
    if type_t == 'conditions':
        features = datas_t['conditions_model']
        target = 'evolucaoCaso_{}'.format(case_t[1])
        report_name = 'Conditions'
    else:
        features = datas_t['symptoms']
        target = 'resultadoTeste'
        report_name = 'Symptoms'

    qty_features = len(features)

    try:
        report = sm.Logit(df_t[target], df_t[features]).fit(method='bfgs', disp=False)
        diagn_res_fit(report, pd.concat([df_t[features], df_t[target]], axis=1), '{}Diagnostic/'.format(folder_t),
                      report_name)
        single_information(report.summary(), '{}Resumo_Modelo_{}'.format(folder_t, report_name))
        temp = pd.DataFrame({report_name: report.params.index.values, 'Base': np.repeat(kind_t, qty_features),
                             'Values': report.params.values})
        coefs_report_t = coefs_report_t.append(temp)
    except Exception as e:
        print('Erro: {}'.format(e))

    return coefs_report_t

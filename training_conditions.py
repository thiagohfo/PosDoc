# from plot_functions import box_plot
import pandas as pd
from plot_functions import box_plot
from metrics_script import metrics_calc
from summary_report import model_summary
from analysis_regressions import logistic_prediction
from useful_functions import dataset_conditions_balancing, convert_to_categorical, loading_model


def conditions_training(df_t, datas_t, folder_t, train_t=False):
    backup_df = df_t.copy()

    for case in [['Cura', 'Óbito'], ['Cura', 'Internado']]:
        df_t = df_t.apply(convert_to_categorical, axis=1)
        df_t = df_t.drop(df_t[~((df_t['evolucaoCaso'] == case[0]) | (df_t['evolucaoCaso'] == case[1]))].index)
        df_t = pd.get_dummies(df_t, columns=['evolucaoCaso'], drop_first=True)
        df_t = dataset_conditions_balancing(df_t, case)

        if train_t:
            logistic_prediction(df_t[datas_t['conditions_model']], df_t['evolucaoCaso_{}'.format(case[1])],
                                '{}_{}'.format(case[0], case[1]))

        model = loading_model('logistic_model_{}_{}'.format(case[0], case[1]))
        metrics_calc(df_t[datas_t['conditions_model']], df_t['evolucaoCaso_{}'.format(case[1])], model,
                     folder_t, '{}-{}'.format(case[0], case[1]), 'Conditions')
        datas_t['coefs_report_conditions'] = model_summary(df_t, datas_t, folder_t, case[1], 'conditions',
                                                           datas_t['coefs_report_conditions'], case)
        box_plot(datas_t['coefs_report_conditions'], '{}boxplot_conditions'.format(folder_t[:-9]), 'Conditions')
        df_t = backup_df.copy()

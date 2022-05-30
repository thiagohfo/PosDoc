import numpy as np
from plot_functions import roc_curve_plot
from data_info_save import single_information
from sklearn.model_selection import cross_validate


# Cálculo das métricas
def metrics_calc(X_df_t, y_df_t, model_t, folder_t, model_name_t, feature_t='Symptoms'):
    y_true = y_df_t
    y_probs = model_t.decision_function(X_df_t)

    # Salva o plot da Curva ROC
    roc_curve_plot(y_true, y_probs, '{}{}_'.format(folder_t, feature_t))

    # Salvando informações em arquivos
    scores = cross_validate(model_t, X_df_t, y_df_t, cv=10, scoring=('f1', 'precision', 'recall', 'accuracy'))
    information = 'Accuracy: {:.2f}%\n'.format(100 * np.mean(scores['test_accuracy']))
    information += 'Precision: {:.2f}%\n'.format(100 * np.mean(scores['test_precision']))
    information += 'Recall: {:.2f}%\n'.format(100 * np.mean(scores['test_recall']))
    information += 'F1-Score: {:.2f}%\n'.format(100 * np.mean(scores['test_f1']))

    # Salva em um arquivo somente com as métricas
    metrics_information = 'Nome do modelo: {}\n{}'.format(model_name_t, information)
    single_information(metrics_information, '{}Métricas_e_Resumo_Dados'.format(folder_t))

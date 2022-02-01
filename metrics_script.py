import numpy as np
from plot_functions import roc_curve_plot
from data_info_save import single_information
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score


# Cálculo das métricas
def metrics_calc(x_df_t, y_df_t, model_t, folder_t, model_name_t, feature_t='Symptoms'):
    y_true = y_df_t
    y_pred = model_t.predict(x_df_t)
    y_probs = model_t.decision_function(x_df_t)

    # Salva o plot da Curva ROC
    # if save_t:
    roc_curve_plot(y_true, y_probs, '{}{}_'.format(folder_t, feature_t))

    # Salvando informações em arquivos
    information = 'Accuracy: {:.2f}%\n'.format(100 * model_t.score(x_df_t, y_df_t))
    information += 'Precision: {:.2f}%\n'.format(100 * precision_score(y_true, y_pred, average='weighted',
                                                                       zero_division=0))
    information += 'Recall: {:.2f}%\n'.format(100 * recall_score(y_true, y_pred, average='weighted', zero_division=1))
    information += 'F1-Score: {:.2f}%\n'.format(100 * f1_score(y_true, y_pred, average='weighted', zero_division=0))
    information += 'Accuracy Mean by K-Fold: {:.2f}%\n'.format(100 * np.mean(cross_val_score(model_t, x_df_t, y_df_t,
                                                                                             cv=10)))

    # Salva em um arquivo somente com as métricas
    metrics_information = 'Nome do modelo: {}\n{}'.format(model_name_t, information)
    single_information(metrics_information, '{}Métricas_e_Resumo_Dados'.format(folder_t))

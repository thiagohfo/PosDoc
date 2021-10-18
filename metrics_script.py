from data_info_save import *
from plot_functions import *
from useful_functions import *


# Cálculo das métricas
def metrics_calc(x_data_t, y_data_t, model_t, folder_t, file_fullname_t):
    y_true = y_data_t
    y_pred = model_t.predict(x_data_t)
    y_probs = model_t.decision_function(x_data_t)

    # Salva o plot da Curva ROC
    roc_curve_plot(y_true, y_probs, folder_t)

    # Salvando informações em arquivos
    information = 'Accuracy: {:.2f}%\n'.format(100 * model_t.score(x_data_t, y_data_t))
    information += 'Precision: {:.2f}%\n'.format(100 * precision_score(y_true, y_pred, average='weighted', zero_division=0))
    information += 'Recall: {:.2f}%\n'.format(100 * precision_score(y_true, y_pred, average='weighted', zero_division=1))
    information += 'F1-Score: {:.2f}%\n'.format(100 * f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Salva no mesmo arquivo com outras informações
    single_information(information, file_fullname_t)

    # Salva em um arquivo somente com as métricas
    metrics_information = 'Nome da base: {}\n{}'.format(str(re.findall("/.*/", file_fullname_t))[3:-3], information)
    single_information(metrics_information, 'Bases/Métricas')
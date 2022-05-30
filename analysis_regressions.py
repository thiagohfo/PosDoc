"""Análises de regressão, contendo Regressão Linear, Ridge, Lasso e Regressão Logística"""
import numpy as np
# import statsmodels.api as sm
import matplotlib.pyplot as plt
from useful_functions import saving_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression


# Predição Regressão Linear
def lr_prediction(data_t_X, data_t_y, test_size_t=0.3, random_state_t=0):
    X_train, X_test, y_train, y_test = train_test_split(data_t_X, data_t_y, test_size=test_size_t,
                                                        random_state=random_state_t)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Salvando o modelo
    saving_model(lr, 'linear_model')

    # Dados da regressão
    # y_pred = lr.predict(X_test)
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)

    # Print dados da regressão linear
    print("Coeficiente LR: {}".format(lr.coef_))
    print("Train Score: {:.2f}%".format(train_score * 100))
    print("Test Score: {:.2f}%".format(test_score * 100))

    # Predição e aplicação da Regressão pelo StatsModels
    # X = sm.add_constant(X)
    # ols = sm.OLS(data_t_y, data_t_x).fit()
    # print(ols.summary())
    # print(ols.predict(X))

    return lr.coef_


# Predição Ridge
def rr_prediction(data_t_X, data_t_y, alpha_t=None, test_size_t=0.3, random_state_t=0):
    if alpha_t is None:
        alpha_t = [0.01, 1, 100]

    rr_coef = []

    for i in alpha_t:
        X_train, X_test, y_train, y_test = train_test_split(data_t_X, data_t_y, test_size=test_size_t,
                                                            random_state=random_state_t)
        rr = Ridge(alpha=i)
        rr.fit(X_train, y_train)
        train_score = rr.score(X_train, y_train)
        test_score = rr.score(X_test, y_test)
        coeff_used = np.sum(rr.coef_ != 0)

        print("Training score: {:.2f}%".format(train_score * 100))
        print("Test score: {:.2f}%".format(test_score * 100))
        print("Número de características usadas: {}".format(coeff_used))

        rr_coef.append(rr.coef_)

    lr_coef = lr_prediction(data_t_X, data_t_y)

    plt.plot(rr_coef[0], alpha=0.7, linestyle='none', marker='*', markersize=5, color='red',
             label=r'Ridge; $\alpha = 0.01$', zorder=7)
    plt.plot(rr_coef[1], alpha=0.5, linestyle='none', marker='d', markersize=6, color='blue',
             label=r'Ridge; $\alpha = 1$')
    plt.plot(rr_coef[2], alpha=0.8, linestyle='none', marker='v', markersize=6, color='black',
             label=r'Ridge; $\alpha = 100$')
    plt.plot(lr_coef, alpha=0.7, linestyle='none', marker='o', markersize=5, color='green',
             label='Linear Regression', zorder=2)
    plt.xlabel('Coeficientes', fontsize=16)
    plt.ylabel('Magnitude do Coeficiente', fontsize=16)
    plt.legend(fontsize=13, loc=4)
    plt.tight_layout()
    plt.show()


# Predição Lasso
def lasso_prediction(data_t_X, data_t_y, alpha_t=None, test_size_t=0.3, random_state_t=0):
    if alpha_t is None:
        alpha_t = [1, 0.01, 0.00001]

    max_iteration = 10e4  # 10e5 é o recomendado
    lasso_coef = []

    for i in alpha_t:
        X_train, X_test, y_train, y_test = train_test_split(data_t_X, data_t_y, test_size=test_size_t,
                                                            random_state=random_state_t)
        lasso = Lasso(alpha=i, max_iter=max_iteration)
        lasso.fit(X_train, y_train)
        train_score = lasso.score(X_train, y_train)
        test_score = lasso.score(X_test, y_test)
        coeff_used = np.sum(lasso.coef_ != 0)

        print("Training score: {:.2f}%".format(train_score * 100))
        print("Test score: {:.2f}%".format(test_score * 100))
        print("Número de características usadas: {}".format(coeff_used))

        lasso_coef.append(lasso.coef_)

    lr_coef = lr_prediction(data_t_X, data_t_y)

    plt.plot(lasso_coef[0], alpha=0.7, linestyle='none', marker='*', markersize=5, color='red',
             label=r'Lasso; $\alpha = 1$', zorder=7)
    plt.plot(lasso_coef[1], alpha=0.5, linestyle='none', marker='d', markersize=6, color='blue',
             label=r'Lasso; $\alpha = 0.01$')
    plt.plot(lasso_coef[2], alpha=0.8, linestyle='none', marker='v', markersize=6, color='black',
             label=r'Lasso; $\alpha = 0.00001$')
    plt.plot(lr_coef, alpha=0.7, linestyle='none', marker='o', markersize=5, color='green',
             label='Linear Regression', zorder=2)
    plt.xlabel('Coeficientes', fontsize=16)
    plt.ylabel('Magnitude do Coeficiente', fontsize=16)
    plt.legend(fontsize=13, loc=4)
    plt.tight_layout()
    plt.show()


# Predição Regressão Logística
def logistic_prediction(df_t_X, df_t_y, kind_t, test_size_t=0.3, random_state_t=0):
    X_train, X_test, y_train, y_test = train_test_split(df_t_X, df_t_y, test_size=test_size_t,
                                                        random_state=random_state_t)
    logistic = LogisticRegression(solver='liblinear', random_state=0)
    logistic.fit(X_train, y_train)

    # Salvando o modelo
    saving_model(logistic, 'logistic_model_{}'.format(kind_t))

    # Dados da regressão
    # y_pred = logistic.predict(X_test)
    train_score = logistic.score(X_train, y_train)
    test_score = logistic.score(X_test, y_test)

    # Print dados da regressão linear
    print("Train Score: {:.2f}%".format(train_score * 100))
    print("Test Score: {:.2f}%".format(test_score * 100))

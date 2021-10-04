from plot_functions import *
from useful_functions import *


# Lendo a instância
file = 'Bases/dados-ce-1.csv'
data = pd.read_csv(file, sep=';')
print(len(data))


# Predição
def prediction(X, Y, alpha_t, test_size_t=0.3, random_state_t=0):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size_t, random_state=random_state_t)
    rr = Ridge(alpha=alpha_t)
    rr.fit(x_train, y_train)
    train_score = rr.score(x_train, y_train)
    test_score = rr.score(x_test, y_test)
    coeff_used = np.sum(rr.coef_ != 0)
    print("Training score: {}".format(train_score))
    print("Test score: {}".format(test_score))
    print("Número de características usadas: {}".format(coeff_used))

    return rr.coef_


# Regressão Linear predição
def prediction_lr(X, Y, test_size_t=0.3, random_state_t=0):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size_t, random_state=random_state_t)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    yhat = lr.predict(x_test)
    train_score = lr.score(x_train, y_train)
    test_score = lr.score(x_test, y_test)
    print("Predição: {}".format(yhat))
    print("Train Score: {}".format(train_score))
    print("Test Score: {}".format(test_score))
    return lr.coef_


# Variáveis
conditions = ['cardiacas', 'diabetes', 'respiratorias', 'renais', 'imunologica', 'obesidade', 'imunossupressao']
symptoms = ['tosse', 'febre', 'garganta', 'dispneia', 'cabeca', 'coriza']
others = ['tipoTeste', 'evolucaoCaso', 'profissionalSaude', 'diasSintomas', 'sexo', 'idade']


# Pega somente rows com testes positivos ou negativos
delete_rows_by_value(data, 'resultadoTeste', 0, False)


# Trocas os tipos para float
X = data[symptoms].astype(float)
Y = data['resultadoTeste'].astype(float)


#
rr_coef = []
for i in [1, 0.01, 100]:
    rr_coef.append(prediction(data[symptoms], data['resultadoTeste'], i))

lr_coef = prediction_lr(data[symptoms], data['resultadoTeste'])

plt.plot(rr_coef[0], alpha=0.7, linestyle='none', marker='*', markersize=5, color='red', label=r'Ridge; $\alpha = 1$', zorder=7)
plt.plot(rr_coef[1], alpha=0.5, linestyle='none', marker='d', markersize=6, color='blue', label=r'Ridge; $\alpha = 0.01$')
plt.plot(rr_coef[2], alpha=0.8, linestyle='none', marker='v', markersize=6, color='black', label=r'Ridge; $\alpha = 100$')
plt.plot(lr_coef, alpha=0.7, linestyle='none', marker='o', markersize=5, color='green', label='Linear Regression', zorder=2)
plt.xlabel('Coeficientes', fontsize=16)
plt.ylabel('Magnitude do Coeficiente', fontsize=16)
plt.legend(fontsize=13, loc=4)
plt.tight_layout()
plt.show()
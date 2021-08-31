from useful_functions import *


# Lendo a instância
file = 'Bases/dados-ce-1.csv'
data = pd.read_csv(file, sep=';')
print(len(data))


# Predição
def prediction(X, Y, test_size_t=0.3, random_state_t=0):
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


# Predição e aplicação da Regressão pelo StatsModels
#X = sm.add_constant(X)
ols = sm.OLS(Y, X).fit()
print(ols.summary())
#print(ols.predict(X))


# Printa predição do modelo linear
lr_coef = prediction(data[symptoms], data['resultadoTeste'])
print(lr_coef)


exit(0)
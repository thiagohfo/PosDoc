from useful_functions import saving_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def naive_bayes(df_t_X, df_t_y, kind_t, test_size_t=0.3, random_state_t=0):
    X_train, X_test, y_train, y_test = train_test_split(df_t_X, df_t_y, test_size=test_size_t,
                                                        random_state=random_state_t)

    nb = BernoulliNB()
    nb.fit(X_train, y_train)

    saving_model(nb, 'naive_bayes_model_{}'.format(kind_t))

    train_score = nb.score(X_train, y_train)
    test_score = nb.score(X_test, y_test)

    # Print dados da regressão linear
    print("Train Score: {:.2f}%".format(train_score * 100))
    print("Test Score: {:.2f}%".format(test_score * 100))


def decision_tree(df_t_X, df_t_y, kind_t, test_size_t=0.3, random_state_t=0):
    X_train, X_test, y_train, y_test = train_test_split(df_t_X, df_t_y, test_size=test_size_t,
                                                        random_state=random_state_t)

    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=7, random_state=random_state_t)
    dt.fit(X_train, y_train)

    saving_model(dt, 'decision_tree_model_{}'.format(kind_t))

    train_score = dt.score(X_train, y_train)
    test_score = dt.score(X_test, y_test)

    # Print dados da regressão linear
    print("Train Score: {:.2f}%".format(train_score * 100))
    print("Test Score: {:.2f}%".format(test_score * 100))
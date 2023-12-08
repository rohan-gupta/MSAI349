import joblib
from sklearn.model_selection import train_test_split

from src.data.data import load_organic, load_synthetic
from src.configs.configs import *
from src.tests.tests import *
from src.models.logistic_regression import train_logistic_regression
from src.models.random_forest import train_random_forest
from src.models.knn import train_knn
from src.models.neural_net import train_neural_net


def load(dname='organic'):
    df = load_organic() if dname == 'organic' else load_synthetic()
    return df


def train_and_test(df, dname='organic'):
    X, y = df.drop('defects', axis=1), df['defects']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_classifiers = [
        (train_logistic_regression, get_logistic_regression_config),
        (train_random_forest, get_random_forest_config),
        (train_knn, get_knn_config),
        (train_neural_net, get_neural_net_config),
    ]

    for clf in train_classifiers:
        t, c = clf

        print("Executing", str(t.__name__))
        pipeline, best_params = t(X_train, y_train, c(), 3, 'f1')

        print("Saving", str(t.__name__))
        model_name = "_".join(t.__name__.split("_")[1:])
        joblib.dump(pipeline, f'./src/models/{dname}/{model_name}.pkl')

        print('Params\n', best_params)

    test(X_test, y_test, y, dname)


def test(X_test, y_test, y, dname='organic'):
    test_logistic_regression(X_test, y_test, y, dname)
    test_neural_net(X_test, y_test, y, dname)
    test_random_forest(X_test, y_test, y, dname)
    test_knn(X_test, y_test, y, dname)


if __name__ == '__main__':
    data = load('organic')
    train_and_test(data, 'organic')

    # data = load('synthetic')
    # train_and_test(data, 'synthetic')

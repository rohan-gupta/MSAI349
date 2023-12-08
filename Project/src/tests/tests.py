import joblib
import numpy as np
from keras.models import load_model
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.pipeline import Pipeline

import pickle

def test_logistic_regression(X_test, y_test, y, dname='organic'):
    print('Logistic Regression')
    pipeline = joblib.load(f'./src/models/{dname}/logistic_regression.pkl')

    y_pred = pipeline.predict(X_test)
    accuracy, baseline_accuracy = get_accuracy(y, y_test, y_pred)
    report = classification_report(y_test, y_pred)

    log_params(pipeline)
    log_metrics(accuracy, baseline_accuracy, report)

def test_neural_net(X_test, y_test, y, dname='organic'):
    print('Neural Net')
    pipeline = joblib.load(f'./src/models/{dname}/neural_net.pkl')
    neural_net = load_model(f'./src/models/{dname}/neural_net.keras')
    neural_net = KerasClassifier(neural_net)

    X = pickle.load(open(f'./src/models/{dname}/neural_net_data', 'rb'))['X']
    y = np.random.choice([True, False], size=X.shape[0])

    neural_net.initialize(X, y)

    steps = list(pipeline.steps)
    steps.append(('neural_net', neural_net))
    pipeline = Pipeline(steps)

    y_pred = pipeline.predict(X_test)

    accuracy, baseline_accuracy = get_accuracy(y, y_test, y_pred)
    report = classification_report(y_test, y_pred)

    log_metrics(accuracy, baseline_accuracy, report)


def test_random_forest(X_test, y_test, y, dname='organic'):
    print('Random Forest')

    pipeline = joblib.load(f'./src/models/{dname}/random_forest.pkl')

    y_pred = pipeline.predict(X_test)
    accuracy, baseline_accuracy = get_accuracy(y, y_test, y_pred)
    report = classification_report(y_test, y_pred)

    log_params(pipeline)
    log_metrics(accuracy, baseline_accuracy, report)


def test_knn(X_test, y_test, y, dname='organic'):
    print("KNN")

    pipeline = joblib.load(f'./src/models/{dname}/knn.pkl')

    y_pred = pipeline.predict(X_test)
    accuracy, baseline_accuracy = get_accuracy(y, y_test, y_pred)
    report = classification_report(y_test, y_pred)

    log_params(pipeline)
    log_metrics(accuracy, baseline_accuracy, report)


def get_accuracy(y, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    random_pred = np.random.choice(np.unique(y), size=len(y_test))
    random_accuracy = accuracy_score(y_test, random_pred)

    return accuracy, random_accuracy


def log_metrics(accuracy, baseline_accuracy, report):
    print("Accuracy\n", accuracy)
    print("Baseline Accuracy\n", baseline_accuracy)
    print("Classification Report\n", report)


def log_params(pipeline):
    print("Parameters\n", pipeline.get_params())

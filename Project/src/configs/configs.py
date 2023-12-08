def get_logistic_regression_config():
    return {
        'pca__n_components': [0.95],
        'logreg__C': [0.1, 1, 10],
        'logreg__solver': ['lbfgs', 'sag', 'saga'],
        'logreg__max_iter': [100, 200, 500],
        'logreg__class_weight': ['balanced'],
        'logreg__random_state': [42],
    }


def get_random_forest_config():
    return {
        'pca__n_components': [0.90, 0.95],
        'forest__n_estimators': [100, 200],
        'forest__max_depth': [None, 10, 20, 30],
        'forest__criterion': ['gini', 'entropy', 'log_loss'],
        'forest__class_weight': ['balanced'],
    }


def get_knn_config():
    return {
        'pca__n_components': [0.90, 0.95],
        'knn__n_neighbors': [5, 10],
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    }


def get_neural_net_config():
    return {
        'neural_net__epochs': [50, 100],
        'neural_net__batch_size': [32],
        'neural_net__optimizer': ['adam', 'sgd'],
        'neural_net__validation_split': [0.25],
    }

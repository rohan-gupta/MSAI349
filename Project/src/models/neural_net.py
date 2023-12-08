from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2

from scikeras.wrappers import KerasClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import pickle

from .utils.log_transformer import LogTransformer

def train_neural_net(X, y, param_grid, cv, scoring):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
    neural_net = KerasClassifier(model=build_neural_net, verbose=1, callbacks=[early_stopping])

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE()),
        ('pca', PCA(n_components=15)),
        ('saver', DataSaver()),
        ('neural_net', neural_net)
    ])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1, cv=cv, scoring=scoring)
    grid_search.fit(X, y)

    pipeline = grid_search.best_estimator_
    history = pipeline.named_steps['neural_net'].model_.history
    save_history(history)

    steps = list(pipeline.steps)
    steps = [step for step in steps if step[0] != 'neural_net']

    pipeline.steps[-1][1].model_.save('./src/models/organic/neural_net.keras')

    return Pipeline(steps), grid_search.best_params_


def build_neural_net(optimizer='adam', learning_rate=0.001, dropout_rate=0.5):
    neural_net = Sequential()
    neural_net.add(Input(shape=(15,)))
    neural_net.add(Dense(11, activation='relu'))
    neural_net.add(Dropout(dropout_rate))
    neural_net.add(Dense(11, activation='relu'))
    neural_net.add(Dropout(dropout_rate))
    neural_net.add(Dense(1, activation='sigmoid'))

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)

    if optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)

    neural_net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return neural_net


class DataSaver(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data_to_save = {'X': X}
        if y is not None:
            data_to_save['y'] = y
        pickle.dump(data_to_save, open('./src/models/organic/neural_net_data', 'wb'))
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)


def save_history(history):
    pickle.dump(history, open('./src/models/organic/neural_net_history', 'wb'))

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from .utils.log_transformer import LogTransformer

def train_logistic_regression(X, y, param_grid, cv, scoring):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE()),
        ('pca', PCA()),
        ('logreg', LogisticRegression())
    ])

    pipeline.fit(X, y)

    grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=cv, scoring=scoring)
    grid_search.fit(X, y)

    return pipeline, pipeline.get_params()

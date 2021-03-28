from algothon2021ml import ML
import pandas as pd
import numpy as np 
from sklearn import (
    preprocessing,
    svm,
    feature_selection,
    neural_network,
    gaussian_process,
    pipeline,
    decomposition,
    model_selection,
)
from scipy import stats


from data.read_data import read_data_val

param_grid = [
    {
        "hidden_layer_sizes": [
            (10, 20, 20, 20, 10),
            (50, 50, 50, 50, 50),
            (100, 200, 200, 200, 200, 50),
            (50, 50, 50, 50, 50, 50, 20, 10),
            (400, 400, 400, 400, 400, 200, 100),
        ],
        "alpha": [0.001, 0.005, 0.01],
    }
]


class SimpleNNPredictor(ML):
    def __init__(self, hp_opt: bool, **params):
        super(SimpleNNPredictor, self).__init__(**params)

        self.scaler = preprocessing.StandardScaler()
        self.normalizer = preprocessing.Normalizer()
        self.pca = decomposition.PCA(n_components=20, random_state=123)

        self.hp_opt = hp_opt

        self.model = neural_network.MLPRegressor(
            hidden_layer_sizes=(500, 500, 500, 500, 500, 200, 200, 200),
            activation="relu",
            random_state=123,
            solver="adam",
            alpha=0.002,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=400,
            shuffle=True,
            tol=0.0001,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=True,
            validation_fraction=0.2,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            n_iter_no_change=10,
            max_fun=15000,
        )
        # self.model = svm.SVR(kernel='rbf', C=1, gamma=0.1, epsilon=.1)
        # self.model = gaussian_process.GaussianProcessRegressor(random_state=123)

        if self.hp_opt:
            self.model = model_selection.GridSearchCV(
                n_jobs=8,
                estimator=self.model,
                param_grid=param_grid,
                scoring="explained_variance",
            )

        self.pipeline = pipeline.make_pipeline(self.model)

    def train(self, X_train, y_train):
        # Fit scaler and transform
        X_train = self.scaler.fit_transform(X_train)
        # Fit PCA and transform
        X_train = self.pca.fit_transform(X_train)
        # Fit pipeline
        self.pipeline.fit(X_train, y_train)

    def predict(self, X) -> float:
        if 'Market' in X:
            # We're in prediction mode
            X = read_data_val(X)
        # Transform data by previously fitted scaler
        X = self.scaler.transform(X)
        # Transform data by previously fitted PCA
        X = self.pca.transform(X)
        prediction = self.pipeline.predict(X)
        greater_idx = (prediction>0.05)
        smaller_idx = (prediction<0.05)
        prediction[greater_idx] = 1.0
        prediction[smaller_idx] = -1.0
        prediction[(~greater_idx) & (~smaller_idx)] = 0
        prediction = prediction.item() if prediction.size == 1 else prediction
        return prediction

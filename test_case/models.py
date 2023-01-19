import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from lightgbm import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from constants import *
from helper_functions import *

tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
np.random.seed(seed)

class LGB:
    def __init__(self, data, use_class_weights=False, *args, **kwargs):
        Y = data.loc[:, 'target']
        X = data.loc[:, data.columns != 'target']

        x_train, self.x_test, y_train, self.y_test = \
            train_test_split(X.values, Y.values, train_size=0.85, random_state=kwargs['random_state'] if 'random_state' in kwargs.keys() else None)

        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(x_train, y_train, train_size=0.85, random_state=kwargs['random_state'] if 'random_state' in kwargs.keys() else None)

        self.class_weights = {0: self.y_train[self.y_train == 0].size / self.y_train.size,\
                              1: self.y_train[self.y_train == 1].size / self.y_train.size}\
                              if use_class_weights else None

        self.model = LGBMClassifier(class_weight=self.class_weights, *args, **kwargs)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.model.predict(X)

    def display_metrics(self, X, y_true, subset_name='Val', show_confusion_matrix=False):
        y_pred = self.predict(X)
        recall_metric = recall_score(y_true, y_pred)
        precision_metric = precision_score(y_true, y_pred)
        accuracy_metric = accuracy_score(y_true, y_pred)
        f1_metric = f1_score(y_true, y_pred)

        print(f'---{subset_name} Subset Results---')
        print("Recall score:", recall_metric)
        print("Precision score:", precision_metric)
        print("Accuracy score:", accuracy_metric)
        print("F1 score:", f1_metric)
        print('\n')
        if show_confusion_matrix:
            plt.figure(figsize=(14, 8))
            matrix = confusion_matrix(y_true, y_pred)
            strings = ['TN', 'FP', 'FN', 'TP']
            labels = np.asarray(["{0} {1:.3f}".format(string, value) for string, value in
                                  zip(strings, matrix.flatten())]).reshape(2, 2)

            sns.heatmap(matrix, annot=labels, fmt='')
            plt.title(f"Confusion Matrix for {subset_name} Subset", fontsize=14)
            plt.show()

        return f1_metric

    def validate(self, show_confusion_matrix=False):
        recall_metric = self.display_metrics(self.x_val, self.y_val, show_confusion_matrix=show_confusion_matrix)

        return recall_metric

    def test(self, show_confusion_matrix=False):
        self.display_metrics(self.x_test, self.y_test, subset_name='Test', show_confusion_matrix=show_confusion_matrix)

    def predict(self, sample, *args):
        return self.model.predict(sample, *args)

class GradientBoost(LGB):
    def __init__(self, data, *args, **kwargs):
        Y = data.loc[:, 'target']
        X = data.loc[:, data.columns != 'target']

        x_train, self.x_test, y_train, self.y_test = train_test_split(X.values, Y.values, train_size=0.85, random_state=seed)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, train_size=0.85, random_state=seed)

        self.model = GradientBoostingClassifier(*args, **kwargs)

class NeuralNetwork(LGB):
    def __init__(self, data, *args, **kwargs):
        Y = data.loc[:, 'target']
        X = data.loc[:, data.columns != 'target']
        X = X[scalar_columns + categorical_columns]

        x_train, self.x_test, y_train, self.y_test = train_test_split(X.values, Y.values, train_size=0.85, random_state=seed)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, train_size=0.85, random_state=seed)

        self.scaler, X_scaled = scale_dataset(self.x_train[:, :len(scalar_columns)])
        self.onehot_encoder, X_onehot = onehot_dataset(self.x_train[:, len(scalar_columns):])
        self.x_train = np.concatenate((X_scaled, X_onehot), axis=1)

        self.model = Sequential([Dense(512, activation='relu', input_shape=(self.x_train.shape[1],)),
                                 Dense(256, activation='relu'),
                                 Dense(128, activation='relu'),
                                 Dense(64, activation='relu'),
                                 Dense(1, activation='sigmoid')])

        self.class_weights = {0: self.y_train[self.y_train == 0].size / self.y_train.size, 1: self.y_train[self.y_train == 1].size / self.y_train.size}

        self.model.compile(loss='binary_crossentropy', optimizer=Adam(*args, **kwargs))

    def fit(self):
        self.model.fit(self.x_train, self.y_train, batch_size=1024, epochs=1000, verbose=1, class_weight=self.class_weights)

    def predict(self, X):
        X_scaled = self.scaler.transform(X[:, :len(scalar_columns)])
        X_onehot = self.onehot_encoder.transform(X[:, len(scalar_columns):])
        X = np.concatenate((X_scaled, X_onehot), axis=1)

        Y_pred = self.model.predict(X)
        Y_categorical = np.round(Y_pred)

        return Y_categorical


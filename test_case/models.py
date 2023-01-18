from lightgbm import *
from sklearn.model_selection import train_test_split
from constants import *
from sklearn.metrics import *

class LGBM_model:
    def __init__(self, data, use_class_weights=False, *args, **kwargs):
        Y = data.loc[:, 'target']
        X = data.loc[:, data.columns != 'target']

        x_train, self.x_test, y_train, self.y_test = train_test_split(X.values, Y.values, train_size=0.85, random_state=seed)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, train_size=0.85, random_state=seed)

        self.class_weights = {0: self.y_train[self.y_train == 0].size / self.y_train.size,\
                              1: self.y_train[self.y_train == 1].size / self.y_train.size}\
                              if use_class_weights else None

        self.model = LGBMClassifier(class_weight=self.class_weights, *args, **kwargs)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def validate(self):
        y_pred = self.model.predict(self.x_val)
        recall_metric = recall_score(self.y_val, y_pred)
        precision_metric = precision_score(self.y_val, y_pred)
        accuracy_metric = accuracy_score(self.y_val, y_pred)

        print('---Validation Subset Results---')
        print("Recall score:", recall_metric)
        print("Precision score:", precision_metric)
        print("Accuracy score:", accuracy_metric)
        print('\n')
        return recall_metric

    def test(self):
        y_pred = self.model.predict(self.x_test)
        recall_metric = recall_score(self.y_test, y_pred)
        precision_metric = precision_score(self.y_test, y_pred)
        accuracy_metric = accuracy_score(self.y_test, y_pred)

        print('---Test Subset Results---')
        print("Recall score:", recall_metric)
        print("Precision score:", precision_metric)
        print("Accuracy score:", accuracy_metric)
        print('\n')

    def predict(self, sample, *args):
        return self.model.predict(sample, *args)

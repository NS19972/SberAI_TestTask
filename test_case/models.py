# Данный модуль содержит все виды моделей, которые можно использовать и обучить в файле main.py
# Также он позволяет легко создавать новые классы моделей, используя наследие классов (class inheritance)


# Импортируем библиотеки
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from lightgbm import *
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from constants import *
from helper_functions import *

# Задаем сид, чтобы затем можно было воспроизвести результат
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
np.random.seed(seed)


# Класс для работы с алгоритмом LGBMclassifier
# Данный алгоритм НЕ ТРЕБУЕТ скалирования или onehot-кодирования входных данных
class LGB:
    def __init__(self, data, use_class_weights=False, *args, **kwargs):
        Y = data.loc[:, 'target']                   # Выбираем столбец с целевыми значениями
        X = data.loc[:, data.columns != 'target']   # Выбираем столбцы с фичами

        # Делим датасет на обучающую + валидационную выборку и тестовую выборку
        x_train, self.x_test, y_train, self.y_test = \
            train_test_split(X.values, Y.values, train_size=0.85, random_state=kwargs['random_state'] if 'random_state' in kwargs.keys() else None)

        # Делим обучающую + валидационную выборку на отдельную обучающую и валидационную выборку
        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(x_train, y_train, train_size=0.85, random_state=kwargs['random_state'] if 'random_state' in kwargs.keys() else None)

        # Задаем относительные веса классов (только если используется параметр use_class_weights)
        self.class_weights = {0: self.y_train[self.y_train == 0].size / self.y_train.size,\
                              1: self.y_train[self.y_train == 1].size / self.y_train.size}\
                              if use_class_weights else None

        # Создаем класс модели (гиперпараметры были подобраны вручную)
        self.model = LGBMClassifier(class_weight=self.class_weights, max_depth=5, *args, **kwargs)

    # Метод для обучения алгоритма
    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    # Воспомогательная функция - позволяет вывести все метрики и матрицу путаницы одной строкой кода
    # (данная функция применяется к валидационной и к тестовой выборке)
    def display_metrics(self, X, y_true, subset_name='Val', show_confusion_matrix=False):
        y_pred = self.predict(X)                             # Получаем предсказание модели
        recall_metric = recall_score(y_true, y_pred)         # Считаем значение метрики Recall
        precision_metric = precision_score(y_true, y_pred)   # Считаем значение метрики Precision
        accuracy_metric = accuracy_score(y_true, y_pred)     # Считаем значение метрики Accuracy
        f1_metric = f1_score(y_true, y_pred)                 # Считаем значение метрики F1

        # Выводим значение всех полученных метрик, а также название выборки на котором они считались
        print(f'---{subset_name} Subset Results---')
        print("Recall score:", recall_metric)
        print("Precision score:", precision_metric)
        print("Accuracy score:", accuracy_metric)
        print("F1 score:", f1_metric)
        print('\n')
        if show_confusion_matrix:  # Если хотим вывести матрицу корреляций:
            plt.figure(figsize=(14, 8))  # Создаем полотно
            matrix = confusion_matrix(y_true, y_pred)  # Считаем значения матрицы путаницы
            strings = ['TN', 'FP', 'FN', 'TP']  # Название категорий (True Negative, False Positive, False Negative, True Positive)

            # Создаем лейблы для матрицы корреляции - название категории в матрице затем количество данной категории
            labels = np.asarray(["{0} {1:.3f}".format(string, value)
                                for string, value in zip(strings, matrix.flatten())]).reshape(2, 2)

            sns.heatmap(matrix, annot=labels, fmt='')  # Отрисовываем матрицу путаницы
            plt.title(f"Confusion Matrix for {subset_name} Subset", fontsize=14)  # Устанавливаем заголовок
            plt.show()  # Отображаем результат

        return f1_metric # Возвращаем F1 метрику (для валидационной выборки)

    # Метод для оценки работы модели на валидационной выборки
    def validate(self, show_confusion_matrix=False):
        F1_metric = self.display_metrics(self.x_val, self.y_val, show_confusion_matrix=show_confusion_matrix)

        return F1_metric  # Возвращаем F1 метрику (для дальнейшей интеграции с алгоритмами поиска гиперпараметров, например Optuna)

    # Метод для оценки работы модели на тестовой выборки
    def test(self, show_confusion_matrix=False):
        self.display_metrics(self.x_test, self.y_test, subset_name='Test', show_confusion_matrix=show_confusion_matrix)

    # Метод для инференса обученного алгоритма
    def predict(self, sample, *args):
        return self.model.predict(sample, *args)

# Класс для работы с алгоритмом GradientBoosting
# Данный алгоритм НЕ ТРЕБУЕТ скалирования или onehot-кодирования входных данных
class GradientBoost(LGB):
    def __init__(self, data, *args, **kwargs):
        Y = data.loc[:, 'target']                   # Выбираем столбец с целевыми значениями
        X = data.loc[:, data.columns != 'target']   # Выбираем столбцы с фичами

        # Делим датасет на обучающую + валидационную выборку и тестовую выборку
        x_train, self.x_test, y_train, self.y_test = train_test_split(X.values, Y.values, train_size=0.85, random_state=seed)
        # Делим обучающую + валидационную выборку на отдельную обучающую и валидационную выборку
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, train_size=0.85, random_state=seed)

        # Создаем класс модели (гиперпараметры были подобраны вручную)
        self.model = GradientBoostingClassifier(learning_rate=3e-1, n_estimators=100,
                                                max_depth=3, min_samples_leaf=5,
                                                *args, **kwargs)

# Класс для работы с алгоритмом RandomForest
# Данный алгоритм НЕ ТРЕБУЕТ скалирования или onehot-кодирования входных данных
class RandomForest(LGB):
    def __init__(self, data, use_class_weights=False, *args, **kwargs):
        Y = data.loc[:, 'target']                   # Выбираем столбец с целевыми значениями
        X = data.loc[:, data.columns != 'target']   # Выбираем столбцы с фичами

        # Делим датасет на обучающую + валидационную выборку и тестовую выборку
        x_train, self.x_test, y_train, self.y_test = train_test_split(X.values, Y.values, train_size=0.85, random_state=seed)
        # Делим обучающую + валидационную выборку на отдельную обучающую и валидационную выборку
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, train_size=0.85, random_state=seed)

        # Задаем относительные веса классов (только если используется параметр use_class_weights)
        self.class_weights = {0: self.y_train[self.y_train == 0].size / self.y_train.size, \
                              1: self.y_train[self.y_train == 1].size / self.y_train.size} \
            if use_class_weights else None

        # Создаем класс модели (гиперпараметры были подобраны вручную)
        self.model = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=1,
                                            class_weight=self.class_weights, *args, **kwargs)

# Класс для работы с нейронной сетью (в данном случае логично использовать только полносвязанные сети)
# Данный алгоритм ТРЕБУЕТ (!!!) скалирования и onehot-кодирования входных данных
class NeuralNetwork(LGB):
    def __init__(self, data, *args, **kwargs):
        Y = data.loc[:, 'target']                   # Выбираем столбец с целевыми значениями
        X = data.loc[:, data.columns != 'target']   # Выбираем столбцы с фичами
        X = X[scalar_columns + categorical_columns] # Перестанавливаем порядок столбцов, чтобы их можно было легко индексировать когда переведём в numpy массив

        # Делим датасет на обучающую + валидационную выборку и тестовую выборку
        x_train, self.x_test, y_train, self.y_test = train_test_split(X.values, Y.values, train_size=0.85, random_state=seed)
        # Делим обучающую + валидационную выборку на отдельную обучающую и валидационную выборку
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, train_size=0.85, random_state=seed)

        self.scaler, X_scaled = scale_dataset(self.x_train[:, :len(scalar_columns)])           # Скалируем все скалярные фичи, записываем объект скейлера
        self.onehot_encoder, X_onehot = onehot_dataset(self.x_train[:, len(scalar_columns):])  # onehot-кодируем все категориальные фичи, записываем объект кодировщика
        self.x_train = np.concatenate((X_scaled, X_onehot), axis=1)  # Конкатенируем все скалярные и категориальные фичи в один массив

        # Создаем архитектуру нейронной сети - глубокая модель без Dropout, BatchNormalization, или регуляризации показала наилучшие результаты на валидационной выборке
        self.model = Sequential([Dense(512, activation='relu', input_shape=(self.x_train.shape[1],)),
                                 Dense(256, activation='relu'),
                                 Dense(128, activation='relu'),
                                 Dense(64, activation='relu'),
                                 Dense(1, activation='sigmoid')])

        # Задаем относительные веса классов
        self.class_weights = {0: self.y_train[self.y_train == 0].size / self.y_train.size, 1: self.y_train[self.y_train == 1].size / self.y_train.size}

        self.model.compile(loss='binary_crossentropy', optimizer=Adam(*args, **kwargs))  # Компилируем модель

    # Метод обучения модели - большой размер батчей и большое количество эпох показало наилучшие результаты
    def fit(self):
        self.model.fit(self.x_train, self.y_train, batch_size=1024, epochs=1000, verbose=1, class_weight=self.class_weights)

    # Метод для инференса обученного алгоритма
    def predict(self, X):
        X_scaled = self.scaler.transform(X[:, :len(scalar_columns)])          # Скалируем все скалярные фичи входных данных
        X_onehot = self.onehot_encoder.transform(X[:, len(scalar_columns):])  # Кодируем все категориальные фичи входных данных
        X = np.concatenate((X_scaled, X_onehot), axis=1)  # Конкатенируем скалярные и категориальные фичи

        Y_pred = self.model.predict(X)    # Получаем предсказание модели
        Y_categorical = np.round(Y_pred)  # Округляем предсказание чтобы получить индекс класса

        return Y_categorical  # Возвращаем предсказание


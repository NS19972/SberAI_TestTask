# Импортируем модули/библиотеки
import pandas as pd
from models import *

if __name__ == '__main__':
    data = pd.read_csv('test.csv', index_col=0)  # Читаем датасет

    # Выбираем модель из [LGB, GradientBoost, RandomForest, NeuralNetwork] (гиперпараметры внутри класса модели)
    classifier = LGB(data)

    classifier.fit()  # Обучаем
    classifier.validate(show_confusion_matrix=True)  # Валидируем
    classifier.test(show_confusion_matrix=True)  # Тестируем

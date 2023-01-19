# Файл содержит несколько воспомогательных функций

# Импортируем библиотеки
import pandas as pd
from sklearn.preprocessing import *

# Функция для кодирования датасета - принимает на вход только категориальные столбцы из датасета
def onehot_dataset(data):
    encoder = OneHotEncoder(sparse_output=False)  # Создаем объект кодировщика
    onehot_data = encoder.fit_transform(data)  # Обучаем кодировщик и трансформируем данные

    return encoder, onehot_data  # Возвращаем кодировщик и кодированные данные

def scale_dataset(data):
    scaler = StandardScaler()  # Создаем объект скейлера
    scaled_data = scaler.fit_transform(data)  # Обучаем кодировщик и трансформируем данные

    return scaler, scaled_data  # Возвращаем кодировщик и скалированные данные

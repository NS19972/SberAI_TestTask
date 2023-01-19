# Файл содержит различные функции для анализа/визуализации данных

# Импортируем библиотеки
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import *


# Функция для отображения матрицы корреляции
# Данная функция создает две матрицы - первая использует корреляцию Пирсона и работает только с категориальными данными,
# вторая использует корреляцию Спирмэна и использует её для оценки скалярным данных
def visualize_correlation_heatmap(data):
    categorical_data = data.loc[:, categorical_columns + ['target']]  # Берём все столбцы с категориальными данными
    scalar_data = data.loc[:, scalar_columns + ['target']]  # Берём все столбцы со скалярными данными (включая 'target' т.к. мы смотрим корреляцию именно по ней)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Создаем два полотна
    categorical_corr_matrix = categorical_data.corr(method='pearson')  # Считаем корреляцию Пирсона для категориальных данных
    scalar_corr_matrix = scalar_data.corr(method='spearman')  # Считаем корреляцию Спирмэна для скалярных данных
    sns.heatmap(categorical_corr_matrix, ax=axes[0], annot=True, vmin=-1, vmax=1)  # Отображаем первую матрицу в виде тепловой карты
    sns.heatmap(scalar_corr_matrix, ax=axes[1], annot=True, vmin=-1, vmax=1)  # Отображаем вторую матрицу в виде тепловой карты
    axes[0].set_title("Матрица корреляции для категориальных данных", fontsize=16)  # Добавляем заголовок для первой матрицы
    axes[1].set_title("Матрица корреляции для скалярных данных", fontsize=16)  # Добавляем заголовок для второй матрицы
    plt.show()  # Отображаем результат


# Функция для отображения кумулятивной гистограммы
# (даёт хорошее представление о распределении всех данных - нужно указать столбец, который хотим рассмотреть в column_name)
def create_histogram(data, column_name):
    assert column_name in data.columns, 'specified column not found in dataset.'

    plt.figure(figsize=(14, 8))  # Создаем полотно
    plt.hist(data[column_name], alpha=0.5, bins=100, color='red', cumulative=True)  # Рисуем кумулятивную гистограмму (большое количество bins делает это по факту отрисовкой CDF - cumulative distribution function)
    plt.title(f'Cumulative histogram for {column_name} column', fontsize=16)  # Задаем заголовок
    plt.xlabel('Column value')  # Записываем ось Х
    plt.ylabel('Cumulative frequency')  # Записываем ось У
    plt.grid()  # Добавляем клеточный вид
    plt.show()  # Отображаем результат


# Функция с помощью которой можно визуализировать распределение классов в виде гистограммы
def visualize_class_distribution(data):
    plt.figure(figsize=(14, 8))  # Создаем полотно
    plt.bar(('Class 0', 'Class 1'), np.bincount(data['target']), alpha=0.5, color='orange')  # Создаем гистограмму через plt.bar()
    plt.title(f'Class Frequency Graph', fontsize=16)  # Задаем заголовок
    plt.xlabel('Class Name', fontsize=12)  # Записываем ось Х
    plt.ylabel("Times encountered in dataset", fontsize=12)  # Записываем ось У
    plt.grid()  # Добавляем клеточный вид
    plt.show()  # Отображаем результат


if __name__ == '__main__':
    dataset = pd.read_csv('test.csv', index_col=0)  # Читаем датасет

    # Применяем любые желаемые функции для визуализации данных
    visualize_correlation_heatmap(dataset)
    create_histogram(dataset, 'feature_0')
    visualize_class_distribution(dataset)

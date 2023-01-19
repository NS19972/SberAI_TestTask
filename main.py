# Импортируем модули/библиотеки
import pandas as pd
from models import *

if __name__ == '__main__':
    data = pd.read_csv('test.csv', index_col=0)  # Читаем датасет

    # Выбираем модель из [LGB, GradientBoost, RandomForest, NeuralNetwork] (гиперпараметры внутри класса модели)
    classifier = GradientBoost(data)

    classifier.fit()  # Обучаем
    classifier.validate(show_confusion_matrix=True)  # Валидируем
    classifier.test(show_confusion_matrix=True)  # Тестируем

    ### Оценки работы различных алгоритмов на тестовой выборке (по метрике F1) ###
    ### Все ниже представленные числа показывают работу модели после небольшого подбора гиперпараметров на валидационной выборке ###

    # GradientBoosting F1 score: 0.60546875 <--- Выше чем на валидационной!
    # LGB F1 score: 0.588469184890656
    # Нейросеть F1 score: 0.5714285714285715 <--- Обучается намного дольше всех остальных моделей, по сколько 1000 эпох
    # Random Forest F1 score: 0.5110132158590308  <--- Выше чем на валидационной!

### Ответы на прочие вопросы из ноутбука:

# 3. Определите тип задачи: регрессия или классификация. Какие метрики качества потенциально можно использовать? Выберите метрики для определения качества данной модели

# Данная задача является бинарной классификацией с расбалансированной выборкой (~80% примеров пренадлежат нулевому классу)
# Выбор метрики зависит от пожелания к задачи (что важнее в контексте задания, определить Positive или Negative?)
# Итоговое решение рассматривает и Recall и Precision как важные метрики, по этому итоговая метрика - F1 score
# ----------------------------------------------------------------------------------------------------------------------------
# 8. В процессе разработки моделей проведите отбор признаков любыми методиками, которые считаете нужными

# Отбор признаков не производился, по сколько любые попытки это сделать приводили к ухудшению результатов нейросети
# При этом, классические ML-алгоритмы тоже не улучшали свою работоспособность при удалении фичей
# (хотя, матрица корреляции намекает на то, что две фичи практический никак не коррелируют с target)
# ----------------------------------------------------------------------------------------------------------------------------
# 9. Выведите итоговый результат: обязательны значения метрик на выбранных моделях, выбор лучшей модели и итоговый набор признаков.
# Наилучшая модель (если мерить по F1 score) оказалось GradientBoostingClassifier.
# Стоит отметить, что работоспособность модели, а также их сильные и слабые стороны (у какой выше recall, у какой выше precision) очень сильно зависит от гиперпараметров.
# В рамках тестового задания был производился лишь совсем небольшой подбор гиперпараметров, полностью в ручную. Также, не производилась кросс-валидация или тесты на разных сидах, по сколько всё делалось за один день.
# При наличии большего времени и необходимости, логично было бы применить Optuna TPESampler для автоматического подбора всех гиперпараметров
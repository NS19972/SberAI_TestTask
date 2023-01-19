import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from lightgbm import *
from sklearn.metrics import *

from models import *

if __name__ == '__main__':
    data = pd.read_csv('test.csv', index_col=0)

    #Выбираем модель из [LGB, GradientBoost, RandomForest, NeuralNetwork]
    classifier = LGB(data)

    classifier.fit()
    classifier.validate(show_confusion_matrix=True)
    classifier.test(show_confusion_matrix=True)

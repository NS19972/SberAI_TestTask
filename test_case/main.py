import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from lightgbm import *
from sklearn.metrics import *

from models import *

data = pd.read_csv('test.csv', index_col=0)

classifier = LGBM_model(data, use_class_weights=True)

classifier.fit()
classifier.validate()
classifier.test()

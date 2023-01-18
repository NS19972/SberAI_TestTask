import numpy as np
import pandas as pd
from sklearn.preprocessing import *
from constants import *

def onehot_dataset(data):
    encoder = OneHotEncoder(sparse_output=False)
    onehot_data = encoder.fit_transform(data.loc[:, categorical_columns].values)
    onehot_data = pd.DataFrame(onehot_data)
    data.drop(categorical_columns, axis=1, inplace=True)
    new_data = pd.concat((data, onehot_data), axis=1)

    return encoder, new_data

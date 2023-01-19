import numpy as np
import pandas as pd
from sklearn.preprocessing import *
from constants import *

def onehot_dataset(data):
    encoder = OneHotEncoder(sparse_output=False)
    onehot_data = encoder.fit_transform(data)
    onehot_data = pd.DataFrame(onehot_data)

    return encoder, onehot_data

def scale_dataset(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return scaler, scaled_data
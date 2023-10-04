import pandas as pd

from utils import *
from mlpnas import MLPNAS
from CONSTANTS import TOP_N

import keras


def init_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return x_train, y_train


data = pd.read_csv('DATASETS/wine-quality.csv')
x = data.drop('quality_label', axis=1, inplace=False).values
y = pd.get_dummies(data['quality_label']).values

# x, y = init_data()

nas_object = MLPNAS(x, y)
data = nas_object.search()

get_top_n_architectures(TOP_N)

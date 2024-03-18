# Neural network for binary classification by using Keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *


import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
# X,y = load_data()
print(f'The shape of X is: {X.shape}')
print(f'The shape of y is: {y.shape}')
# visualize the data
# ---



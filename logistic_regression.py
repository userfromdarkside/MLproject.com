import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

x_train = np.array([
    [34.6, 78.0],
    [30.3, 43.9],
    [35.8, 72.9],
    [60.1, 86.3],
    [79.0, 75.3]
])
y_train = np.array([0,0,0,1,1])
print(x_train.shape)
print(y_train.shape)

def sigmoid_function(z):
    g=1/(1+np.exp(-z))
    return g
def compute_cost(x,y,w,b,*argv):
    m,n = x.shape
    loss_sum = 0
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb+=x[i,j]*w[j]
        z_wb+=b
        f_wb=sigmoid_function(z_wb)
        loss_sum+=-(y[i]*np.log(f_wb))-(1-y[i])*np.log(1-f_wb)
        total_cost = loss_sum / m
        return total_cost
def compute_gradient(x,y,w,b,*argv):
    m,n = x.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb+= x[i,j]*w[j]
        z_wb +=b
        f_wb = sigmoid_function(z_wb)
        
        dj_db_i = f_wb - y[i]
        dj_db+=dj_db_i
        for j in range(n):
            dj_dw[j] += (f_wb-y[i])*(x[i,j])
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db

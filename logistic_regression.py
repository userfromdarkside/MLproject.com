<<<<<<< HEAD
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
def compute_cost(x, y, w, b, *argv):
    m, n = x.shape
    loss_sum = 0

    for i in range(m):
        z_wb = np.dot(x[i], w) + b
        f_wb = sigmoid_function(z_wb)

        # Add a small epsilon to prevent division by zero in log
        epsilon = 1e-15
        loss_sum += -y[i] * np.log(f_wb + epsilon) - (1 - y[i]) * np.log(1 - f_wb + epsilon)

    total_cost = loss_sum / m
    return total_cost
def compute_gradient(X, y, w, b, *argv): 
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        z_wb = 0
        for j in range(n): 
            z_wb += X[i][j]*w[j]
        z_wb += b
        f_wb = sigmoid_function(z_wb)
        
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        
        for j in range(n):
            dj_dw_ij = (f_wb - y[i]) * X[i][j]
            dj_dw[j] += dj_dw_ij
            
    dj_dw = (1/m)*dj_dw
    dj_db = (1/m)*dj_db     
    return dj_db, dj_dw
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    m = len(X)
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)
import matplotlib.pyplot as plt

# Plotting the training data
plt.scatter(x_train[y_train == 0][:, 0], x_train[y_train == 0][:, 1], marker='o', label='Not pass')
plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], marker='*', label='Pass the course')

# Plotting the decision boundary
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = sigmoid_function(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')  # Decision boundary

plt.title('Decision Boundary and Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
=======
>>>>>>> parent of c6fc640 (update for fun)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Functions
# 1.Cost Calculation Function
def compute_cost(X, y, theta):
    # inner product
    inner = np.power((X*theta.T-y), 2)
    inner_value = np.sum(inner)/(2*len(X))
    return inner_value


# 2.Batch Gradient Decent
def batch_gradient_decent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    # number of parameters that should be optimized in one iteration
    parameters = theta.shape[1]
    cost = np.zeros(iters)
    for i in range(iters):
        error = X*theta.T-y
        for j in range(parameters):
            term = error.T * X[:, j]
            temp[0, j] = theta[0, j] - (alpha/len(X))*np.sum(term)
        theta = temp
        cost[i] = compute_cost(X, y, theta)
    return theta, cost


# 3.Linear Plotting Function
def linear_plotting(x, f, data):
    plt.figure()
    figure1 = plt.plot(x, f, color='r', label='Prediction')
    figure2 = plt.scatter(data['Population'], data['Profit'], label='Training Data')


# 4.Normal Equation
def normal_eqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y # X.T@X等价于X.T.dot(X)
    return theta

# EXERCISE 1-1 Single-Variable Linear Regression

path1 = "ex1data1.txt"
data = pd.read_csv(path1, header=None, names=['Population', 'Profit'])
# data.plot(kind='scatter', x='Population', y='Profit')
# plt.show()
data.insert(0, 'Ones', 1)
print(data.head())

# set training value:X and target value:y
col = data.shape[1]
X = data.iloc[:, 0:col-1]
y = data.iloc[:, col-1:col]

# transform pd matrices into np matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))
cost_ini = compute_cost(X, y, theta)
print(cost_ini)

# parameters
alpha = 0.01
iters = 2000
Theta, Cost = batch_gradient_decent(X, y, theta, alpha, iters)
cost_final = compute_cost(X, y, Theta)
print(cost_final)

# Plotting
# 1.Linear Model
x = np.linspace(data['Population'].min(), data['Population'].max())
f = Theta[0, 0] + (Theta[0, 1] * x)
linear_plotting(x, f, data)

# 2.Loss Function
plt.figure()
x = np.arange(iters)
plt.plot(x, Cost, 'r')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Error vs. Training Epoch')


# EXERCISE 1-2 Multi-Variable Linear Regression

path2 = "ex1data2.txt"
data2 = pd.read_csv(path2, header=None, names=['Size', 'Bedrooms', 'Price'])
# Feature Normalization
data2 = (data2-data2.mean())/data2.std()
data2.insert(0, 'Ones', 1)
print(data2.head())
col2 = data2.shape[1]
X2 = data2.iloc[:, 0:col2-1]
y2 = data2.iloc[:, col2-1:col2]
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))
Theta2, Cost2 = batch_gradient_decent(X2, y2, theta2, alpha, iters)
x2 = np.arange(iters)
# Plotting
plt.figure()
plt.plot(x2, Cost2, 'r')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Error vs. Training Epoch')
plt.show()

# Normal Equation
final_theta2 = normal_eqn(X, y)
print(final_theta2)

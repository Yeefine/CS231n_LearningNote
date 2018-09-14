import numpy as np
import random

def gradientDescent(x, y, theta, alpha, m, numIterations): # numIterations 迭代次数
    xTrans = x.transpose()  # 转置矩阵
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d / Cost: %f" % (i, cost))
        gradient = np.dot(xTrans, loss) / m  # ??
        theta = theta  - alpha * gradient
    return theta

def genData(numPoints, bias, variance):  # 生成数据
    x = np.zeros(shape = (numPoints, 2))
    y = np.zeros(shape = numPoints)

    for i in range(0, numPoints):
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

x, y = genData(100, 25, 10)
print("x: ")
print(x)
print("y: ")
print(y)

m, n = np.shape(x)
n_y = np.shape(y)
print("x shape: ", str(m), "  ", str(n))
print("y length: ", str(n_y))

numIterations = 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)
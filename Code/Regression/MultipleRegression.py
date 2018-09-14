from  numpy import genfromtxt  # 转换文本
import numpy as np
from sklearn import linear_model, datasets

dataPath = r"E:/Python/Regression/Delivery.csv"
deliveryData = genfromtxt(dataPath, delimiter = ',')

print(deliveryData)

X = deliveryData[:, :-1]
Y = deliveryData[:, -1]

print("X: ")
print(X)
print("Y: ")
print(Y)

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print("coeficients: ")
print(regr.coef_)  # 系数
print("intercept: ")
print(regr.intercept_) # 截距

X_pred = [102, 6]
Y_pred = regr.predict([X_pred]) # predict的参数要求数组
print("Y_pred: ")
print(Y_pred)
 
import numpy as np
from astropy.units import Ybarn
import math

def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = math.sqrt(varX * varY)
    return SSR / SST

def polyfit(X, Y, digree): # x的最高次
    result = {}
    coeffs = np.polyfit(X, Y, digree) #求得系数 b0, b1

    result['polynomial'] = coeffs.tolist()


    p = np.poly1d(coeffs) # p为带入了具体系数coeffs之后的方程
    yhat = p(X)
    ybar = np.mean(Y)
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((Y - ybar)**2)
    result['determination'] = ssreg / sstot

    return result

testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]
print("r: ",computeCorrelation(testX, testY))
print("r^2: ", str(computeCorrelation(testX,testY)**2))
print(polyfit(testX, testY, 1)['determination'])
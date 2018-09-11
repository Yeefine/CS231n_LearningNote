print(__doc__)

import numpy as np
import pylab as pl  #画图
from sklearn import svm

np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]  # np._r 按列连接矩阵   np.random.randn(a,b)生成a行b列的矩阵
Y = [0] * 20 + [1] * 20

# print(X)

clf = svm.SVC(kernel = 'linear')
clf.fit(X, Y)

w = clf.coef_[0] # logistics regression 中的模型参数, w_0 和 w_1 的集合
a = -w[0] / w[1]
xx = np.linspace(-5, 5) # linspace() 通过开始值、终值和元素个数创建表示等差数列的一维数组
yy = a * xx -(clf.intercept_[0]) / w[1] # clf.intercept 即 w_3

# w_0 x + w_1 y + w_3 = 0 can be rewritten y = - (w_0 / w_1) x - (w_3 / w_1)

b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

print("w: ", w)
print("a: ", a)
print("support_vectors_: ", clf.support_vectors_)
print("clf.coef_: ", clf.coef_)

pl.plot(xx, yy, 'k-')  # k代表黑色， -代表实线
pl.plot(xx, yy_down, 'k--') # --代表虚线
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 80, facecolor = 'none') # [:,0]取所有行的第0个元素
pl.scatter(X[:, 0], X[:, 1], c = Y, cmap = pl.cm.Paired)

pl.axis('tight')
pl.show()

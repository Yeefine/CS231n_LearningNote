import csv
import numpy as np
from numpy.linalg import inv # 包含线性代数的函数，inv计算逆矩阵
import random
import math
import sys

data = []
# 每个维度储存一种污染物的资讯
for i in range(18):
    data.append([])

n_row = 0
text = open(r'E:\Python\predPM2.5\train.csv', encoding = 'big5')
row = csv.reader(text, delimiter = ",")

for r in row:
    # 第0行没有资讯
    if n_row != 0:
        # 每一行只有第3-27格有值(1天内24小时的数值)
        for i in range(3, 27):
            if r[i] != "NR":
                data[(n_row - 1) % 18].append(float(r[i]))
            else:
                data[(n_row - 1) % 18].append(float(0))
    n_row += 1
text.close()
print(np.shape(data))

x = []
y = []
# 每12个月
for i in range(12):
    # 一个月取连续10小时的data可以有471笔
    for j in range(471):
        x.append([])
        # 18种污染物
        for t in range(18):
            # 连续9小时
            for s in range(9):
                x[471 * i + j].append(data[t][480 * i + j + s]) # x[0]表示1月第1天18种污染物每种1-9点的数据；x[1]表示2-10点，以此类推
        y.append(data[9][480 * i + j + 9]) # y[9]表示1月第1天第9种污染物10点的数据
x = np.array(x) # 5652 * 162
y = np.array(y) # 5652 * 1
# print(y)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0])) # 162 * 1
l_rate = 10
repeat = 10000

# use close form to check whether ur gradient descent is good
# however, this cannot be used in hw1.sh
# w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)

x_t = x.transpose() # 162 * 5652
s_gra = np.zeros(len(x[0])) # 162 * 1

for i in range(repeat):
    # Adagrad
    hypo = np.dot(x, w) # 5652 * 1
    loss = hypo - y #5652 * 1
    cost = np.sum(loss**2) / len(x)
    cost_a = math.sqrt(cost) # ?
    gra = np.dot(x_t, loss) # 162 * 1
    s_gra += gra**2 # 之前梯度之和
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra / ada
    print('iteration: %d | Cost: %f   ' % ( i, cost_a))

# save model
np.save('E:\Python\predPM2.5\model.npy', w)
# read model
w = np.load('E:\Python\predPM2.5\model.npy')

# read text sample
test_x = []
n_row = 0
text = open(r'E:\Python\predPM2.5\test.csv')
row = csv.reader(text, delimiter = ",")

for r in row:
    if n_row % 18 == 0:
        test_x.append([])
        for i in range(2, 11):
            test_x[n_row // 18].append(float(r[i]))  # //返回不大于结果的一个最大整数
    else :
        for i in range(2, 11):
            if  r[i] != 'NR':
                test_x[n_row // 18].append(float(r[i]))
            else:
                test_x[n_row // 18].append(0)
    n_row += 1
    # test_x[0]表示第1天18种污染物1-9小时的数据

# print(np.shape(test_x))
text.close()
test_x = np.array(test_x) # test_x  240 * 162

# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis = 1)  # 数组拼接，axis = 1表示对应行的数组进行拼接
# print(np.shape(test_x))
# test_x  240 * 163


ans = []
print(np.shape(w), "    ", np.shape(test_x))
for i in range(len(test_x)):
    ans.append(["id_" + str(i)])
    a = np.dot(w, test_x[i])
    ans[i].append(a)


filename = "E:\Python\predPM2.5\predict.csv"
test = open(filename, "w+")
s = csv.writer(test, delimiter = ',', lineterminator = '\n')
s.writerow(["id", "value"])
for i in range(len(ans)):  # 240
    s.writerow(ans[i])
test.close()

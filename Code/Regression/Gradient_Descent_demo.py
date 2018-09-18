import numpy as np
import pylab as pl

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
# y_data = b + w * x_data

x = np.arange(-200, -100, 1) # bias 取[-200,-100]之间差值为1的等差数列
y = np.arange(-5, 5, 0.1) # weight
Z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y) # 产生一个以向量x为行，y行x的矩阵X，Y同理
# print("X: ")
# print(X)
# print("Y: ")
# print(Y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n])**2 # loss function
        Z[j][i] = Z[j][i] / len(x_data)

# y_data = b + w * x_data
b = -120 # initial b
w = -4 # initial w
lr = 1 # learning rate
iteration = 100000

# Store initial values for plotting
b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0

# Iterations
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

    lr_b = lr_b + b_grad**2
    lr_w = lr_w + w_grad**2

    # Update parameters
    b = b - lr/np.sqrt(lr_b) * b_grad
    w = w - lr/np.sqrt(lr_w) * w_grad


    # Store parameters for plotting
    b_history.append(b)
    w_history.append(w)

# plot the figure
pl.contourf(x, y, Z, 50, alpha = 0.5, cmap = pl.get_cmap('jet'))
pl.plot([-188.4], [2.67], 'x', ms = 12, markeredgewidth = 3, color = 'orange')
pl.plot(b_history, w_history, 'o-', ms = 3, lw = 1.5, color = 'black')
pl.xlim(-200, -100)
pl.ylim(-5, 5)
pl.xlabel(r'$b$', fontsize = 16)
pl.ylabel(r'$w$', fontsize = 16)
pl.show()
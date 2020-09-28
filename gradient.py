from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import numpy as np


def make_noisy_data(m=0.1, b=0.3, n=100):
    x = np.random.uniform(size=n)
    noise = np.random.normal(size=len(x), scale=0.01)
    y = m*x + b + noise
    return x, y

x_train, y_train = make_noisy_data()

#instantiate m and b
m = 0
b = 0
c_func = list()
m_arr = list()
#simple linear model
def predict(x):
    y = m*x + b
    return y


#error 
def squared_error(y_pred, y_true):
    diff = y_pred - y_true
    sq = np.square(diff)
    error = np.mean(sq)
    return error

loss = squared_error(predict(x_train), y_train)






learning_rate = 0.05
steps = 300
n = 100
def gradient(m, b, predictions):
    diff = predictions - y_train
    tempb = tempm = 0
    for i in range(n):
        tempm += diff[i] * x_train[i]
    tempb += diff
    
    tempm = np.mean(tempm)
    print(tempm)
    tempb = np.mean(tempb)
    
    return tempm, tempb
    
    
    

#actual gradient descending will be happening here
for i in range(steps):
    predictions = predict(x_train)
    loss = squared_error(predictions, y_train)

    gr_d = gradient(m , b, predictions)

    m = m - learning_rate * gr_d[0]
    b = b - learning_rate * gr_d[1]
    c_func.append(loss)
    m_arr.append(m)

    if i % 20 == 0:
        print('step= %d, loss= %f'%(i,  loss))

plt.subplot(121)
plt.plot(x_train, y_train, 'ro')
plt.plot(x_train, predict(x_train))

plt.subplot(122)
plt.plot(m_arr, c_func,  'k')
plt.show()


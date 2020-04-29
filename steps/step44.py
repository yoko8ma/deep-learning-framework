import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
l1 = L.Linear(10)
l2 = L.Linear(1)

# 重みの初期化
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))


def predict(x):
    y = l1(x)
    y = F.sigmoid_simple(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000


for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data = lr * p.grad.data
    if i % 1000 == 0:
        predict(loss)


# Plot
import matplotlib.pyplot as plt
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()
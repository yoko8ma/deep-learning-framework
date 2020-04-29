import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L


lr = 0.2
max_iter = 10000
hidden_size = 10

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid_simple(self.l1(x))
        y = self.l2(y)
        return y

model = TwoLayerNet(hidden_size, 1)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000:
        print(loss)

# Plot
# import matplotlib.pyplot as plt
# plt.scatter(x, y, s=10)
# plt.xlabel('x')
# plt.ylabel('y')
# t = np.arange(0, 1, .01)[:, np.newaxis]
# y_pred = predict(t)
# plt.plot(t, y_pred.data, color='r')
# plt.show()
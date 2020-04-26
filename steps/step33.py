import numpy as np
from dezero import Variable, Function


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleangrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleangrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
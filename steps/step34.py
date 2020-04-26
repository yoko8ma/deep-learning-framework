import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functons as F


x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data.flatten()]

for i in range(3):
    logs.append(x.grad.data.flattern())
    gx = x.grad
    x.cleangrad()
    gx.backward(create_graph=True)


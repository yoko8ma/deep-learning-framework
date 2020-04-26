import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functons as F
from dezero.utils import plot_dot_graph


x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 6

for i in range(iters):
    gx = x.grad
    x.cleangrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx'
plot_dot_graph(gx, verbose=False, to_file='tanh.png')
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functons as F
from dezero.utils import plot_dot_graph


x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad
x.cleangrad()

z = gx ** 3 + y
z.backward()
print(x.grad)
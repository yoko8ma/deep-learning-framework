import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph


# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.reshape(x, (6,))
# y.backward(retain_grad=True)
# print(x.grad)

x = Variable(np.random.randn(1, 2, 3))
y = x.reshape((2, 3))
y = x.reshape(2, 3)


x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.transpose(x)
y.backward()
print(x.grad)

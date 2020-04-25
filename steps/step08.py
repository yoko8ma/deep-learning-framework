import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None    # 微分した値
        self.creator = None # 生成した関数

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input  # 入力されたされた変数を覚える
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 逆向きに計算グラフのノードを辿る
assert y.creator == C
assert y.creator.input == b

y.grad = np.array(1.0)
C = y.creator
b = C.input
b.grad = C.backward(y.grad)
B = b.creator
a = B.input
a.grad = B.backward(b.grad)
A = a.creator
x = A.input
x.grad = A.backward(a.grad)
print(x.grad)

# 逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)
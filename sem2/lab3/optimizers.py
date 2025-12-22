import numpy as np
from base import Optimizer

class SGD(Optimizer):
    def __init__(self, module, lr = 1e-2, momentum = 0.0,
                 weight_decay = 0.0):
        super().__init__(module)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]

        for param, grad, m in zip(parameters, gradients, self.state['m']):
            grad = grad + self.weight_decay * param
            np.multiply(self.momentum, m, out=m)
            np.add(m, grad, out=m)
            np.add(param, -self.lr * m, out=param)

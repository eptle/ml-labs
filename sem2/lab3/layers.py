import numpy as np
from base import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input):
        if self.bias is not None:
            return input @ self.weight.T + self.bias
        else:
            return input @ self.weight.T

    def compute_grad_input(self, input, grad_output):
        return grad_output @ self.weight

    def update_grad_parameters(self, input, grad_output):
        self.grad_weight += grad_output.T @ input
        if self.bias is not None:
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self):
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self):
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input):
        for module in self.modules:
            input = module(input)
        return input

    def compute_grad_input(self, input, grad_output):
        inputs = [input]
        for module in self.modules:
            input = module(input)
            inputs.append(input)
        for i in range(len(self.modules) - 1, -1, -1):
            grad_output = self.modules[i].backward(inputs[i], grad_output)
        return grad_output

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self):
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self):
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self):
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str

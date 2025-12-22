import numpy as np
from base import Module

class Sigmoid(Module):
    def compute_output(self, input):
        return 1 / (1 + np.exp(-input))

    def compute_grad_input(self, input, grad_output):
        return grad_output * self.compute_output(input) * (1 - self.compute_output(input))

class Softmax(Module):
    def compute_output(self, input):
        return 1 / (1 + np.exp(-input))

    def compute_grad_input(self, input, grad_output):
        return grad_output * input * (1 - input)

class LogSoftmax(Module):
    def compute_output(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        shifted = input - input_max
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        return shifted - log_sum_exp


    def compute_grad_input(self, input, grad_output):
        log_softmax = self.compute_output(input)
        softmax = np.exp(log_softmax)
        grad = grad_output - softmax * np.sum(grad_output, axis=1, keepdims=True)
        return grad

from abc import ABC, abstractmethod

class Module(ABC):
    def __init__(self):
        self.output = None
        self.training = True

    @abstractmethod
    def compute_output(self, input):
        raise NotImplementedError

    @abstractmethod
    def compute_grad_input(self, input, grad_output):
        raise NotImplementedError

    def update_grad_parameters(self, input, grad_output):
        pass

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        self.output = self.compute_output(input)
        return self.output

    def backward(self, input, grad_output):
        grad_input = self.compute_grad_input(input, grad_output)
        self.update_grad_parameters(input, grad_output)
        return grad_input

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def zero_grad(self):
        pass

    def parameters(self):
        return []

    def parameters_grad(self):
        return []

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class Criterion(ABC):
    def __init__(self):
        self.output = None

    @abstractmethod
    def compute_output(self, input, target):
        raise NotImplementedError

    @abstractmethod
    def compute_grad_input(self, input, target):
        raise NotImplementedError

    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        self.output = self.compute_output(input, target)
        return self.output

    def backward(self, input, target):
        grad_input = self.compute_grad_input(input, target)
        return grad_input

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class Optimizer(ABC):
    def __init__(self, module):
        self.module = module
        self.state = {}

    def zero_grad(self):
        self.module.zero_grad()

    @abstractmethod
    def step(self):
        raise NotImplementedError

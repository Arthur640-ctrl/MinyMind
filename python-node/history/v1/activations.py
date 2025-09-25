import math
import numpy as np

class Linear:
    @staticmethod
    def f(x):
        return x
    
    @staticmethod
    def d(x):
        return 1

class Sigmoid:
    @staticmethod
    def f(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def d(x):
        s = Sigmoid.f(x)
        return s * (1 - s)

class Tanh:
    @staticmethod
    def f(x):
        return np.tanh(x)
    
    @staticmethod
    def d(x):
        return 1 - np.tanh(x)**2

class ReLU:
    @staticmethod
    def f(x):
        return np.maximum(0, x)
    
    @staticmethod
    def d(x):
        return np.where(x > 0, 1, 0)

class LeakyReLU:
    @staticmethod
    def f(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def d(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

class ELU:
    @staticmethod
    def f(x, alpha=1.0):
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def d(x, alpha=1.0):
        return np.where(x >= 0, 1, alpha * np.exp(x))

class Softmax:
    @staticmethod
    def f(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    @staticmethod
    def d(x):
        s = Softmax.f(x).reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)


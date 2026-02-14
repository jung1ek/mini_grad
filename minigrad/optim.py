from minigrad import nn
from minigrad.tensor import Tensor
import numpy as np

class SGD(nn.Module):
    
    def __init__(self,parameters: nn.Parameter,rate=0.001):
        self.parameters = parameters
        
    def step(self):
        for param in self.parameters:
            update = param.value.data - (self.rate * param.value.grad.data)
            param.update(Tensor(update))
            
class SGDMomentum(nn.Module):
    
    def __init__(self,parameters: nn.Parameter, rate=0.001):
        self.parameters = parameters
        self.v = [Tensor.zeros_like(param.value) for param in parameters]
        
    def step(self):
        for i,param in enumerate(self.parameters):
            vt = self.v[i].data + self.rate * param.value.grad.data
            self.v[i] = Tensor(vt)
            update = param.value.data - vt
            param.update(Tensor(update))

class RMSprop(nn.Module):
    
    def __init__(self,parameters: nn.Parameter, rate=0.001,beta=0.9):
        self.parameters = parameters
        self.g = [Tensor.zeros_like(param.value) for param in parameters]
        self.beta = beta
        self.rate = rate
        self.eps = 1e-5
    
    def step(self):
        for i,param in enumerate(self.parameters):
            grad = param.value.grad.data
            gt = self.beta*self.g[i].data + (1-self.beta) * grad ** 2
            self.g[i] = Tensor(gt)
            update = param.value.data - (self.rate * grad /(np.sqrt(gt) + self.eps))
            param.update(Tensor(update))     
     
class Adam(nn.Module):
    
    def __init__(self,parameters: nn.Parameter,rate=0.001,b1=0.9,b2=0.99):
        self.parameters = parameters
        self.b1 = b1
        self.b2 = b2
        self.r = rate
        self.m = [Tensor.zeros_like(param.value) for param in parameters]
        self.v = [Tensor.zeros_like(param.value) for param in parameters]
        self.t = 0
    
    def step(self):
        for i,param in enumerate(self.parameters):
            self.t += 1
            grad = param.value.gard.data
            mt = self.m[i].data * self.b1 + (1-self.b1) * grad  
            vt = self.v[i].data * self.b2 + (1-self.b2) * grad ** 2
            self.m[i] = Tensor(mt)
            self.v[i] = Tensor(vt)
            mh = mt / (1-self.b1**self.t)
            vh = vt / (1-self.b2**self.t)
            update = param.value.data - (self.r * mh / (np.sqrt(vh)+1e-5))
            param.update(Tensor(update))
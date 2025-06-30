import numpy as np
from minigrad.ops import GenericExecAST

class CPUBuffer(np.ndarray,GenericExecAST):
    @staticmethod
    def fromCPU(x) : return x.view(CPUBuffer)
    def toCPU(x): return x

    def load_op(x): return CPUBuffer.fromCPU(x)
    def unary_op(self,op): return self
    def binary_op(x,op,y): return x + y
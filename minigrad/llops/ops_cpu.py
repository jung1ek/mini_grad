from __future__  import annotations
import numpy as np
from minigrad.ops import GenericExecAST
from minigrad.ops import BinaryOps,UnaryOps,MovementOps,ReduceOps,ProcessingOps
import operator
from minigrad.helpers import shape_to_axis

class CPUBuffer(np.ndarray,GenericExecAST):

    fxn_for_op = {
        BinaryOps.MUL: operator.mul, BinaryOps.ADD: operator.add,
        MovementOps.EXPAND: lambda x,shape: CPUBuffer.expand(x,shape), MovementOps.RESHAPE: lambda x, shape: CPUBuffer.reshape(x,shape),
        ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape,new_shape),keepdims=True), ReduceOps.MAX: None,
    }
    def expand(x,shape) : return np.broadcast_to(x,shape=shape).view(CPUBuffer)
    def reshape(x,shape) : return x.reshape(shape).view(CPUBuffer)
    def permute(x,order) : return None
    def transpose(x,order) : return None

    @staticmethod
    def fromCPU(x: np.ndarray) : return x.view(CPUBuffer)
    def toCPU(x: np.ndarray) -> np.ndarray: return x

    def load_op(x): return CPUBuffer.fromCPU(x)
    def unary_op(x,op): return CPUBuffer.fxn_for_op[op](x)
    def binary_op(x,op,y): return CPUBuffer.fxn_for_op[op](x,y)
    def movement_op(x,op,shape): return CPUBuffer.fxn_for_op[op](x,shape)
    def reduce_op(x,op,new_shape): return CPUBuffer.fxn_for_op[op](x,new_shape)
    def processing_op(x,op,c): return None

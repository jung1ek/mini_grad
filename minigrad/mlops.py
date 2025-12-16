from __future__ import annotations
from minigrad.tensor import Function
from minigrad.lazy import MovementOps, LazyBuffer,BinaryOps,ReduceOps
from minigrad.helpers import reduce_shape, shape_to_axis
# TODO making forward and backward for every math function

class Mul(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer)-> LazyBuffer:
        return x.binary_op(BinaryOps.MUL,y)
    def backward(self, output_grad: LazyBuffer)-> LazyBuffer:
        return 0

class Add(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer)-> LazyBuffer: 
        return x.binary_op(BinaryOps.ADD,y)
    def backward(self,output_grad: LazyBuffer):
        return 0

# Movement ops
class Reshape(Function):
    # x is LazyBuffer.
    def forward(self, x: LazyBuffer, shape: tuple):
        self.input_shape = x.shape # original shape
        return x.movement_op(MovementOps.RESHAPE, shape)
    
    def backward(self, output_grad: LazyBuffer):
        return output_grad.movement_op(MovementOps.RESHAPE, self.input_shape)

class Expand(Function):

    def forward(self, x: LazyBuffer, shape: tuple):
        self.input_shape = x.shape
        return x.movement_op(MovementOps.EXPAND, shape)
    
    def backward(self, output_grad: LazyBuffer)-> LazyBuffer:
        return output_grad.reduce_op(ReduceOps.SUM, self.input_shape)

# Reduce ops
class Sum(Function):

    def forward(self, x: LazyBuffer, axis=None, keepdims=False):
        self.input_shape = x.shape
        return x.reduce_op(ReduceOps.SUM,axis,keepdims)

    def backward(self, output_grad: LazyBuffer):
        return output_grad.movement_op(MovementOps.EXPAND,self.input_shape)
    
class Permute(Function):

    def forward(self, x: LazyBuffer, order: tuple):
        return x.movement_op(MovementOps.PERMUTE, order)
    
    def backward(self,output_grad: LazyBuffer):
        pass

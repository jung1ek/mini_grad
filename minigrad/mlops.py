from __future__ import annotations
from minigrad.tensor import Function
from minigrad.lazy import MovementOps, LazyBuffer,BinaryOps
# TODO making forward and backward for every math function

class Mul(Function):
    def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
        return x.binary_op(BinaryOps.MUL,y)
    def backward(self, output_grad:LazyBuffer) -> LazyBuffer:
        return 0

class Add(Function):
    def forward(self,x,y): 
        return x.binary_op(BinaryOps.ADD,y)
    def backward(self,output_grad):
        return 0

class Reshape(Function):
    # x is LazyBuffer.
    def forward(self, x, shape):
        self.input_shape = x.shape
        return x.movement_op(MovementOps.RESHAPE,shape)
    
    def backward(self, output_grad):
        return output_grad.movement_op(MovementOps.RESHAPE,self.input_shape)
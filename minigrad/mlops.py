from __future__ import annotations
from minigrad.tensor import Function
from minigrad.lazy import MovementOps, LazyBuffer,BinaryOps,ReduceOps
from minigrad.helpers import reduce_shape
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

# Movement ops
class Reshape(Function):
    # x is LazyBuffer.
    def forward(self, x, shape):
        self.input_shape = x.shape # original shape
        return x.movement_op(MovementOps.RESHAPE,shape)
    
    def backward(self, output_grad):
        return output_grad.movement_op(MovementOps.RESHAPE,self.input_shape)

class Expand(Function):

    def forward(self,x,shape):
        return x.movement_op(MovementOps.Expand, shape)
    
    def backward(self, output_grad):
        return output_grad

# Reduce ops
class Sum(Function):

    def forward(self,x,axis=None,keepdims=False):
        self.input_shape = x.shape
        return x.reduce_op(ReduceOps.SUM,reduce_shape(x.shape,axis))

    def backward(self):
        pass
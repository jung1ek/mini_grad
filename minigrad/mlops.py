from __future__ import annotations
from minigrad.tensor import Function
from minigrad.lazy import MovementOps, LazyBuffer,BinaryOps,ReduceOps,ProcessingOps
from minigrad.helpers import reduce_shape, shape_to_axis, ConvArgs, get_conv_args
# TODO making forward and backward for every math function
# TODO Relu, Log,Exp,Reciprocal,Sum, Max,Sub,Pow
# TODO slice and flip with 3 arguments stride
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

class Slice(Function):
    def forward(self, x : LazyBuffer,arg=None):

        # the range in the padded tensor that corresponds to the original tensor.
        self.narg = tuple((0-p[0],x.shape[i]-p[0]) for i,p in enumerate(arg))
        return x.slice(arg=tuple(arg))
    
    def backward(self, output_grad):
        return None
    
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

class Conv2D(Function):

    def forward(self,x: LazyBuffer,w: LazyBuffer,stride=1, groups=1, dilation=1, padding=0):
        self.conv_args = get_conv_args(x.shape, w.shape, stride=stride, groups=groups, dilation=dilation, padding=padding)
        return x.processing_op(ProcessingOps.CONV,w,self.conv_args)

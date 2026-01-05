from __future__ import annotations
from minigrad.tensor import Function
from minigrad.lazy import MovementOps, LazyBuffer,BinaryOps,ReduceOps,ProcessingOps, UnaryOps
from minigrad.helpers import reduce_shape, shape_to_axis, ConvArgs, get_conv_args
# TODO making forward and backward for every math function
# TODO Relu, Log,Exp,Reciprocal,Sum, Max,Sub,Pow
# TODO slice and flip with 3 arguments stride
class Mul(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer)-> LazyBuffer:
        self.save_for_backward(x,y)
        return x.binary_op(BinaryOps.MUL,y)
    def backward(self, output_grad: LazyBuffer)-> LazyBuffer:
        x,y = self.saved_tensors
        return output_grad.binary_op(BinaryOps.MUL,y) if self.need_input_grad[0] else None,\
            output_grad.binary_op(BinaryOps.MUL,x) if self.need_input_grad[1] else None

class Add(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer)-> LazyBuffer:
        return x.binary_op(BinaryOps.ADD,y)
    def backward(self,output_grad: LazyBuffer):
        return output_grad,output_grad if self.need_input_grad[0] else None
    
class Sub(Function):
    def forward(self,x,y):
        return x.binary_op(BinaryOps.SUB,y)
    def backward(self, output_grad):
        return output_grad, output_grad.unary_op(UnaryOps.NEG) if self.need_input_grad[0] else None

class Pow(Function):
    def forward(self,x,y):
        ret = x.binary_op(BinaryOps.POW,y)
        self.save_for_backward(x,y,ret)
        return ret 
    def backward(self, output_grad: LazyBuffer):
        x,y,powxy = self.saved_tensors
        # y*x**y-1; y*(x**y)/x
        # log(x)*pow(x,y)
        return output_grad.binary_op(BinaryOps.MUL,y.binary_op(BinaryOps.MUL,x.binary_op(BinaryOps.POW,y).binary_op(BinaryOps.DIV,x))) if self.need_input_grad[0] else None,\
            output_grad.binary_op(BinaryOps.MUL,x.unary_op(UnaryOps.LOG).binary_op(BinaryOps.MUL,powxy)) if self.need_input_grad[1] else None

class Reciprocal(Function):
    def forward(self,x):
        ret = x.unary_op(UnaryOps.RECIPROCAL)
        self.save_for_backward(ret)
        return ret
    
    def backward(self,output_grad: LazyBuffer):
        # x**-1; 1/x * 1/x
        # out_grad*(-1)*(1/y)*(1/y)
        return output_grad.unary_op(UnaryOps.NEG).binary_op(BinaryOps.MUL,self.saved_tensors[0]).binary_op(BinaryOps.MUL,self.saved_tensors[0])

class Relu(Function):
    def forward(self,x):
        ret = x.unary_op(UnaryOps.RELU)
        self.save_for_backward(ret)
        return ret

    def backward(self,output_grad):
        # if x > 0 =1; else 0
        return output_grad.binary_op(BinaryOps.MUL, self.saved_tensors[0].unary_op(UnaryOps.SIGN))

class Log(Function):
    def forward(self,x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOps.LOG)

    def backward(self,output_grad):
        return output_grad.binary_op(BinaryOps.DIV,self.saved_tensors[0])

class Exp(Function):
    def forward(self,x):
        ret = x.unary_op(UnaryOps.EXP)
        self.save_for_backward(ret)
        return ret

    def backward(self,output_grad):
        return output_grad.binary_op(BinaryOps.MUL,self.saved_tensors[0])

class Slice(Function):
    def forward(self, x : LazyBuffer,arg=None):

        # the range in the padded tensor that corresponds to the original tensor.
        self.narg = tuple((0-p[0],x.shape[i]-p[0]) for i,p in enumerate(arg))
        return x.slice(arg=tuple(arg))
    
    def backward(self, output_grad: LazyBuffer):
        return output_grad.slice(arg=self.narg)
    
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
        return output_grad.reduce_op(ReduceOps.SUM, shape_to_axis(self.input_shape,output_grad.shape),False)

# Reduce ops
class Sum(Function):

    def forward(self, x: LazyBuffer, axis=None, keepdim=False):
        self.input_shape = x.shape
        return x.reduce_op(ReduceOps.SUM,axis,keepdim)

    def backward(self, output_grad: LazyBuffer):
        return output_grad.movement_op(MovementOps.EXPAND,self.input_shape)
    
class Permute(Function):

    def forward(self, x: LazyBuffer, order: tuple):
        self.input_order = order
        return x.movement_op(MovementOps.PERMUTE, order)
    
    def backward(self,output_grad: LazyBuffer):
        return output_grad.movement_op(MovementOps.PERMUTE, sorted(range(len(self.input_order)), key=self.input_order.__getitem__))

class Conv2D(Function):

    def forward(self,x: LazyBuffer,w: LazyBuffer,stride=1, groups=1, dilation=1, padding=0):
        self.conv_args = get_conv_args(x.shape, w.shape, stride=stride, groups=groups, dilation=dilation, padding=padding)
        return x.processing_op(ProcessingOps.CONV,w,self.conv_args)

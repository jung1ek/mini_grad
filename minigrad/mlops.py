from __future__ import annotations
from minigrad.tensor import Function
from minigrad.lazy import MovementOps, LazyBuffer,BinaryOps,ReduceOps,ProcessingOps, UnaryOps
from minigrad.helpers import reduce_shape, shape_to_axis, ConvArgs, get_conv_args, _normalize_axis

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
        return output_grad if self.need_input_grad[0] else None\
            ,output_grad if self.need_input_grad[1] else None
    
class Sub(Function):
    def forward(self,x,y):
        return x.binary_op(BinaryOps.SUB,y)
    def backward(self, output_grad):
        return output_grad if self.need_input_grad[0] else None, \
            output_grad.unary_op(UnaryOps.NEG) if self.need_input_grad[1] else None

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
        return output_grad.unary_op(UnaryOps.NEG).binary_op(BinaryOps.MUL,self.saved_tensors[0]).binary_op(BinaryOps.MUL,self.saved_tensors[0]) if self.need_input_grad[0] else None

class Relu(Function):
    def forward(self,x):
        ret = x.unary_op(UnaryOps.RELU)
        self.save_for_backward(ret)
        return ret

    def backward(self,output_grad):
        # if x > 0 =1; else 0
        return output_grad.binary_op(BinaryOps.MUL, self.saved_tensors[0].unary_op(UnaryOps.SIGN)) if self.need_input_grad[0] else None

class Log(Function):
    def forward(self,x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOps.LOG)

    def backward(self,output_grad):
        return output_grad.binary_op(BinaryOps.DIV,self.saved_tensors[0]) if self.need_input_grad[0] else None

class Exp(Function):
    def forward(self,x):
        ret = x.unary_op(UnaryOps.EXP)
        self.save_for_backward(ret)
        return ret

    def backward(self,output_grad):
        return output_grad.binary_op(BinaryOps.MUL,self.saved_tensors[0]) if self.need_input_grad[0] else None
    
class Masked_Fill(Function):

    def forward(self, x,mask,value):
        self.mask = mask
        return x.movement_op(MovementOps.MASKED_FILL,(mask,value))
    
    def backward(self, output_grad: LazyBuffer):
        return output_grad.movement_op(MovementOps.MASKED_FILL,(self.mask,0)) if self.need_input_grad[0] else None

class Slice(Function):
    def forward(self, x : LazyBuffer,arg=None):

        # the range in the padded tensor that corresponds to the original tensor.
        self.narg = tuple((0-p[0],x.shape[i]-p[0]) for i,p in enumerate(arg))
        ret = x.slice(arg=tuple(arg))
        ret.shape = tuple(end - start for start, end in arg)
        return ret
    
    def backward(self, output_grad: LazyBuffer):
        ret = output_grad.slice(arg=self.narg)
        ret.shape = tuple(end - start for start, end in self.narg)
        return ret if self.need_input_grad[0] else None

class Max(Function):
    def forward(self,x,axis=None,keepdim=False):
        ret = x.reduce_op(ReduceOps.MAX,axis,keepdim)
        self.save_for_backward(x,ret)
        return ret
    
    def backward(self, grad_output):
        x, ret = self.saved_tensors
        # 1s in locations where the max was chosen (can be two locations)
        max_is_1s = x.binary_op(BinaryOps.CMPEQ, ret.movement_op(MovementOps.EXPAND, x.shape))
        # # sum of locations, averaged
        div = max_is_1s.reduce_op(ReduceOps.SUM, shape_to_axis(max_is_1s.shape,grad_output.shape),True)
        div = div.movement_op(MovementOps.EXPAND, x.shape)
        max_is_amount = max_is_1s.binary_op(BinaryOps.DIV, div)

        grad_output_expanded = grad_output.movement_op(MovementOps.EXPAND, x.shape)
        return max_is_amount.binary_op(BinaryOps.MUL, grad_output_expanded) if self.need_input_grad[0] else None
    
# Movement ops
class Reshape(Function):
    # x is LazyBuffer.
    def forward(self, x: LazyBuffer, shape: tuple):
        self.input_shape = x.shape # original shape
        return x.movement_op(MovementOps.RESHAPE, shape)
    
    def backward(self, output_grad: LazyBuffer):
        return output_grad.movement_op(MovementOps.RESHAPE, self.input_shape) if self.need_input_grad[0] else None

class Expand(Function):

    def forward(self, x: LazyBuffer, shape: tuple):
        self.input_shape = x.shape
        return x.movement_op(MovementOps.EXPAND, shape)
    
    def backward(self, output_grad: LazyBuffer)-> LazyBuffer:
        return output_grad.reduce_op(ReduceOps.SUM, shape_to_axis(self.input_shape,output_grad.shape),True) if self.need_input_grad[0] else None

# Reduce ops
class Sum(Function):

    def forward(self, x: LazyBuffer, axis=None, keepdim=False):
        self.input_shape = x.shape
        self.input_shape = x.shape
        self.axis = axis
        self.keepdim = keepdim
        return x.reduce_op(ReduceOps.SUM,axis,keepdim)

    def backward(self, output_grad: LazyBuffer):

        if not self.need_input_grad[0]:
            return None

        # normalize the axis,; -1 = last dim
        axis = _normalize_axis(self.axis, len(self.input_shape))

        grad = output_grad

        # 🔥 If keepdim=False, reinsert dimensions
        if not self.keepdim:
            new_shape = list(grad.shape)
            for ax in sorted(axis):
                new_shape.insert(ax, 1)

            grad = grad.movement_op(
                MovementOps.RESHAPE,
                tuple(new_shape)
            )

        # 🔥 Now broadcast safely
        grad = grad.movement_op(
            MovementOps.EXPAND,
            self.input_shape
        )
        return grad
    
class Permute(Function):

    def forward(self, x: LazyBuffer, order: tuple):
        self.input_order = order
        return x.movement_op(MovementOps.PERMUTE, order)
    
    # argsort; sort the [0,1,2] order based on the key which is input_order
    def backward(self,output_grad: LazyBuffer):
        return output_grad.movement_op(MovementOps.PERMUTE, tuple(sorted(range(len(self.input_order)), key=self.input_order.__getitem__))) if self.need_input_grad[0] else None

class Conv2D(Function):

    def forward(self,x: LazyBuffer,w: LazyBuffer,stride=1, groups=1, dilation=1, padding=0):
        self.conv_args = get_conv_args(x.shape, w.shape, stride=stride, groups=groups, dilation=dilation, padding=padding)
        return x.processing_op(ProcessingOps.CONV,w,self.conv_args)

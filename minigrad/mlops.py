from __future__ import annotations
from minigrad.tensor import Function
from minigrad.lazy import MovementOps, LazyBuffer,BinaryOps,ReduceOps,ProcessingOps, UnaryOps
from minigrad.helpers import reduce_shape, shape_to_axis, ConvArgs, get_conv_args, normalize_axis, keepdim_shape_from_reduced

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
        return output_grad.binary_op(BinaryOps.MUL,y.binary_op(BinaryOps.MUL,x.binary_op(BinaryOps.POW,y)\
                                    .binary_op(BinaryOps.DIV,x))) if self.need_input_grad[0] else None,\
            output_grad.binary_op(BinaryOps.MUL,x.unary_op(UnaryOps.LOG)\
                                  .binary_op(BinaryOps.MUL,powxy)) if self.need_input_grad[1] else None

class Reciprocal(Function):
    def forward(self,x):
        ret = x.unary_op(UnaryOps.RECIPROCAL)
        self.save_for_backward(ret)
        return ret
    
    def backward(self,output_grad: LazyBuffer):
        # x**-1; 1/x * 1/x
        # out_grad*(-1)*(1/y)*(1/y)
        return output_grad.unary_op(UnaryOps.NEG).binary_op(BinaryOps.MUL,self.saved_tensors[0])\
            .binary_op(BinaryOps.MUL,self.saved_tensors[0]) if self.need_input_grad[0] else None

class Relu(Function):
    def forward(self,x):
        ret = x.unary_op(UnaryOps.RELU)
        self.save_for_backward(ret)
        return ret

    def backward(self,output_grad):
        # if x > 0 =1; else 0
        return output_grad.binary_op(BinaryOps.MUL, self.saved_tensors[0].unary_op(UnaryOps.SIGN))\
              if self.need_input_grad[0] else None

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
        return output_grad.movement_op(MovementOps.MASKED_FILL,(self.mask,0))\
             if self.need_input_grad[0] else None

class Slice(Function):
    def forward(self, x : LazyBuffer,arg=None):

        # the range in the padded tensor that corresponds to the original tensor.
        self.narg = tuple((0-p[0],x.shape[i]-p[0]) for i,p in enumerate(arg))
        ret = x.slice(arg=tuple(arg))
        assert ret.shape == tuple(end - start for start, end in arg)
        return ret
    
    def backward(self, output_grad: LazyBuffer):
        ret = output_grad.slice(arg=self.narg)
        return ret if self.need_input_grad[0] else None

class Max(Function):
    def forward(self,x,axis=None,keepdim=False):
        ret = x.reduce_op(ReduceOps.MAX,axis,keepdim)
        self.axis = normalize_axis(axis,len(x.shape))
        self.keepdim = keepdim
        self.save_for_backward(x,ret)
        return ret
    
    # TODO 
    def backward(self, grad_output):
        if not self.need_input_grad[0]:
            return None
        x, ret = self.saved_tensors

        # put 1 in the reduced dim shape if keepdim is False.
        if not self.keepdim:
            ret = ret.movement_op(MovementOps.RESHAPE,keepdim_shape_from_reduced(ret.shape, self.axis, len(x.shape)))
        # 1s in locations where the max was chosen (can be two locations)
        max_is_1s = x.binary_op(BinaryOps.CMPEQ, ret.movement_op(MovementOps.EXPAND, x.shape))
        # # sum of locations, averaged
        div = max_is_1s.reduce_op(ReduceOps.SUM, self.axis,True)

        div = div.movement_op(MovementOps.EXPAND, x.shape)
        max_is_amount = max_is_1s.binary_op(BinaryOps.DIV, div)

        # put 1 in reduced dims; 
        if not self.keepdim:
            grad_output = grad_output.movement_op(MovementOps.RESHAPE,keepdim_shape_from_reduced(grad_output.shape, self.axis, len(x.shape)))
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
        self.axis, self.keepdim = normalize_axis(axis, len(self.input_shape)), keepdim
        return x.reduce_op(ReduceOps.SUM,axis,keepdim)

    def backward(self, output_grad: LazyBuffer):
        if not self.need_input_grad[0]:
            return None
        
        grad = output_grad

        # If keepdim=False, reinsert dimensions
        if not self.keepdim:
            new_shape = keepdim_shape_from_reduced(grad.shape,self.axis,len(self.input_shape))
            grad = grad.movement_op(MovementOps.RESHAPE,tuple(new_shape))
            
        # Now broadcast safely
        return grad.movement_op(MovementOps.EXPAND,self.input_shape)
    
class Permute(Function):

    def forward(self, x: LazyBuffer, order: tuple):
        self.input_order = order
        return x.movement_op(MovementOps.PERMUTE, order)
    
    # argsort; sort the [0,1,2] order based on the key which is input_order
    def backward(self,output_grad: LazyBuffer):
        return output_grad.movement_op(MovementOps.PERMUTE, tuple(sorted(range(len(self.input_order)), key=self.input_order.__getitem__))) if self.need_input_grad[0] else None

class Flip(Function):
    def forward(self,x,axis):
        self.axis = axis
        return x.movement_op(MovementOps.FLIP,axis)
    
    def backward(self,output_grad):
        return output_grad.movement_op(MovementOps.FLIP,self.axis)


class Conv2D(Function):

    def forward(self,x: LazyBuffer,w: LazyBuffer,stride=1, groups=1, dilation=1, padding=0):
        self.save_for_backward(x,w)
        self.conv_args = get_conv_args(x.shape, w.shape, stride=stride, groups=groups, dilation=dilation, padding=padding)
        return x.processing_op(ProcessingOps.CONV,w,self.conv_args)
    
    def backward(self,output_grad):
        x,w = self.saved_tensors
        C = self.conv_args
        dx,dw = None, None
        # for input image grad
        if self.need_input_grad[0]: 
            # gradient w.r.t input
            xt = output_grad
            if C.sx > 1 or C.sy > 1:
                # If the forward conv had stride > 1, 
                # the backward conv must insert zeros between elements.
                xt = xt.movement_op(MovementOps.RESHAPE, (output_grad.shape[0], output_grad.shape[1], output_grad.shape[2], 1, output_grad.shape[3], 1))
                xt = xt.movement_op(MovementOps.PAD, ((0,0), (0,0), (0,0), (0,C.sy-1), (0,0), (0,C.sx-1)))
                xt = xt.movement_op(MovementOps.RESHAPE, (xt.shape[0], xt.shape[1], xt.shape[2]*C.sy, xt.shape[4]*C.sx))
            # flip 180 then, swap input/output channels,
            wt = w.movement_op(MovementOps.RESHAPE, (C.groups, C.rcout, C.cin, C.H, C.W)).movement_op(MovementOps.PERMUTE, (0, 2, 1, 3, 4)).movement_op(MovementOps.FLIP, (3, 4))
            wt = wt.movement_op(MovementOps.RESHAPE, (C.groups*C.cin, C.rcout, C.H, C.W))
            py, px = (C.H-1)*C.dy - C.py, (C.W-1)*C.dx - C.px
            Cdx = get_conv_args(xt.shape, wt.shape, out_shape=x.shape, dilation=(C.dy, C.dx), padding=(py, px), groups=C.groups)
            dx = xt.processing_op(ProcessingOps.CONV, wt, Cdx)
        # for filter grad
        if self.need_input_grad[1]:
            xdw = x.movement_op(MovementOps.RESHAPE, (C.bs, C.groups, C.cin, C.iy, C.ix)).movement_op(MovementOps.PERMUTE, (2, 1, 0, 3, 4))
            xdw = xdw.movement_op(MovementOps.RESHAPE, (C.cin, C.groups*C.bs, C.iy, C.ix))
            grad_output_dw = output_grad.movement_op(MovementOps.PERMUTE, (1,0,2,3))
            Cdw = get_conv_args(xdw.shape, grad_output_dw.shape, out_shape=(w.shape[1], w.shape[0], w.shape[2], w.shape[3]), padding=(C.py, C.px), stride=(C.dy, C.dx), dilation=(C.sy, C.sx), groups=C.groups)
            dw = xdw.processing_op(ProcessingOps.CONV, grad_output_dw, Cdw).movement_op(MovementOps.PERMUTE, (1,0,2,3))

        return dx,dw

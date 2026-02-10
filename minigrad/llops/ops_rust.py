import numpy as np
from minigrad.ops import LazyOp, BinaryOps, UnaryOps, ReduceOps, MovementOps
import rust_backend
from minigrad.helpers import gen_stride, stride_broadcast
import math

class RCPUBuffer:
    fn_to_rfn = {
        BinaryOps.MUL: rust_backend.mul, BinaryOps.ADD: rust_backend.add,
        BinaryOps.DIV: rust_backend.div, BinaryOps.POW: rust_backend.pow,
        BinaryOps.SUB: rust_backend.sub, BinaryOps.CMPEQ: rust_backend.cmpeq,
        UnaryOps.RELU: rust_backend.relu, UnaryOps.LOG: rust_backend.log,
        UnaryOps.EXP: rust_backend.exp, UnaryOps.RECIPROCAL: rust_backend.reciprocal,
        UnaryOps.NEG: rust_backend.neg, UnaryOps.SIGN: rust_backend.sign,
        UnaryOps.NOOP: rust_backend.noop, ReduceOps.SUM: rust_backend.sum,
        ReduceOps.MAX: rust_backend.max
    }
    def __init__(self,st,f_data):
        self.st = st
        self.f_data = f_data
    @property
    def stride(self): return self.st.stride
    @property
    def shape(self): return self.st.shape
    def __repr__(self):
        return f"<CPUBuffer with shape{self.st.shape} and stride{self.st.stride}>"
    
    @classmethod
    def exec_ast(cls, ast: LazyOp, out_shape):
        if hasattr(ast,"realize"):
            return ast.realize()
        
        assert type(ast) == LazyOp
        srcs = [cls.exec_ast(x,None) for x in ast.src]
        
        assert out_shape is not None
        if ast.op in BinaryOps:
            ret = srcs[0].binary_op(srcs[1],out_shape,ast.op)
        elif ast.op in UnaryOps:
            ret = srcs[0].unary_op(out_shape,ast.op)
        elif ast.op in ReduceOps:
            ret = srcs[0].reduce_op(out_shape,ast.op,ast.arg[0])
        elif ast.op in MovementOps:
            ret = srcs[0].movement_op(out_shape,ast.op,ast.arg)
        else:
            raise NotImplementedError(f"{ast.op}")
        return ret
    
    @staticmethod
    def fromCPU(x):
        return RCPUBuffer(ShapeTracker(x.shape,gen_stride(x.shape)),x.ravel())
    
    def toCPU(x):
        x = x.contiguous()
        return np.array(x.f_data).reshape(x.st.shape)
    
    def binary_op(x,y,out_shape,op):
        assert out_shape == x.st.shape == y.st.shape
        ret_data = fn_to_rfn[op](x.f_data,y.f_data,x.st.stride,y.st.stride,out_shape)
        return RCPUBuffer(ShapeTracker(out_shape,gen_stride(out_shape)),ret_data)
    
    def unary_op(x,out_shape,op):
        assert out_shape == x.st.shape
        ret_data = fn_to_rfn[op](x.f_data,x.stride,out_shape)
        return RCPUBuffer(ShapeTracker(out_shape,gen_stride(out_shape)),ret_data)
    
    def reduce_op(x,out_shape,op,axes):
        assert math.prod(x.st.shape) == math.prod([x.st.shape[ax] for ax in axes]+list(out_shape))
        ret_data = fn_to_rfn[op](x.f_data,x.shape,x.stride,out_shape,axes)
        return RCPUBuffer(ShapeTracker(out_shape,gen_stride(out_shape)),ret_data)
    
    def movement_op(x,out_shape,op,arg):
        if op is MovementOps.PERMUTE:
            assert out_shape == tuple([x.shape[dim_idx] for dim_idx in arg])
            stride = [x.stride[o] for o in arg]
            return RCPUBuffer(ShapeTracker(out_shape,stride),x.f_data)
        
        if op is MovementOps.MASKED_FILL:
            assert out_shape == x.shape and arg[0].shape == out_shape
            ret_data = rust_backend.masked_fill(x.f_data,x.stride,arg[0].ravel,arg[1],out_shape)
            return RCPUBuffer(ShapeTracker(out_shape,gen_stride(out_shape)),ret_data)
        
        if op is MovementOps.PAD:
            assert out_shape == tuple([x.shape[i]+before_i+after_i for i,(before_i,after_i) in enumerate(arg)])
            ret_data = rust_backend.pad(x.f_data,x.shape,x.stride,arg,out_shape)
            return RCPUBuffer(ShapeTracker(out_shape,gen_stride(out_shape)),ret_data)
        
        if op is MovementOps.SHRINK:
            assert out_shape == tuple([end-start for start,end in arg])
            ret_data = rust_backend.shrink(x.f_data,x.stride,arg,out_shape)
            return RCPUBuffer(ShapeTracker(out_shape,gen_stride(out_shape)),ret_data)
        
        if op is MovementOps.STRIDED:
            assert out_shape ==tuple([a[0] for a in arg])
            stride = [a[1] for a in arg]
            return RCPUBuffer(ShapeTracker(out_shape,stride),x.f_data)
        
        if op is MovementOps.FLIP:
            assert out_shape == x.shape
            ret_data = rust_backend.flip(x.f_data,x.stride,arg,out_shape)
            return RCPUBuffer(ShapeTracker(out_shape,gen_stride(out_shape)),ret_data)
        
        if op is MovementOps.RESHAPE:
            x = x.contiguous()
            return RCPUBuffer(ShapeTracker(out_shape,gen_stride(out_shape)),x.f_data)
        
        if op is MovementOps.EXPAND:
            assert out_shape == arg
            stride = stride_broadcast(x.shape, arg, x.stride)
            return RCPUBuffer(ShapeTracker(out_shape,stride),x.f_data)
    
    def contiguous(x):
        return x if x.st.is_contiguous() else x.unary_op(x.st.shape,UnaryOps.NOOP)

fn_to_rfn = {
        BinaryOps.MUL: rust_backend.mul, BinaryOps.ADD: rust_backend.add,
        BinaryOps.DIV: rust_backend.div, BinaryOps.POW: rust_backend.pow,
        BinaryOps.SUB: rust_backend.sub, BinaryOps.CMPEQ: rust_backend.cmpeq,
        UnaryOps.RELU: rust_backend.relu, UnaryOps.LOG: rust_backend.log,
        UnaryOps.EXP: rust_backend.exp, UnaryOps.RECIPROCAL: rust_backend.reciprocal,
        UnaryOps.NEG: rust_backend.neg, UnaryOps.SIGN: rust_backend.sign,
        UnaryOps.NOOP: rust_backend.noop, ReduceOps.SUM: rust_backend.sum,
        ReduceOps.MAX: rust_backend.max
    }

class ShapeTracker:
    def __init__(self,shape,stride):
        self.shape = shape
        self.stride = stride
        
    def is_contiguous(self,):
        return self.stride == gen_stride(self.shape)
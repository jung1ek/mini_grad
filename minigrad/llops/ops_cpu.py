from __future__  import annotations
import numpy as np
from minigrad.ops import GenericExecAST
from minigrad.ops import BinaryOps,UnaryOps,MovementOps,ReduceOps,ProcessingOps
import operator
from minigrad.helpers import shape_to_axis,ConvArgs

class CPUBuffer(np.ndarray,GenericExecAST):
    # all the x are realized data.
    fxn_for_op = {
        BinaryOps.MUL: operator.mul, BinaryOps.ADD: operator.add,
        MovementOps.EXPAND: lambda x,shape: CPUBuffer.expand(x,shape), MovementOps.RESHAPE: lambda x, shape: CPUBuffer.reshape(x,shape),
        MovementOps.PERMUTE: lambda x, order: CPUBuffer.permute(x,order),
        ReduceOps.SUM: lambda x, axis,keepdims: x.sum(axis=axis,keepdims=keepdims).view(CPUBuffer), ReduceOps.MAX: None,
        MovementOps.STRIDED: lambda x,args: CPUBuffer.strided(x,args)
    }

    def relu(x): return np.maximum(0,x)
    def log(x): return np.log(x)
    def exp(x): return np.exp(x)
    def sign(x): return np.sign(x)
    def float(x): return x.astype(np.float32)
    def flip(x,axis): return np.flip(x,axis)
    def amax(x,*args,**kwargs): return np.amax(x,*args,**kwargs)
    def pad(x,padding): return np.pad(x,padding).view(CPUBuffer)
    def expand(x,shape) : return np.broadcast_to(x,shape=shape).view(CPUBuffer)
    def reshape(x,shape) : return np.reshape(x,shape).view(CPUBuffer)
    def strided(x,arg): return np.lib.stride_tricks.as_strided(x.ravel().reshape(x.shape),shape=[y[0] for y in arg],strides=[y[1]*x.dtype.itemsize for y in arg]).view(CPUBuffer)
    def permute(x,order) : return np.transpose(x,order).view(CPUBuffer)

    @staticmethod
    def fromCPU(x: np.ndarray) : return x.view(CPUBuffer)
    def toCPU(x: np.ndarray) -> np.ndarray: return x

    def load_op(x): return CPUBuffer.fromCPU(x)
    def unary_op(x,op): return CPUBuffer.fxn_for_op[op](x)
    def binary_op(x,op,y): return CPUBuffer.fxn_for_op[op](x,y)
    def movement_op(x,op,shape_order): return CPUBuffer.fxn_for_op[op](x,shape_order)
    def reduce_op(x,op,axis,keepdims): return CPUBuffer.fxn_for_op[op](x,axis,keepdims)
    
    def processing_op(x,op,w,C: ConvArgs):
        # TODO can use sliding window view? much safer
        tx = x.movement_op(MovementOps.STRIDED,((C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
            (C.oy, C.sy*x.shape[3]), (C.ox, C.sx), (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
        tw = w.reshape((C.groups, C.rcout, C.cin, C.H,C.W)) # separate groups and cout per group
        out = np.einsum("nGhwCHW, GkCHW -> nGkhw", tx.ravel().reshape(tx.shape), tw.ravel().reshape(tw.shape))
        return out.reshape((C.bs,C.groups*C.rcout,C.oy,C.ox)).view(CPUBuffer)

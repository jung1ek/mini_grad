from __future__ import annotations
from typing import NamedTuple, Union, Tuple,Any
from enum import Enum
import numpy as np
from minigrad.ops import LazyOp, OpType,LoadOps,ReduceOps,MovementOps,BinaryOps,UnaryOps,ProcessingOps
from minigrad.llops.ops_cpu import CPUBuffer
from minigrad.helpers import reduce_shape
from minigrad.helpers import ConvArgs, get_conv_args
import sys

sys.setrecursionlimit(10000)

class Device:
    pass

# TODO optimization, 
class LazyBuffer:
    def __init__(self,shape,device,op_type:OpType,op:LazyOp):
        self.op_type = op_type
        self.shape = shape
        self.op: LazyOp = op
        self.device = device
        self.realized = None # DeviceBuffer; computed data
        self.children = None # Other lazy Buffer

    def __repr__(self): return f"<LB {self.shape} op:{self.op.op if self.realized is None else 'realized'}>"
    
    def _realize_binaryops(self):
        return CPUBuffer.exec_ast(self.op)
    
    def _realize_unaryops(self):
        return CPUBuffer.exec_ast(self.op)

    def _realize_loadops(self):
        return CPUBuffer.exec_ast(self.op)
    
    def _realize_reduceops(self):
        return CPUBuffer.exec_ast(self.op)
    
    def _realize_movementops(self):
        return CPUBuffer.exec_ast(self.op)
    
    def _realize_processingops(self):
        return CPUBuffer.exec_ast(self.op)

    # Forcing Computation.
    def realize(self) -> CPUBuffer:
        if self.realized is not None:
            return self.realized
        _realize = {LoadOps: self._realize_loadops,BinaryOps: self._realize_binaryops,
                    ReduceOps: self._realize_reduceops, MovementOps: self._realize_movementops,
                    ProcessingOps: self._realize_processingops, UnaryOps: self._realize_unaryops}
        
        # Realize via AST
        self.realized = _realize[self.op_type]()
        assert self.realized.shape == self.shape,f"Shape mismatch: expected {self.shape}, got {self.realized.shape}"
        
        # free graph memory
        del self.op
        del self.children
        return self.realized

    @staticmethod
    # Creating a LazyBuffer from CPU data
    def fromCPU(x,device):
        return LazyBuffer(x.shape,device,LoadOps,LazyOp(LoadOps.FROMCPU,src=tuple(),arg=x))

    # Getting data back to CPU
    def toCPU(self):
        return self.realize().toCPU()
    
    # creating lazy buffer through operations, z(new_buffer) = x(current_buffer)+(op) y(other_buffer)
    def movement_op(self,op:MovementOps,arg:tuple):
        if op is MovementOps.PERMUTE:
            # parameter shape is order for permute op.
            shape = tuple([self.shape[dim_idx] for dim_idx in arg])
        elif op is MovementOps.MASKED_FILL:
            shape = self.shape
        elif op is MovementOps.PAD:
            shape = tuple([self.shape[i]+before_i+after_i for i,(before_i,after_i) in enumerate(arg)])
        elif op is MovementOps.SHRINK:
            shape = tuple([end-start for start,end in arg])
        else:
            shape = arg
        
        # merge the consecutive op into single; a -> reshpae -> b -> reshape; a -> reshape (arg is merge.); a(src) of b(self)
        if self.realized is None and self.op.op==op:
            if op in [MovementOps.RESHAPE, MovementOps.EXPAND, MovementOps.SHRINK]:
                return self.op.src[0].movement_op(op,arg)
            elif op is MovementOps.PERMUTE:
                return self.op.src[0].movement_op(op,tuple(self.op.arg[i] for i in arg))
            elif op is MovementOps.PAD:
                return self.op.src[0].movement_op(op,tuple((b1+b2,a1+a2) for (b1,a1),(b2,a2) in zip(self.op.arg,arg)))
            
        return LazyBuffer(shape=shape,device=self.device,op_type=MovementOps,op=LazyOp(op,(self,),arg=arg))
    
    def binary_op(self,op:BinaryOps,other:LazyBuffer): # x(self) & y(other)
        return LazyBuffer(self.shape,self.device,BinaryOps,LazyOp(op,(self,other)))
    
    def unary_op(self,op:UnaryOps):
        return LazyBuffer(self.shape,self.device,UnaryOps,LazyOp(op,(self,)))
    
    def reduce_op(self, op: ReduceOps,axis,keepdim):
        return LazyBuffer(reduce_shape(self.shape,axis=axis,keepdim=keepdim),self.device,ReduceOps,LazyOp(op,(self,),arg=(axis,keepdim)))
    
    def processing_op(x,op:ProcessingOps,w:LazyBuffer,C: ConvArgs):
        assert op==ProcessingOps.CONV
        # pad
        x = x.slice(((0, x.shape[0]), (0, x.shape[1]), (-C.py, x.shape[2]+C.py_), (-C.px, x.shape[3]+C.px_)))
        # strided view, with channel out single dimension; rcout, channel out per group
        x = x.movement_op(MovementOps.STRIDED, (
        (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
        (1, 1), (C.oy, C.sy*x.shape[3]), (C.ox, C.sx),
        (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
        # expand to the broadcasting shape
        x = x.movement_op(MovementOps.EXPAND, (C.bs, C.groups, C.rcout, C.oy, C.ox, C.cin, C.H, C.W))
        # braodcast the w to match x
        w = w.movement_op(MovementOps.RESHAPE, (1, C.groups, C.rcout, 1, 1, C.cin, C.H, C.W)) \
            .movement_op(MovementOps.EXPAND, (C.bs, C.groups, C.rcout, C.oy, C.ox, C.cin, C.H, C.W))
        # mul and reduce,
        return x.binary_op(BinaryOps.MUL, w).reduce_op(ReduceOps.SUM, (5, 6, 7),True) \
                                            .movement_op(MovementOps.RESHAPE, (C.bs, C.cout, C.oy, C.ox))
        
    def slice(x, arg):
        # Pad first then shrink, pad to make valid indices; only if needed
        padding = [(max(0,-p[0]),max(0,p[1]-x.shape[i])) for i,p in enumerate(arg)]
        return x.movement_op(MovementOps.PAD,padding).movement_op(MovementOps.SHRINK,tuple((p[0]+padding[i][0],p[1]+padding[i][0]) for i,p in enumerate(arg)))
    
    def elementwise_op(op,*srcs):
        srcs = None
        return LazyBuffer


from __future__ import annotations
from typing import NamedTuple, Union, Tuple,Any
from enum import Enum
import numpy as np
from minigrad.ops import LazyOp, OpType,LoadOps,ReduceOps,MovementOps,BinaryOps,UnaryOps
from minigrad.llops.ops_cpu import CPUBuffer

class Device:
    pass

class LazyBuffer:
    def __init__(self,shape,device,op_type:OpType,op:LazyOp):
        self.op_type = op_type
        self.shape = shape
        self.op: LazyOp = op
        self.device = device
        self.realized = None # DeviceBuffer, computed data
        self.children = None # Other lazy Buffer

    def __repr__(self): return f"<LB {self.shape} op:{self.op.op if self.realized is None else 'realized'}>"
    
    def _realize_binaryops(self):
        return CPUBuffer.exec_ast(self.op)

    def _realize_loadops(self):
        return CPUBuffer.exec_ast(self.op)

    # Forcing Computation.
    def realize(self):
        if self.realized is not None:
            return self.realized
        _realize = {LoadOps: self._realize_loadops,BinaryOps: self._realize_binaryops}
        self.realized = _realize[self.op_type]()
        del self.op
        return self.realized

    @staticmethod
    # Creating a LazyBuffer from CPU data
    def fromCPU(x,device):
        return LazyBuffer(x.shape,device,LoadOps,LazyOp(LoadOps.FROMCPU,src=tuple(),arg=x))

    # Getting data back to CPU
    def toCPU(self):
        return self.realize().toCPU()
    
    # creating lazy buffer through operations, z(new_buffer) = x(current_buffer)+(op) y(other_buffer)
    def movement_op(self,op:MovementOps,shape:tuple):
        return LazyBuffer(shape=shape,device=self.device,op_type=MovementOps,op=LazyOp(op,(self,),shape))
    
    def binary_op(self,op:BinaryOps,other:LazyBuffer): # x(self) & y(other)
        return LazyBuffer(self.shape,self.device,BinaryOps,LazyOp(op,(self,other)))
    
    def unary_op(self,op:UnaryOps):
        return LazyBuffer(self.shape,self.device,UnaryOps,LazyOp(op,(self,)))


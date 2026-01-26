from __future__ import annotations
import sys, os
import math
import pyopencl as cl
from typing import Optional, Union
import numpy as np

from minigrad.ops import ExplicitExecAST, LazyOp, BinaryOps, OpType, MovementOps, LoadOps, UnaryOps

CL_DEVICE = int(os.getenv("CL_DEVICE","0"))

class CL:
    cl_ctx: Optional[cl.Context] = None
    cl_queue: Optional[cl.CommandQueue] = None
    def __init__(self):
        if CL.cl_ctx is not None:
            return
        devices = devices = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
        # connection to device
        CL.cl_ctx = cl.Context(devices=[devices[CL_DEVICE]])
        # command stream
        CL.cl_queue = cl.CommandQueue(self.cl_ctx) #or CL.cl_ctx
    
    @staticmethod
    def enqueue_copy(a:np.ndarray,b:cl.Buffer):
        cl.enqueue_copy(CL().cl_queue,a,b)

# kernal: function executed on GPU
# ND Range: grid of threads
# Global ID: Thread index.
class CLProgram:
    parameters = []
    expression = []

    @classmethod
    def build_kernal(cls,out_shape):
        code = []
        # kernal name and parameters
        code.append("__kernel void fused(")
        code.append(", ".join(f"__global float* {b}" for b in cls.parameters))
        code.append(", __global float* out) {")

        # expression or statements
        code.append("int gid = get_global_id(0);")
        code.append(f"if (gid >= {int(math.prod(out_shape))}) return;")

        code+=cls.expression
        return "\n".join(code)


class GPUBUffer(ExplicitExecAST):
    load_buf_counter = 0
    expression_buf_counter = 0

    fxn_to_code: dict[OpType,str] = {
        BinaryOps.MUL: "*", BinaryOps.ADD:"+"
    }

    def __init__(self, shape, stride,name=None, hostbuf: cl.Buffer=None):
        super().__init__()
        if name is None:
            name = f"a{GPUBUffer.load_buf_counter}"
            GPUBUffer.load_buf_counter+=1
        self.name = name
        self.shape = shape
        self.stride = stride
        self.hostbuf = hostbuf
    
    def __repr__(self):
        return f"<GPU Buffer with shape:{self.shape!r}>"
    
    # expression building
    @classmethod
    def exec_ast(cls, ast: LazyOp,shape):
        CLProgram.parameters = []
        CLProgram.expression = []
        cls.expression_buf_counter = 0
        load_buffers: list[cl.Buffer] = []

        def _ast(x: LazyOp) -> str:
            if hasattr(x, "realize"):
                # detect leaf
                if x.op_type == LoadOps:
                    buf = x.realize()
                    if buf.hostbuf not in load_buffers:
                        load_buffers.append(buf.hostbuf) 
                    return f"{buf.name}[gid]"
                # continue recursion
                return _ast(x.op)
            
            srcs = [_ast(x) for x in x.src]

            if x.op in BinaryOps:
                a = srcs[0]
                b = srcs[1]
                var = f"t{cls.expression_buf_counter}"
                cls.expression_buf_counter += 1

                if x.op == BinaryOps.ADD:
                    CLProgram.expression.append(f"float {var} = {a} + {b};")
                elif x.op == BinaryOps.MUL:
                    CLProgram.expression.append(f"float {var} = {a} * {b};")

                return var

            elif x.op in MovementOps:
                return srcs[0]
            elif x.op in UnaryOps:
                pass
            else:
                raise NotImplementedError(x.op)

        out = _ast(ast)
        CLProgram.expression.append(f"out[gid] = {out};"+"}")

        out_buf = cl.Buffer(CL().cl_ctx,cl.mem_flags.WRITE_ONLY,4*math.prod(shape))
        program = cl.Program(CL().cl_ctx, CLProgram.build_kernal(shape))
        program.build()
        program.fused(CL().cl_queue,(math.prod(shape),),None,*load_buffers,out_buf)
        return GPUBUffer(shape, None, name="out",hostbuf=out_buf)


    # creating hostbuf through pointer in cpu; register variables
    @staticmethod
    def fromCPU(x: np.ndarray):
        data = x.view(np.ndarray).astype(np.float32).ravel()
        hostbuf = cl.Buffer(CL().cl_ctx,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=data)
        ret = GPUBUffer(x.shape,None,None,hostbuf)

        CLProgram.parameters.append(ret.name)
        return ret
    
    def toCPU(self):
        data = np.empty(self.shape,dtype=np.float32)
        CL.enqueue_copy(data,self.hostbuf)
        return data
from __future__ import annotations
import sys, os
import math
import pyopencl as cl
from typing import Optional, Union
import numpy as np

from minigrad.ops import ExplicitExecAST, LazyOp, BinaryOps, OpType, MovementOps, LoadOps, UnaryOps

CL_DEVICE = int(os.getenv("CL_DEVICE","0"))

# TODO movements ops, unary, broadcasting, reduce
def gen_index(shape, strides, name):
    code = []
    div = 1
    idx = []

    for dim in reversed(shape):
        idx.append(div)
        div *= dim
    idx = idx[::-1]

    for i, d in enumerate(shape):
        code.append(f"int i{i} = (gid / {idx[i]}) % {d};")

    expr = " + ".join(f"i{i}*{strides[i]}" for i in range(len(shape)))
    code.append(f"int {name}_idx = {expr};")

    return code

def gen_stride(shape):
    strides= [1]*len(shape)
    for i in range(len(shape)-2,-1,-1):
        strides[i] = strides[i+1] * shape[i+1]
    return tuple(strides)

def stride_boradcast(orig_shape,target_shape,orig_stride):
    new_stride = list(orig_stride)
    for i in range(len(orig_stride)):
        if orig_shape[i] == target_shape[i]:
            new_stride[i] = orig_stride[i]
        elif orig_shape[i] == 1:
            new_stride[i] = 0
        # else:
        #     raise Exception
    return tuple(new_stride)


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
    def build_kernal(cls,out_shape,named_buffers):
        code = []
        # kernal name and parameters
        code.append("__kernel void fused(")
        code.append(", ".join(f"__global float* {b}" for b in cls.parameters))
        code.append(", __global float* out) {")

        # expression or statements
        code.append("int gid = get_global_id(0);")
        code.append(f"if (gid >= {int(math.prod(out_shape))}) return;")

        # indexing statements
        for _,buf in named_buffers.items():
            code+= gen_index(buf.shape,buf.stride,buf.name)
        
        #arithmetic exp
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
        named_buffers = {}

        def _ast(x: LazyOp,shape) -> str:
            if hasattr(x, "realize"):
                # detect leaf
                if x.op_type == LoadOps:
                    buf = x.realize()
                    name = f"{buf.name}"+f"[{buf.name}_idx]"
                    if buf.hostbuf not in load_buffers:
                        CLProgram.parameters.append(buf.name)
                        load_buffers.append(buf.hostbuf)
                        named_buffers[name] = buf
                    return name,buf
                # continue recursion
                return _ast(x.op,x.shape)
            
            srcs = [_ast(x,shape) for x in x.src]

            if x.op in BinaryOps:
                a,b = srcs[0][0],srcs[1][0]
                var = f"t{cls.expression_buf_counter}"
                cls.expression_buf_counter += 1

                if x.op == BinaryOps.ADD:
                    CLProgram.expression.append(f"float {var} = {a} + {b};")
                elif x.op == BinaryOps.MUL:
                    CLProgram.expression.append(f"float {var} = {a} * {b};")

                return var,GPUBUffer(shape,(1,1),var)

            elif x.op in MovementOps:
                if x.op == MovementOps.RESHAPE:
                    # print("reshpae",srcs)
                    # srcs[0][1].shape = shape
                    return srcs[0][0],srcs[0][1]
                elif x.op == MovementOps.EXPAND:
                    buf = named_buffers[srcs[0][0]]
                    # print(srcs)
                    # buf.stride = stride_boradcast(srcs[0][1].shape,shape,buf.stride)
                    return srcs[0][0],srcs[0][1]
            elif x.op in UnaryOps:
                pass
            else:
                raise NotImplementedError(x.op)

        out,_ = _ast(ast,shape)
        CLProgram.expression.append(f"out[gid] = {out};"+"}")

        out_buf = cl.Buffer(CL().cl_ctx,cl.mem_flags.WRITE_ONLY,4*math.prod(shape))
        # program = cl.Program(CL().cl_ctx, CLProgram.build_kernal(shape))
        # program.build()
        # program.fused(CL().cl_queue,(math.prod(shape),),None,*load_buffers,out_buf)
        print(CLProgram.build_kernal(shape,named_buffers))
        return GPUBUffer(shape, None, name="out",hostbuf=out_buf)


    # creating hostbuf through pointer in cpu; register variables
    @staticmethod
    def fromCPU(x: np.ndarray):
        data = x.view(np.ndarray).astype(np.float32).ravel()
        hostbuf = cl.Buffer(CL().cl_ctx,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=data)
        ret = GPUBUffer(x.shape,gen_stride(x.shape),None,hostbuf)
        return ret
    
    def toCPU(self):
        data = np.empty(self.shape,dtype=np.float32)
        CL.enqueue_copy(data,self.hostbuf)
        return None
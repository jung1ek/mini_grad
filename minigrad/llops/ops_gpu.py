from __future__ import annotations
import os, math
import pyopencl as cl
import numpy as np
from typing import Optional

from minigrad.ops import (ExplicitExecAST, LazyOp,BinaryOps, UnaryOps,
                          ReduceOps,MovementOps, LoadOps, OpType)
from minigrad.helpers import (gen_index_with_padding, gen_stride, 
                            is_contiguous, stride_broadcast)
from minigrad.helpers import find_reduce, replace_node
#TODO implement shape tracker for contigious.
CL_DEVICE = int(os.getenv("CL_DEVICE", "0"))

# OpenCL setup
class CL:
    ctx: Optional[cl.Context] = None
    queue: Optional[cl.CommandQueue] = None

    def __init__(self):
        if CL.ctx is not None:
            return
        devices = sum([p.get_devices(cl.device_type.GPU)\
                        for p in cl.get_platforms()],[])
        CL.ctx = cl.Context([devices[CL_DEVICE]])
        CL.queue = cl.CommandQueue(CL.ctx)

    @staticmethod
    def copy_to_host(dst: np.ndarray, src: cl.Buffer):
        cl.enqueue_copy(CL().queue, dst, src)

# GPU Buffer
class GPUBuffer(ExplicitExecAST):
    load_id = 0
    tmp_id = 0

    OPFN = {
    BinaryOps.ADD: "+", BinaryOps.MUL: "*",
    BinaryOps.SUB: "-",BinaryOps.DIV: "/",
    BinaryOps.CMPEQ: "==",
    UnaryOps.RECIPROCAL: "(1/",UnaryOps.RELU: "max(0.0f,",
    UnaryOps.EXP: "exp(",UnaryOps.LOG: "log(",
    UnaryOps.NEG: "(-", UnaryOps.SIGN: "sign(",
    }

    def __init__(self, shape, stride, name=None, buf=None):
        super().__init__()
        self.shape = shape
        self.stride = stride
        self.name = name
        self.buf = buf

    def __repr__(self):
        return f"<GPUBuffer shape={self.shape}>"
    
    # cpu -> gpu
    # Create GPUBuffer from numpy array
    @staticmethod
    def fromCPU(x: np.ndarray):
        x = x.astype(np.float32)
        buf = cl.Buffer(CL().ctx, cl.mem_flags.READ_ONLY |\
                        cl.mem_flags.COPY_HOST_PTR,hostbuf=x.ravel())
        name = f"a{GPUBuffer.load_id}"
        GPUBuffer.load_id+=1
        return GPUBuffer(x.shape, gen_stride(x.shape), name, buf)

    # gpu-cpu
    def toCPU(self):
        out = np.empty(self.shape, np.float32)
        CL.copy_to_host(out, self.buf)
        return out

    @classmethod
    def exec_ast(cls, ast: LazyOp, out_shape):
        CLProgram.params = []
        CLProgram.expr = []
        CLProgram.reduce_axes = []
        CLProgram.reduce_size = 0

        buffers = {}
        cl_buffers = []

        orig_meta = {}

        def snapshot(buf):
            if buf not in orig_meta:
                orig_meta[buf] = (buf.shape, buf.stride)
        def walk(x: LazyOp):
            if hasattr(x, "realize"):
              
              # detect leaf
              if x.op_type == LoadOps:
                buf = x.realize()
                if buf.name not in buffers:
                 CLProgram.params.append(buf.name)
                 buffers[buf.name] = buf
                 cl_buffers.append(buf.buf)
                return f"{buf.name}[{buf.name}_idx]", buf
              # continue tree
              return walk(x.op)
            assert type(x) == LazyOp
            src = [walk(s) for s in x.src]

            #TODO movement ops after the binary ops, changes is applied to new buffer?? issue
            # ---------------- Binary ----------------
            if x.op in BinaryOps:
                if CLProgram.reduce_size >= 1:
                    raise RuntimeError("Elementwise op after reduce detected.\
                                       ""Split kernel before reduce.")
                a, b = src[0][0], src[1][0]
                t = f"t{cls.tmp_id}"
                cls.tmp_id += 1
                if x.op == BinaryOps.POW:
                    CLProgram.expr.append(f"float {t} = pow({a} , {b});")
                else:
                    CLProgram.expr.append(f"float {t} = {a} {cls.OPFN[x.op]} {b};")
                assert src[0][1].shape==src[1][1].shape or src[1][1].shape == ()
                return t, GPUBuffer(src[0][1].shape, src[0][1].stride, t)
            
            if x.op in UnaryOps:
                if CLProgram.reduce_size >= 1:
                    raise RuntimeError("Elementwise op after reduce detected.\
                                       ""Split kernel before reduce.")
                a = src[0][0]
                t = f"t{cls.tmp_id}"
                cls.tmp_id += 1
                CLProgram.expr.append(f"float {t} = {cls.OPFN[x.op]} {a});")
                return t, src[0][1]
            # GPUBuffer(src[0][1].shape, src[0][1].stride, t)

            # ---------------- Reduce ----------------
            if x.op in ReduceOps:
                # only one reduce per kernel
                if CLProgram.reduce_size != 0:
                    raise RuntimeError("Internal error: multiple reduces in one kernel")
                axis = x.arg[0]
                CLProgram.reduce_axes = list(axis)
                CLProgram.reduce_size = 1
                CLProgram.reduce_size *= math.prod([src[0][1].shape[a] for a in axis])
                if x.op == ReduceOps.MAX:
                    CLProgram.reduce_op = x.op
                    CLProgram.expr.append(f"acc = max(acc,{src[0][0]});")
                elif x.op == ReduceOps.SUM:
                    CLProgram.reduce_op = x.op
                    CLProgram.expr.append("acc += " + src[0][0] + ";")
                return "acc", src[0][1]

            # ---------------- Broadcast ----------------
            if x.op == MovementOps.EXPAND:
                buf = src[0][1]
                snapshot(buf)
                buf.stride = stride_broadcast(buf.shape, x.arg, buf.stride)
                buf.shape = x.arg
                return src[0][0], buf

            # ---------------- Reshape ----------------
            if x.op == MovementOps.RESHAPE:
                buf = src[0][1]
                snapshot(buf)
                # what if some permuted tensor is being reshaped.
                if not is_contiguous(buf.shape,buf.stride):
                    if len(x.arg) != len(buf.stride):
                        raise NeedRealize(x.src[0])
                else:
                    buf.stride = gen_stride(x.arg) # contiguous view
                # if len(x.arg)>= len(buf.shape):
                buf.shape = x.arg
                return src[0][0], buf

            if x.op == MovementOps.PERMUTE:
                if x.src[0].op.op in BinaryOps:
                    print("Yes Binary")
                buf = src[0][1]
                snapshot(buf)
                order = x.arg
                buf.stride = [buf.stride[o] for o in order]
                buf.shape = tuple(buf.shape[o] for o in order)
                return src[0][0], buf
            
            if x.op == MovementOps.SHRINK:
                if x.src[0].op.op in BinaryOps:
                    print("Yes Binary")
                buf = src[0][1]
                snapshot(buf)
                buf.shape = tuple(e-s for s,e in x.arg)
                # offset; how much to ignore
                if not hasattr(buf,"offset"):
                    buf.offset = [0]*len(buf.shape)
                for i,(s,_) in enumerate(x.arg):
                    buf.offset[i] += s
                buf.base_offset = sum(buf.offset[i]*buf.stride[i] for i\
                                    in range(len(buf.offset)))
                return src[0][0],buf
            
            if x.op == MovementOps.PAD:
                if x.src[0].op.op in BinaryOps:
                    print("Yes Binary")
                buf = src[0][1]
                snapshot(buf)
                # buf.shape = tuple(buf.shape[i]+before+after for \
                #                   i,(before,after) in enumerate(x.arg))
                if not hasattr(buf,"offset"):
                    buf.offset = [0]*len(buf.shape)
                buf.padding = x.arg
                for i, (b,a) in enumerate(x.arg):
                    buf.offset[i]-=b
                    buf.shape = tuple(
                    buf.shape[j] + b + a if j == i else buf.shape[j]
                    for j in range(len(buf.shape))
                )
                buf.base_offset = sum(buf.offset[i]*buf.stride[i] for i\
                                    in range(len(buf.offset)))
                return f"{buf.name}_val",buf
            
            if x.op == MovementOps.FLIP:
                buf = src[0][1]
                snapshot(buf)
                axes = x.arg

                if not hasattr(buf, "offset"):
                    buf.offset = [0] * len(buf.shape)

                stride = list(buf.stride)
                offset = buf.offset

                for ax in axes:
                    stride[ax] *= -1
                    offset[ax] += buf.shape[ax] - 1

                buf.stride = tuple(stride)
                buf.offset = offset
                buf.base_offset = sum(buf.offset[i]*buf.stride[i] for i\
                                    in range(len(buf.offset)))
                return src[0][0], buf
            
            if x.op == MovementOps.STRIDED:
                if x.src[0].op.op in BinaryOps:
                    print("Yes Binary")
                buf = src[0][1]
                snapshot(buf)
                buf.stride = tuple(y[1]*np.float32().itemsize for y in x.arg)
                buf.shape = tuple(y[0] for y in x.arg)
                # *np.float32().itemsize
                return src[0][0],buf
            
            if x.op == LoadOps.FROMCPU:
                buf = cls.fromCPU(x.arg)
                if buf.name not in buffers:
                    CLProgram.params.append(buf.name)
                    buffers[buf.name] = buf
                    cl_buffers.append(buf.buf)
                return f"{buf.name}[{buf.name}_idx]", buf
            
            raise NotImplementedError(x.op)
        
        out_var, _ = walk(ast)
        if CLProgram.reduce_size == 0:
            CLProgram.expr.append(f"out[gid] = {out_var};")

        out_buf = cl.Buffer(CL().ctx,cl.mem_flags.WRITE_ONLY,\
                            4 * math.prod(out_shape))# in bytes

        src = CLProgram.build_kernal(out_shape, buffers)
        # print(src)
        program = cl.Program(CL().ctx, src).build()
        program.fused(CL().queue,(math.prod(out_shape),),\
                      None,*cl_buffers,out_buf)
        # getting back the orig shape and stride.
        for buf, (shape, stride) in orig_meta.items():
            buf.shape = shape
            buf.stride = stride
        return GPUBuffer(out_shape, gen_stride(out_shape), "out", out_buf)
    
    @classmethod
    def schedule(cls, ast, out_shape):
        while True:
            try:
                reduce_node, reduced_shape = find_reduce(ast, out_shape)
                if reduce_node is None:
                    if ast.op == LoadOps.FROMCPU:
                      return cls.fromCPU(ast.arg)
                    return cls.exec_ast(ast, out_shape)
                tmp = cls.exec_ast(reduce_node, reduced_shape)
                CLProgram.reset()
                ast = replace_node(ast,reduce_node,
                    LazyOp(LoadOps.FROMCPU,tuple(), arg=tmp.toCPU()))
            except NeedRealize as e:
                # materialize the offending subgraph
                sub = e.node
                if sub.realized is not None:
                    raise RuntimeError("Node requested realization twice (infinite loop guard)")
                # print(e.op_type)
                tmp = cls.exec_ast(sub.op, sub.shape)
                ast = replace_node(ast,sub.op,
                    LazyOp(LoadOps.FROMCPU,tuple(), arg=tmp.toCPU()))
        
class NeedRealize(Exception):
    def __init__(self, node):
        self.node = node

# Kernal Builder
class CLProgram:
    params = []
    expr = []
    reduce_axes = []
    reduce_size = 0

    reduce_op = None
    @classmethod
    def reset(cls):
        CLProgram.params = []
        CLProgram.expr = []
        CLProgram.reduce_axes = []
        CLProgram.reduce_size = 0

    @classmethod
    def build_kernal(cls, out_shape, buffers):
        code = []
        code.append("__kernel void fused(")
        code.append(", ".join(f"__global float* {b}" for b in cls.params))
        code.append(", __global float* out) {")

        code.append("int gid = get_global_id(0);")
        code.append(f"if (gid >= {math.prod(out_shape)}) return;")

        if cls.reduce_size > 0:
            if cls.reduce_op == ReduceOps.SUM:
                code.append("float acc = 0.0f;")
            elif cls.reduce_op == ReduceOps.MAX:
                code.append("float acc = -INFINITY;")
            code.append(f"for (int r = 0; r < {cls.reduce_size}; r++) {{")

        for i,buf in enumerate(buffers.values()):
            code += gen_index_with_padding(buf,buf.shape,buf.stride,buf.name,
                cls.reduce_axes if cls.reduce_size > 0 else None,
                prefix=f"a{i}_",out_shape=out_shape)

        code += cls.expr

        if cls.reduce_size > 0:
            code.append("}")
            code.append("out[gid] = acc;")
            code.append("}")
        else:
            code.append("}")

        return "\n".join(code)
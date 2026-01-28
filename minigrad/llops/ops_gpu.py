from __future__ import annotations
import os, math
import pyopencl as cl
import numpy as np
from typing import Optional

from minigrad.ops import (
    ExplicitExecAST, LazyOp,
    BinaryOps, UnaryOps, ReduceOps,
    MovementOps, LoadOps, OpType
)

# ============================================================
# OpenCL setup
# ============================================================

CL_DEVICE = int(os.getenv("CL_DEVICE", "0"))

class CL:
    ctx: Optional[cl.Context] = None
    queue: Optional[cl.CommandQueue] = None

    def __init__(self):
        if CL.ctx is not None:
            return
        devices = sum(
            [p.get_devices(cl.device_type.GPU) for p in cl.get_platforms()],
            []
        )
        CL.ctx = cl.Context([devices[CL_DEVICE]])
        CL.queue = cl.CommandQueue(CL.ctx)

    @staticmethod
    def copy_to_host(dst: np.ndarray, src: cl.Buffer):
        cl.enqueue_copy(CL().queue, dst, src)


# ============================================================
# Shape helpers
# ============================================================

def gen_stride(shape):
    stride = [1] * len(shape)
    for i in range(len(shape)-2, -1, -1):
        stride[i] = stride[i+1] * shape[i+1]
    return tuple(stride)

def stride_broadcast(orig_shape, target_shape, orig_stride):
    assert len(orig_shape) == len(target_shape) == len(orig_stride)
    out = []
    for os, ts, st in zip(orig_shape, target_shape, orig_stride):
        if os == ts:
            out.append(st)
        elif os == 1:
            out.append(0)
        else:
            raise ValueError("Invalid broadcast")
    return tuple(out)

# ============================================================
# Index generation (supports reduce!)
# ============================================================

def gen_index(shape, strides, name, reduce_axes=None, reduce_var="r",prefix=""):
    reduce_axes = reduce_axes or []
    code = []

    divs = []
    div = 1
    for d in reversed(shape):
        divs.append(div)
        div *= d
    divs = list(reversed(divs))

    for i, d in enumerate(shape):
        var = f"{prefix}"
        if i in reduce_axes:
            code.append(f"int {var}i{i} = {reduce_var};")
        else:
            code.append(f"int {var}i{i} = (gid / {divs[i]}) % {d};")

    expr = " + ".join(f"{prefix}i{i}*{strides[i]}" for i in range(len(shape)))
    # expr = " + ".join(
    # f"i{i}*{s}" for i, s in enumerate(strides)
    # )
    code.append(f"int {name}_idx = {expr};")
    return code


# ============================================================
# Kernel builder
# ============================================================

class CLProgram:
    params = []
    expr = []
    reduce_axes = []
    reduce_size = 1

    @classmethod
    def build(cls, out_shape, buffers):
        code = []
        code.append("__kernel void fused(")
        code.append(", ".join(f"__global float* {b}" for b in cls.params))
        code.append(", __global float* out) {")

        code.append("int gid = get_global_id(0);")
        code.append(f"if (gid >= {math.prod(out_shape)}) return;")

        if cls.reduce_size > 1:
            code.append("float acc = 0.0f;")
            code.append(f"for (int r = 0; r < {cls.reduce_size}; r++) {{")

        for i,buf in enumerate(buffers.values()):
            code += gen_index(
                buf.shape,
                buf.stride,
                buf.name,
                cls.reduce_axes if cls.reduce_size > 1 else None,
                prefix=f"a{i}_"

            )

        code += cls.expr

        if cls.reduce_size > 1:
            code.append("}")
            code.append("out[gid] = acc;")
        else:
            code.append("}")

        return "\n".join(code)


# ============================================================
# GPU Buffer
# ============================================================

class GPUBUffer(ExplicitExecAST):
    load_id = 0
    tmp_id = 0

    BINOP = {
    BinaryOps.ADD: "+",
    BinaryOps.MUL: "*",
    BinaryOps.SUB: "-",
    BinaryOps.DIV: "/",
    }

    def __init__(self, shape, stride, name=None, buf=None):
        super().__init__()
        self.shape = shape
        self.stride = stride
        self.name = name
        self.buf = buf

    def __repr__(self):
        return f"<GPUBuffer shape={self.shape}>"

    # --------------------------------------------------------
    # CPU → GPU
    # --------------------------------------------------------

    @staticmethod
    def fromCPU(x: np.ndarray):
        x = x.astype(np.float32)
        buf = cl.Buffer(
            CL().ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=x.ravel()
        )
        name = f"a{GPUBUffer.load_id}"
        GPUBUffer.load_id+=1
        return GPUBUffer(x.shape, gen_stride(x.shape), name, buf)

    def toCPU(self):
        out = np.empty(self.shape, np.float32)
        CL.copy_to_host(out, self.buf)
        return out


    # --------------------------------------------------------
    # Exec AST
    # --------------------------------------------------------

    @classmethod
    def exec_ast(cls, ast: LazyOp, out_shape):
        CLProgram.params = []
        CLProgram.expr = []
        CLProgram.reduce_axes = []
        CLProgram.reduce_size = 1

        buffers = {}
        cl_buffers = []

        def walk(x: LazyOp):
            if hasattr(x, "realize"):
                if x.op_type == LoadOps:
                    buf = x.realize()
                    if buf.name not in buffers:
                        CLProgram.params.append(buf.name)
                        buffers[buf.name] = buf
                        cl_buffers.append(buf.buf)
                    return f"{buf.name}[{buf.name}_idx]", buf

                return walk(x.op)

            src = [walk(s) for s in x.src]

            # ---------------- Binary ----------------
            if x.op in BinaryOps:
                a, b = src[0][0], src[1][0]
                t = f"t{cls.tmp_id}"
                cls.tmp_id += 1
                CLProgram.expr.append(f"float {t} = {a} {cls.BINOP[x.op]} {b};")
                return t, GPUBUffer(out_shape, None, t)

            # ---------------- Reduce ----------------
            if x.op in ReduceOps:
                axis = x.op.axis
                CLProgram.reduce_axes.append(axis)
                CLProgram.reduce_size *= src[0][1].shape[axis]
                CLProgram.expr.append("acc += " + src[0][0] + ";")
                return "acc", src[0][1]

            # ---------------- Broadcast ----------------
            if x.op == MovementOps.EXPAND:
                buf = src[0][1]
                buf.stride = stride_broadcast(buf.shape, out_shape, buf.stride)
                buf.shape = out_shape
                return src[0][0], buf

            # ---------------- Reshape ----------------
            if x.op == MovementOps.RESHAPE:
                new_shape = x.arg  # or x.op.arg / x.op.new_shape
                buf = src[0][1]
                buf.shape = new_shape
                buf.stride = gen_stride(new_shape)  # contiguous view
                return src[0][0], src[0][1]

            raise NotImplementedError(x.op)
        

        out_expr, _ = walk(ast)
        if CLProgram.reduce_size == 1:
            CLProgram.expr.append(f"out[gid] = {out_expr};")

        out_buf = cl.Buffer(
            CL().ctx,
            cl.mem_flags.WRITE_ONLY,
            4 * math.prod(out_shape) # in bytes
        )

        src = CLProgram.build(out_shape, buffers)
        program = cl.Program(CL().ctx, src).build()
        program.fused(
            CL().queue,
            (math.prod(out_shape),),
            None,
            *cl_buffers,
            out_buf
        )
        # print(src)
        return GPUBUffer(out_shape, None, "out", out_buf)




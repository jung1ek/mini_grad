from collections import namedtuple
from minigrad.ops import LazyOp, ReduceOps
# normalized the axes
def normalize_axis(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a if a >= 0 else a + ndim for a in axis)

# add 1 dims to the reduced axes, for keepdim false, 
def keepdim_shape_from_reduced(out_shape, axes, ndim):
    shape = list(out_shape)
    for ax in sorted(axes):
        shape.insert(ax, 1)
    assert len(shape) == ndim
    return tuple(shape)

# axis based on the old shape and new shape
def shape_to_axis(old_shape, new_shape):
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple([i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b])

# new shape based on the axis to sum
def reduce_shape(shape, axis,keepdim=False):
    
    # convert axis in to list of axes
    ndim = len(shape)
    if axis is None:
        axes = list(range(ndim))
    elif isinstance(axis, int):
        axes = [axis%ndim]
    else:
        axes = [(a%ndim) for a in axis]
    
    if keepdim:
        # change reduce axis to 1
        new_shape = [1 if i in axes else shape[i] for i in range(ndim)]
    else:
        # drop the reduced axis
        new_shape = [shape[i] for i in range(ndim) if i not in axes]
    return tuple(new_shape)

# def reduce_shape(shape, axis): return tuple(1 if i in axis else shape[i] for i in range(len(shape)))

ConvArgs = namedtuple("ConvArgs",['H', 'W', 'groups', 'rcout', 'cin', 'oy', 'ox', 'iy', 'ix', 'sy', 'sx', 'bs', 'cout', 'py', 'py_', 'px', 'px_', 'dy', 'dx', 'out_shape'])
def get_conv_args(x_shape, w_shape, stride=1, groups=1, padding=0, dilation=1, out_shape=None):
    cout,cin,H,W = w_shape
    sy,sx = (stride, stride) if isinstance(stride, int) else stride
    if not isinstance(padding, int) and len(padding) == 4:
        px,px_,py,py_ = padding
    else:
        py,px = (padding, padding) if isinstance(padding, int) else padding
        py_, px_ = py, px
    dy,dx = (dilation, dilation) if isinstance(dilation, int) else dilation
    bs,cin_,iy,ix = x_shape

    # this can change px_ and py_ to make the out_shape right
    # TODO: copy padding names from http://nvdla.org/hw/v1/ias/unit_description.html
    if out_shape is not None:
        py_ = (out_shape[2] - 1) * sy + 1 + dy * (H-1) - iy - py
        px_ = (out_shape[3] - 1) * sx + 1 + dx * (W-1) - ix - px

    # TODO: should be easy to support asymmetric padding by changing output size
    # output spatial size, (h,w)
    oy = (iy + py + py_ - dy * (H-1) - 1)//sy + 1
    ox = (ix + px + px_ - dx * (W-1) - 1)//sx + 1
    if cin*groups != cin_:
        raise Exception(f"Input Tensor shape {x_shape} does not match the shape of the weights {w_shape}. ({cin*groups} vs. {cin_})")
    assert cout % groups == 0 and (out_shape is None or out_shape == (bs, cout, oy, ox))
    return ConvArgs(H, W, groups, cout//groups, cin, oy, ox, iy, ix, sy, sx, bs, cout, py, py_, px, px_, dy, dx, (bs, cout, oy, ox))


# ============================================================
# GPU Shape helpers and opencl kernal index gen
# ============================================================

def is_contiguous(shape, stride):
    """Check if tensor is contiguous (C-order)"""
    if not stride:
        return True
    expected_stride = gen_stride(shape)
    return tuple(stride) == tuple(expected_stride)

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

def gen_index(buf,shape, strides, name, reduce_axes=None, reduce_var="r", prefix="",out_shape=None):
    reduce_axes = reduce_axes or []
    code = []
    
    # For reduce operations, we need to use out_shape for gid indexing
    # and shape for the full tensor indexing
    gid_shape = out_shape if (reduce_axes and out_shape) else shape

    # gid divs
    divs = []
    div = 1
    for d in reversed(gid_shape):
        divs.append(div)
        div *= d
    divs = list(reversed(divs))

    # reduce divs
    reduce_divs = []
    if reduce_axes:
        div = 1
        for ax in reversed(reduce_axes):
            reduce_divs.append(div)
            div *= shape[ax]
        reduce_divs = list(reversed(reduce_divs))
    out_dim = 0
    for i, d in enumerate(shape):
        if i in reduce_axes:
            ridx = reduce_axes.index(i)
            code.append(
                f"int {prefix}i{i} = ({reduce_var} / {reduce_divs[ridx]}) % {d};"
            )
        else:
            code.append(
                f"int {prefix}i{i} = (gid / {divs[out_dim]}) % {d};"
            )
            out_dim+=1

    expr_terms = [f"{prefix}i{i}*{strides[i]}" for i in range(len(shape))]
    if hasattr(buf, "offset"):
        expr_terms += [f"{buf.offset[i]}*{strides[i]}" for i in range(len(shape))]
    expr = " + ".join(expr_terms)
    code.append(f"int {name}_idx = {expr};")
    return code

def gen_index_with_padding(buf, shape, strides, name, reduce_axes=None, reduce_var="r",
                           prefix="", out_shape=None):
    """
    Generate C-like indexing code for a tensor, with optional reduction axes
    and optional padding (out-of-bounds values return 0.0f).
    
    padding: list of tuples [(pad_before_dim0, pad_after_dim0), ...] same length as shape
    """
    reduce_axes = reduce_axes or []
    code = []
    padding = buf.padding if hasattr(buf,"padding") else [(0,0)]*len(shape)
    
    # For reduce operations, we need to use out_shape for gid indexing
    # and shape for the full tensor indexing
    gid_shape = out_shape if (reduce_axes and out_shape) else shape

    # gid divs
    divs = []
    div = 1
    for d in reversed(gid_shape):
        divs.append(div)
        div *= d
    divs = list(reversed(divs))

    # reduce divs
    reduce_divs = []
    if reduce_axes:
        div = 1
        for ax in reversed(reduce_axes):
            reduce_divs.append(div)
            div *= shape[ax]
        reduce_divs = list(reversed(reduce_divs))
    
    out_dim = 0
    for i, d in enumerate(shape):
        pad_before, pad_after = padding[i]
        if i in reduce_axes:
            ridx = reduce_axes.index(i)
            code.append(
                f"int {prefix}i{i} = ({reduce_var} / {reduce_divs[ridx]}) % {d};"
            )
        else:
            # compute index for this dimension
            code.append(
                f"int {prefix}i{i} = (gid / {divs[out_dim]}) % {d};"
            )
            out_dim += 1

        if hasattr(buf,"padding"):
            # Apply padding logic
            if pad_before > 0 or pad_after > 0:
                # Compute shifted index
                code.append(f"int {prefix}i{i}_padded = {prefix}i{i} - {pad_before};")
                # Validity check
                code.append(
                    f"bool {prefix}i{i}_valid = ({prefix}i{i}_padded >= 0) && "
                    f"({prefix}i{i}_padded < {d});"
                )
            else:
                code.append(f"int {prefix}i{i}_padded = {prefix}i{i};")
                code.append(f"bool {prefix}i{i}_valid = true;")

    # Compute final flat index
    if hasattr(buf,"padding"):
        expr_terms = [f"{prefix}i{i}_padded*{strides[i]}" for i in range(len(shape))]
    else:
        expr_terms = [f"{prefix}i{i}*{strides[i]}" for i in range(len(shape))]
    if hasattr(buf, "offset"):
        expr_terms += [f"{buf.offset[i]}*{strides[i]}" for i in range(len(shape))]
    expr = " + ".join(expr_terms)
    code.append(f"int {name}_idx = {expr};")
    
    if hasattr(buf,"padding"):
        # Overall validity mask (all dims must be valid)
        valid_terms = [f"{prefix}i{i}_valid" for i in range(len(shape))]
        code.append(f"bool {name}_valid = " + " && ".join(valid_terms) + ";")

        # Load value with padding
        code.append(f"float {name}_val = {name}_valid ? {name}[{name}_idx] : 0.0f;")

    return code

def find_reduce(node, shape):
    if not isinstance(node, LazyOp):
        return None, None
    
    # Search children first (depth-first, post-order traversal)
    for s in getattr(node, "src", []):
        # Handle LazyBuffer wrapper
        if hasattr(s, "op"):
            r, sh = find_reduce(s.op, s.shape)
            if r is not None:
                return r, sh
        # Handle direct LazyOp
        elif isinstance(s, LazyOp):
            r, sh = find_reduce(s, shape)
            if r is not None:
                return r, sh
    
    # Only check current node if no reduce found in children
    if node.op in ReduceOps and isinstance(node, LazyOp):
        return node, shape
    return None, None

# broken
def replace_node(node, target, replacement):
    if node is target:
        return replacement
    if not hasattr(node, "src"):
        return node
    # Handle LazyBuffer sources
    if hasattr(node, "src") and len(node.src) > 0 and hasattr(node.src[0], "op"):
        new_src = tuple([
            type(s)(s.shape,s.device,s.op_type,replace_node\
                    (s.op, target, replacement))for s in node.src])
    else:
        # Handle direct LazyOp sources
        new_src = tuple([
            replace_node(s, target, replacement) for s in node.src
        ])
    return LazyOp(node.op, new_src, arg=node.arg)
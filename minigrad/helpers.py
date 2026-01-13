from collections import namedtuple

def normalize_axis(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a if a >= 0 else a + ndim for a in axis)

# axis based on the old shape and new shape
# TODO issue, while (3,5,1) - > (3,5,6) , gives ()
def shape_to_axis(old_shape,new_shape):
    # Case 1: keepdims=True
    if len(old_shape) == len(new_shape):
        return tuple(i for i, (o, n) in enumerate(zip(old_shape, new_shape)) if n == 1 and o != 1)

    # Case 2: keepdims=False
    axes = []
    j = 0  # pointer in new_shape
    for i, o in enumerate(old_shape):
        if j < len(new_shape) and new_shape[j] == o:
            j += 1
        else:
            axes.append(i)
    return tuple(axes)

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
        new_shape = [
            1 if i in axes else shape[i]
            for i in range(ndim)
        ]
    else:
        # drop the reduced axis
        new_shape = [
            shape[i] for i in range(ndim) if i not in axes
        ]
    return tuple(new_shape)

# def reduce_shape(shape, axis): return tuple(1 if i in axis else shape[i] for i in range(len(shape)))
def shape_to_axis(old_shape, new_shape):
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple([i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b])

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



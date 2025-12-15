# axis based on the old shape and new shape
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
def reduce_shape(shape, axis,keepdims=False):
    
    # convert axis in to list of axes
    ndim = len(shape)
    if axis is None:
        axes = list(range(ndim))
    elif isinstance(axis, int):
        axes = [axis%ndim]
    else:
        axes = [(a%ndim) for a in axis]
    
    if keepdims:
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
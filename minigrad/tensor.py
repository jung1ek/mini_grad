from __future__ import annotations
import numpy as np
from minigrad.lazy import LazyBuffer
from typing import Optional, Literal, Sequence
import functools, inspect, importlib
import math

# TODO view , squeeze, unsqueeze, getitem, setitem, activation fn,
# TODO fix shape, 
class Tensor:
    def __init__(self,data,device=None,requires_grad=None):
        if isinstance(data,list):
            data = np.array(data,dtype=np.float32)
        elif isinstance(data, LazyBuffer) and data.device != device:
            # lazy buffer realized on cpu, 
            data = data.realize().toCPU()

        if isinstance(data, np.ndarray):
            if data.shape == tuple(): # for scalar value, reshape to dim 1
                data = data.reshape((1,))
            # creates lazybuffer on specific device, with fromCPU operation LazyOp.
            self.lazydata = LazyBuffer.fromCPU(data.astype(np.float32),device)
        elif isinstance(data, LazyBuffer):
            self.lazydata = data
        else:
            raise Exception(f"Can't create Tensor from {data}")
        # tensor have grad but lazybuffer dont.
        self.grad: Optional[Tensor] = None
        self._ctx : Optional[Function] = None
        self.requires_grad : Optional[bool] = requires_grad

    
    def __repr__(self):
        return f"<Tensor {self.lazydata if self.lazydata.realized is None else self.lazydata.realized!r} with grad {(self.grad.lazydata if self.grad else None)!r}>"

    @property
    def data(self):
        return self.lazydata.realize()
    
    @property
    def device(self):
        return self.lazydata.device
    
    def detach(self): pass
    def numpy(self): return np.array(self.lazydata.toCPU())

    @classmethod
    def rand(cls,shape: tuple):
        assert type(shape)==tuple,""
        return cls(np.random.rand(*shape))
    
    @property
    def shape(self): return self.lazydata.shape

    @property
    def T(self): return self.transpose()
    
    def realize(self):
        self.lazydata.realize()
        return self
    
    #TODO halding unsupported shape
    def reshape(self,*shape) : return self._reshape.apply(self,shape=shape)
    def expand(self,*shape) : return self._expand.apply(self,shape=shape)

    def ndim(self) -> int: return len(self.shape)
    
    def flatten(self,start_dim=0,end_dim=-1):
        # resolve dim, index out of range
        start_dim, end_dim = self._resolve_dim(start_dim), self._resolve_dim(end_dim)
        flatten_shape = tuple(self.shape[:start_dim]+(math.prod(self.shape[start_dim:end_dim+1]),)+self.shape[end_dim+1:])
        return self.reshape(*flatten_shape)

    # uses reshape
    def squeeze(self):
        pass
    
    # uses reshape
    def unsqueeze(self):
        pass
    
    # TODO contiguous, 
    # view is alias for reshape
    def view(self,*shape):
        self.reshape(*shape)

    #function to resovle dim, make positive axes
    def _resolve_dim(self,dim:int):
        ndim = len(self.shape)
        # index out of range
        if not -max(1,ndim)<=dim<=max(1,ndim)-1:
            raise IndexError(f"{dim=} out of range {[-max(1, ndim), max(1, ndim) - 1]}")
        return dim+ndim if dim < 0 else dim
    
    def permute(self,order):
        order_arg = tuple(self._resolve_dim(x) for x in order)
        assert sorted(order_arg)==list(range(len(self.shape))),"Invalide Permutation"
        return self._permute.apply(self,order=order_arg)

    #TODO transpose uses permute
    def transpose(self, dim0=1,dim1=0):
        order = list(range(len(self.shape)))
        order[dim0],order[dim1] = order[dim1],order[dim0]
        return self.permute(order)

    # toposort and backpropagation
    def deepwalk(self):
        def _deepwalk(node,visited,nodes):
            visited.add(node)
            if node._ctx: # leaf doesnot have ctx, when we hit the final leaf, the recurse ends
                [_deepwalk(child,visited,nodes) for child in node._ctx.parents if child not in visited]
                nodes.append(node)
            return nodes
        return _deepwalk(self,set(),[])

    def backward(self):
        # Implicit gradient creation (assumes scalar output).
        assert self.shape == (1,), "Backward can only be called on scalar tensors."
        self.grad = Tensor([1],requires_grad=False,device=self.device)  # Initial gradient (dy/dy = 1)

        for node in reversed(self.deepwalk()):  # Process nodes in reverse topological order.
            if not any(x.requires_grad for x in node._ctx.parents):
                continue  # Skip if no parents require gradients.

            assert node.grad is not None, f"Gradient not found for node {node}"

            # Compute gradients via the backward pass of the operation.
            grads = node._ctx.backward(node.grad.lazydata)  # Returns gradients for parents.

            # Convert gradients to Tensors (handles single/multiple parents).
            grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
                    for g in ([grads] if len(node._ctx.parents) == 1 else grads)]

            # Accumulate gradients for each parent.
            for t, g in zip(node._ctx.parents, grads):
                if g is not None and t.requires_grad:
                    assert g.shape==t.shape, f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
                    t.grad = g if t.grad is None else (t.grad + g)  # Gradient accumulation.
            del node._ctx
    
    def __getitem__(self,value):
        arg, new_shape = [],[]
        # iterate through each slice index.
        for i, rs in enumerate(value if isinstance(value, (list,tuple)) else [value]) if value is not None else []:
            # if the indexing vlue is int not slice; (1) int , (1:2) slice
            s = slice(rs,rs+1,None) if isinstance(rs, int) else rs
            # slice into absolute bounds
            arg.append((s.start if s.start is not None else 0,(s.stop if s.stop>=0 else self.shape[i]+s.stop) if s.stop is not None else self.shape[i]))
            # disallow stride
            assert s.step is None or s.step ==1

            if not isinstance(rs,int):
                # stop - start, new shape if it is int then remove that dim
                new_shape.append(arg[-1][1]-arg[-1][0])
        # add un touched dim
        new_shape+= [self.shape[i] for i in range(len(arg),len(self.shape))]

        # slice all dimensions explictely (0, dim)
        arg = arg + [(0, self.shape[i]) for i in range(len(arg),len(self.shape))]
        # slice based on slice indices and reshape; need reshape (slice always preserves rank.) to drop int dim
        return self.slice.apply(self,arg=arg).reshape(*new_shape)
    
    # broad-casted binary ops
    @staticmethod
    def broadcasted(fxn : Function, x: Tensor, y: Tensor) -> Tensor:
        # prototype tensor
        tt = [arg for arg in [x,y] if isinstance(arg,Tensor)][0] 
        # convert number to Tensor
        x,y = [Tensor([t],device=tt.device,requires_grad=False) if not isinstance(t, Tensor) else t for t in [x,y]]
        # reshape for boradcasting, x=(2,3), then y=(1,1)
        x,y = [t.reshape(*([1]*(max(len(x.shape), len(y.shape))-len(t.shape)) + list(t.shape))) for t in [x,y]]
        # calculate output shape, eg: (2,3) vs (1,1) Result: (2,3)
        ret_shape = tuple(max(sx,sy) for sx, sy in zip(x.shape,y.shape))
        # expand (1,1) to match (2,3) with repeating elements.
        return fxn.apply(x.expand(*ret_shape),y.expand(*ret_shape))

    def mul(self,x: Tensor): return Tensor.broadcasted(self._mul,self,x)
    def add(self,x: Tensor): return Tensor.broadcasted(self._add,self,x)

    # test only before broadcasting
    def __mul__(self,other: Tensor): return self.mul(other)
    def __add__(self,other: Tensor): return self.add(other)
    
    # TODO reduce op function, handle int, negative indexing (-1), and calculate shape
    def _reduce(self,fxn, axis=None, keepdims=False):
        return fxn.apply(self,axis=axis,keepdims=keepdims)

    def sum(self,axis=None,keepdims=False): return self._reduce(Tensor._sum,axis=axis,keepdims=keepdims)
    def max(self,axis=None,keepdims=False): return self._reduce(Tensor._max,axis=axis,keepdims=keepdims)

    def matmul(self: Tensor,other: Tensor):
        # broad cast, multiply and sum over axis.
        x,y,dx,dy = self,other,len(self.shape),len(other.shape)
        assert (dx>0 and dy>0),"Must be 1d"
        assert x.shape[-1] == y.shape[axis_y:=-min(len(y.shape),2)],"Cannot matmul shapes."
        """equivalen to 
        if w.ndim == 1:
            axis_w = -1
        else:
            axis_w = -2"""
        x = x.reshape(*x.shape[0:-1],*[1]*min(dx-1,dy-1,1),x.shape[-1])
        y = y.reshape(*y.shape[0:-2],*[1]*min(dx-1,dy-1,1),*y.shape[axis_y:]).transpose(-1,axis_y)
        return (x*y).sum(axis=-1)
    # TODO pool uses reshape and max,mean
    # TODO pad for backward
    def conv2d(self,w,bias=None,**kwargs):
        # im2col, sliding window.
        # TODO add bias.
        return self._conv2d.apply(self,w,**kwargs)

# act as the context
class Function:
    def __init__(self,device:str,*tensors:Tensor):
        self.device = device
        self.parents: tuple[Tensor] = tensors
        self.saved_tensors : list[LazyBuffer]

        self.need_input_grad = [t.requires_grad for t in self.parents]
        self.requires_grad = True if any(self.need_input_grad) else False
    
    def save_for_backward(self,*x):
        self.saved_tensors.extend(x)

    def forward(self,*x): raise NotImplementedError
    def backward(self,*x): raise NotImplementedError

    @classmethod
    def apply(cls, *x:Tensor, **kwargs) -> Tensor:
        # 1. Create a context (ctx) for the operation
        ctx = cls(x[0].device, *x)
        # 2. Execute the forward pass (compute  lazydata)
        ret_data = ctx.forward(*[t.lazydata for t in x], **kwargs)
        # 3. Wrap the result in a new Tensor
        ret = Tensor(data=ret_data, device=ctx.device, requires_grad=ctx.requires_grad)
        # 4. Store the context for backpropagation (if needed)
        if ctx.requires_grad:
            ret._ctx = ctx
        return ret

# register all math "ops" from mlops
def register(name:str,fxn:Function):
    setattr(Tensor,"_"+name if hasattr(Tensor, name) else name,fxn)
for name,cls in inspect.getmembers(importlib.import_module("minigrad.mlops"),inspect.isclass):
    if name!="Function" and name!="LazyBuffer" and name[0]!="_" and not name.endswith("Ops"): 
        register(name.lower(),cls)
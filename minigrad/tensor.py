from __future__ import annotations
import numpy as np
from minigrad.lazy import LazyBuffer
from typing import Optional, Literal, Sequence, Callable, List
import functools, inspect, importlib, itertools
import math


# other is np.array while doing the broad cast [other] so, this add extra 1 dim
class Tensor:
    training: bool
    def __init__(self,data,device=None,requires_grad=None):
        if isinstance(data,list):
            data = np.array(data,dtype=np.float32)
        elif isinstance(data, LazyBuffer) and data.device != device:
            # lazy buffer realized on cpu, 
            data = data.realize().toCPU()

        if isinstance(data, np.ndarray):
            # if data.shape == tuple(): # for scalar value, reshape to dim 1
            #     data = data.reshape((1,))
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
        return self.numpy()
    
    def to(self,device:str):
        ret = Tensor(self.lazydata,device)
        if ret.grad:
            ret.grad = self.grad.to(device)
        return ret
    
    def to_(self,device:str):
        assert self.lazydata.realized is None
        self.lazydata.device = device
        if self.grad:
            self.grad.lazydata.device = device
    
    def size(self,x=None):
        if x is None:
            return math.prod(self.shape)
        return self.shape[x]
    
    @property
    def device(self):
        return self.lazydata.device
    
    def detach(self): return Tensor(self.lazydata,device=self.device,requires_grad=False)
    def numpy(self): return np.array(self.lazydata.toCPU())
    def requires_grad_(self,value: bool): self.requires_grad=value; return self

    @classmethod
    def rand(cls,shape: tuple):
        assert type(shape)==tuple,""
        return cls(np.random.rand(*shape))
    
    # use numpy for now
    @classmethod
    def zeros_like(cls,tensor,**kwargs): return cls.zeros(*tensor.shape,**kwargs)

    @classmethod
    def zeros(cls,*shape,**kwargs): return cls(np.zeros(shape,dtype=np.float32),**kwargs)

    @classmethod
    def ones(cls,*shape,**kwargs): return cls(np.ones(shape,dtype=np.float32),**kwargs)

    @classmethod
    def randn(cls,*shape,**kwargs): return cls(np.random.default_rng().standard_normal(size=shape,dtype=np.float32),**kwargs)

    @classmethod
    def empty(cls, *shape, **kwargs): return cls(np.empty(shape, dtype=np.float32), **kwargs)

    @classmethod
    def arange(cls,stop,start=0,**kwargs): return cls(np.arange(stop=stop,start=start,dtype=np.float32),**kwargs)

    @classmethod
    def xavier_uniform(cls,*shape,gain=1.0,**kwargs): bound = gain*(6/(shape[0]+math.prod(shape[1:])))**0.5 ;return cls(np.random.uniform(-bound,bound,size=shape))
    
    @property
    def shape(self): return self.lazydata.shape

    @property
    def T(self): return self.transpose()
    
    def realize(self):
        self.lazydata.realize()
        return self
    
    def assign(self,x):
        if not isinstance(x,Tensor):
            x = Tensor(x)
        assert self.shape == x.shape
        self.lazydata = x.lazydata
        return x
    
    def reshape(self,*shape) :
        known_dims = [s for s in shape if s!=-1]
        inferred_dim = self.size()//math.prod(known_dims)
        shape = tuple(inferred_dim if dim == -1 else dim for dim in shape)
        assert self.size() == math.prod(shape),f"Cannot reshape {self.shape} -> {shape}"
        return self._reshape.apply(self,shape=shape)
    
    def expand(self,*shape):
        assert len(shape)==self.ndim,f"Cannot broradcast{self.shape} -> {shape}"
        assert (s==e for e,s in zip(self.shape,shape) if s!=1 ),f"Cannot broradcast{self.shape} -> {shape}"
        return self._expand.apply(self,shape=shape)

    @property
    def ndim(self) -> int: return len(self.shape)
    
    def flatten(self,start_dim=0,end_dim=-1):
        # resolve dim, index out of range
        start_dim, end_dim = self._resolve_dim(start_dim), self._resolve_dim(end_dim)
        flatten_shape = tuple(self.shape[:start_dim]+(math.prod(self.shape[start_dim:end_dim+1]),)+self.shape[end_dim+1:])
        return self.reshape(*flatten_shape)

    # uses reshape
    def squeeze(self,dim=None):
        if dim is None:
            return self.reshape(*tuple(dim for dim in self.shape if dim!=1))
        dim = self._resolve_dim(dim)
        return self if not self.ndim or self.shape[dim] != 1 else self.reshape(*tuple(self.shape[:dim] + self.shape[dim+1:]))
    
    # uses reshape
    def unsqueeze(self,dim):
        dim = self._resolve_dim(dim)
        unsq_shape = tuple(self.shape[:dim]+(1,)+self.shape[dim:])
        return self.reshape(*unsq_shape)
    
    # TODO contiguous, 
    # view is alias for reshape
    def view(self,*shape):
        #TODO to take axis, and change to shape
        return self.reshape(*shape)

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
        assert self.shape == (1,) or self.shape == (), "Backward can only be called on scalar tensors."
        self.grad = Tensor([1] if self.shape == (1,) else np.array(1.0),requires_grad=False,device=self.device)  # Initial gradient (dy/dy = 1)

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
            # for graph visual; comment it
            # del node._ctx
    
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
        # slice based on slice indices and reshape; need reshape (slice always preserves rank.) to drop 1 dim
        return self.slice.apply(self,arg=arg).reshape(*new_shape if len(new_shape) else ())
    
    def cat(self, *args, dim=0):
        # make positive dim
        dim = (dim + len(self.shape)) if dim < 0 else dim
        # check the shape for concat,
        for y in args:
            assert len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != dim)
        # make a list of Tensors to concat; self itself too.
        args = [self] + list(args)
        shape_cumsum = [0, *itertools.accumulate(y.shape[dim] for y in args)]
        slc = [[(0, s) for s in self.shape] for _ in args]
        for s,k in zip(slc, shape_cumsum):
            s[dim] = (-k, shape_cumsum[-1]-k)
        return functools.reduce(Tensor.__iadd__, [arg.slice.apply(arg,arg=s) for arg,s in zip(args, slc)])
    
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

    def mul(self,x): return Tensor.broadcasted(Tensor._mul,self,x)
    def add(self,x): return Tensor.broadcasted(Tensor._add,self,x)
    def sub(self,x): return Tensor.broadcasted(Tensor._sub,self,x)
    def pow(self,x): return Tensor.broadcasted(Tensor._pow,self,x)
    # for  number y, it get 1/number. then broadcast the value;
    def div(self,y): return self * (y.reciprocal() if isinstance(y,Tensor) else (1/y))

    # can be sub function.
    def __neg__(self): return 0.0-self
    
    # TODO reduce op function, handle int, negative indexing (-1), and calculate shape
    def _reduce(self,fxn, axis=None, keepdim=False):
        return fxn.apply(self,axis=axis,keepdim=keepdim)

    def sum(self,axis=None,keepdim=False): return self._reduce(Tensor._sum,axis=axis,keepdim=keepdim)
    def max(self,axis=None,keepdim=False): return self._reduce(Tensor._max,axis=axis,keepdim=keepdim)
    def min(self,axis=None,keepdim=False): return -((-self).max(axis,keepdim))

    def mean(self,axis=None,keepdim=False):
        out = self.sum(axis=axis,keepdim=keepdim)
        return out * (math.prod(out.shape)/math.prod(self.shape))
    
    def var(self,axis=None,keepdim=False):
        mean = self.mean(axis=axis,keepdim=keepdim)
        diff = (self-mean)**2
        return diff.mean(axis=axis,keepdim=keepdim)
    
    def std(self,axis=None,keepdim=False):
        var = self.var(axis=axis,keepdim=keepdim)
        return var.sqrt()
    
    @staticmethod
    def triu(*shape): return Tensor(np.tril(np.ones(shape),k=0),device=None,requires_grad=False)
    # Fills elements of self tensor with value where mask is True. (pytorch.org)
    def masked_fill_(self,mask,value):
        mask = np.broadcast_to(mask, self.shape)
        assert mask.shape==self.shape
        return self.masked_fill.apply(self,mask=mask,value=value)
    
    # Fills self tensor with the specified value.
    def fill_(self,value): return None
    def __eq__(self,other): 
        assert type(other)==int 
        return Tensor(self.data == other,device=None,requires_grad=False)
    def __hash__(self): return id(self)

    def matmul(self: Tensor,other: Tensor):
        # broad cast, multiply and sum over axis.
        x,y,dx,dy = self,other,len(self.shape),len(other.shape)
        assert (dx>0 and dy>0),"Must be 1d"
        assert x.shape[-1] == y.shape[axis_y:=-min(len(y.shape),2)],"Cannot matmul shapes."
        """equivalen to if w.ndim == 1:axis_w = -1 else:axis_w = -2"""
        x = x.reshape(*x.shape[0:-1],*[1]*min(dx-1,dy-1,1),x.shape[-1])
        y = y.reshape(*y.shape[0:-2],*[1]*min(dx-1,dy-1,1),*y.shape[axis_y:]).transpose(-1,axis_y)
        return (x*y).sum(axis=-1)

    dot = matmul

    # if training only else self; use mask*self * 1/(1-p)
    def dropout(self,p=0.5): 
        if not Tensor.training:
            return self
        mask = np.random.binomial(1,p=1.0-p,size=self.shape)
        return self*Tensor(mask,device=self.device,requires_grad=False) * (1/(1.0-p))
    
    # TODO support arbitrary strides
    def _pool2d(self,py,px):
        assert self.ndim==4
        # trim if needed
        xt = self[:,:,:self.shape[2]-self.shape[2]%py,:self.shape[3]-self.shape[3]%px] if (self.shape[2]%py!=0) or (self.shape[3]%px!=0) else self
        # grouping based on kernal, (1,1,4,4) -> (1,1,2,2,2,2)
        return xt.reshape(xt.shape[0],xt.shape[1],xt.shape[2]//py,py,xt.shape[3]//px,px)
    
    def avg_pool2d(self,kernel_size=(2,2)): return self._pool2d(*kernel_size).mean(axis=(3,5))
    def max_pool2d(self,kernel_size=(2,2)): return self._pool2d(*kernel_size).max(axis=(3,5))

    def conv2d(self,w,bias=None,**kwargs):
        # im2col, sliding window.
        # TODO add bias, reshape bias to [1,-1,1,1].
        return self._conv2d.apply(self,w,**kwargs) if bias is None else \
              self._conv2d.apply(self,w,**kwargs).add(bias.reshape(*[1,-1,1,1]))
    def logsoftmax(self,dim=-1):
        m = self - self.max(axis=dim,keepdim=True)
        _exp = m.exp()
        return m - (_exp.sum(axis=dim,keepdim=True)).log()
    
    def softmax(self,axis=-1):
        # normalize, self- max of self, to solve overflow
        m = self - self.max(axis=axis,keepdim=True)
        _exp = m.exp()
        sm = _exp.div(_exp.sum(axis=axis,keepdim=True))
        return sm

    def linear(self,weight,bias=None):
        x = self.mul(weight) if len(weight.shape)==1 else self.matmul(weight)
        return x.add(bias) if bias is not None else x
    
    def layer_norm(self,axis=-1,eps=1e-5):
        y = self - self.mean(axis=axis,keepdim=True)
        norm = y.div((y*y).mean(axis=axis,keepdim=True).add(eps).sqrt())
        return norm
    
    def sequential(self,ll:List[Callable[[Tensor],Tensor]]): return functools.reduce(lambda x,f: f(x),ll,self)

    
    # math unary functions
    def reciprocal(self): return self._reciprocal.apply(self)
    def sqrt(self): return self.pow(0.5)
    def sign(self): return self / (self.abs() + 1e-10)  
    def abs(self): return self.relu() + (-self).relu()
    def square(self): return self*self
    def exp(self): return self._exp.apply(self)
    def log(self): return self._log.apply(self)

    # activation unary functions
    def relu(self): return self._relu.apply(self)
    def sigmoid(self): return (1.0+(-self).exp()).reciprocal()
    def tanh(self): return 2.0 * ((2.0*self).sigmoid())-1.0
    def gelu(self): return 0.5 * self * (1 + (self * 0.7978845608 * (1+0.044715 * self * self)).tanh())

# act as the context
class Function:
    def __init__(self,device:str,*tensors:Tensor):
        self.device = device
        self.parents: tuple[Tensor] = tensors
        self.saved_tensors : list[LazyBuffer] = []

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
def register(name:str,fxn_class:Function):
    setattr(Tensor,"_"+name if hasattr(Tensor, name) else name,fxn_class)
for name,cls in inspect.getmembers(importlib.import_module("minigrad.mlops"),inspect.isclass):
    if name!="Function" and name!="LazyBuffer" and name[0]!="_" and not name.endswith("Ops"): 
        register(name.lower(),cls)

# register the operators
def register_op(name, fxn):
    setattr(Tensor, f"__{name}__", fxn)
    # in place op; +=
    setattr(Tensor, f"__i{name}__", lambda self,x: self.assign(fxn(self,x)))
    # right hand op, 1 + Tensor
    setattr(Tensor, f"__r{name}__", lambda self,x: fxn(x,self))
for name in ['add', 'sub', 'mul', 'pow', 'matmul', 'truediv']:
  register_op(name, getattr(Tensor, name if name != 'truediv' else 'div'))
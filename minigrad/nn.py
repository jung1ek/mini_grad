from __future__ import annotations # let us use, runtime annotations like Module

from typing import Sequence,Tuple, Dict,Any,Optional
from minigrad.tensor import Tensor

class Parameter:
    
    def __init__(self,x: Tensor, name: Optional[str]=None):
        self.value = x
        self.name = name
        if hasattr(x,"requires_grad_"):
            self.value.requires_grad_(True)
    
    def update(self,x: Tensor) -> None:
        self.value = x
        if hasattr(x,"requires_grad_"):
            self.value.requires_grad_(True)

    def __repr__(self)-> str:
        return repr(self.value)

    def __str__(self):
        return str(self.value)
    

class Module:

    _modules: Dict[str, Module]
    _parameters = Dict[str, Parameter]
    training : bool

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True
        
    def register_buffer(self,name,attr):
        setattr(Module, name, attr)
    
    def modules(self) -> Sequence[Module]:
        m: Dict[str,Module] = self.__dict__["_modules"]
        return list(m.values())

    def eval(self) -> None:
        self.training = False
        # recursive tree traverse
        for module in self.modules(): # traverse through child modules
            module.eval()

    def train(self) -> None:
        self.training = True
        for child_module in self.modules():
            child_module.train()

    def named_parameters(self) -> Sequence[Tuple[str,Parameter]]:
        pass

    def parameters(self) -> Sequence[Parameter]:
        result = []
        # add parameters from this module
        result.extend(self.__dict__["_parameters"].values())
        # add parameters from child modules recursively; extend the list from the result of child moudle
        for moudule in self.modules():
            result.extend(moudule.parameters()) # like concat two lists
        return result

    def add_parameter(self,k:str,v:Any) -> Parameter:
        val = Parameter(v,k)
        self.__dict__["_parameters"][k]=val
        return val
    
    def add_module(self,k:str,m:Any) -> Module:
        self.__dict__["_modules"][k] = m
    
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.add_parameter(name, value.value)
            self.register_buffer(name,value)
        elif isinstance(value, Module):
            self.add_module(name,value)
            self.register_buffer(name,value)
        elif isinstance(value, Tensor):
            self.register_buffer(name,value)
        else:
            super().__setattr__(name, value)
    
    def forward(self, *args, **kwds):
        raise NotImplementedError
    
    def __call__(self,*args,**kwds):
        return self.forward(*args,**kwds)

class ModuleList:
    
    def __init__(self,modules: Sequence[Module]):
        self.modules = modules
    
    def __getitem__(self,index):
        return self.modules[index]


class Embedding(Module):
    def __init__(self,vocab,d_model):
        super().__init__()
        self.embedding = Parameter(Tensor.randn(vocab,d_model))

    def forward(self,x: Tensor):
        assert len(x.shape) == 2, f"Expect batch and seq_len"
        batch,seq_len = x.shape
        batch_out = None  # will become (batch, seq_len, d_model)

        for b in range(batch):
            seq_out = None  # will become (seq_len, d_model)
            for s in range(seq_len):
                idx = int(x[b][s].data.item())
                vec = self.embedding.value[idx].unsqueeze(0)
                # vec shape: (1, d_model)
                if seq_out is None:
                    seq_out = vec
                else:
                    seq_out = seq_out.cat(vec, dim=0)
            # seq_out shape: (seq_len, d_model)
            seq_out = seq_out.unsqueeze(0)  # (1, seq_len, d_model)
            if batch_out is None:
                batch_out = seq_out
            else:
                batch_out = batch_out.cat(seq_out, dim=0)
        return batch_out



class Linear(Module):
    def __init__(self,in_features,out_featues):
        super().__init__()
        self.w = Parameter(Tensor.xavier_uniform(in_features,out_featues))
        self.b = Parameter(Tensor.randn(out_featues))

    def forward(self,x):
        return x @ self.w.value + self.b.value
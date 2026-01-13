from __future__ import annotations # let us use, runtime annotations like Module

from typing import Sequence,Tuple, Dict,Any,Optional


class Parameter:
    
    def __init__(self,x: Any, name: Optional[str]=None):
        self.value = x
        self.name = name
        if hasattr(x,"requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name
    
    def update(self,x:Any) -> None:
        self.value = x
        if hasattr(x,"requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

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

class ModuleList:
    
    def __init__(self,modules: Sequence[Module]):
        self.modules = modules
    
    def __getitem__(self,index):
        return self.modules[index]



if __name__=="__main__":
    class A:

        def __init__(self):
            self._module = {"a":1,"b":2}
            res = []
            value = self.__dict__.get("_module",{}).items()
            print(self.__dict__["_module"].items())
            print(res.extend(list(value)))
            print(res)
    A()
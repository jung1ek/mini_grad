import os
import torch
import time
import numpy as np
import unittest
from minigrad.tensor import Tensor
import tqdm

FORWARD_ONLY = bool(int(os.getenv("FORWARD_ONLY","1")))

def test_helper(shapes,my_fxn,torch_fxn,atol=1e-4,rtol=1e-3,a=1,b=0.001,grad_atol=1.e-7,grad_rtol=1e-7,forward_only=False):
    def compare_(ret,out,atol,rtol):
        try:
            assert ret.shape == out.shape
            np.testing.assert_allclose(ret,out,atol=atol,rtol=rtol)
        except Exception as e:
            raise ValueError(e)
        
    np.random.seed(177)
    nt = [(np.random.randn(*s)*a+b).astype(np.float32) for s in shapes]
    tt = [torch.tensor(n,requires_grad=True).to("cuda") for n in nt]
    mt = [Tensor(t.to("cpu").detach().numpy(),requires_grad=True).to("gpu") for t in tt]

    out = torch_fxn(*tt)
    ret = my_fxn(*mt)
    compare_(ret.numpy(),out.detach().to("cpu").numpy(),atol=1e-4,rtol=1e-3)

    if not FORWARD_ONLY and not forward_only:
        out.square().mean().backward()
        ret.square().mean().backward()
        for gmt, gtt in zip(mt,tt):
            compare_(gmt.numpy(),gtt.detach().to("cpu").numpy(),grad_atol,grad_rtol)


class Test(unittest.TestCase):

    def test_add(self):
        test_helper([(45,65),(45,65)],Tensor.add,lambda x,y: x+y)
        test_helper([(65,65),(65,65,65)],Tensor.add,lambda x,y: x+y)
    def test_multi_bin(self):
        test_helper([(65,65),(65,65,65),(65,65),(65,65,65)],lambda x,y,a,b: Tensor.add(x,y)+a+(a*b),lambda x,y,a,b: x+y+a+(a*b))
    def test_sub(self):
        test_helper([(45,65),(45,65)],Tensor.sub,lambda x,y: x-y)
        test_helper([(45,45,65),(45,65)],Tensor.sub,lambda x,y: x-y)
    def test_mul(self):
        test_helper([(45,65),(45,65)],Tensor.mul,lambda x,y: x*y)
        test_helper([(45,45,65),(45,65)],Tensor.mul,lambda x,y: x*y)
        test_helper([(45,45,65),(45,65)],lambda x,y:Tensor.mul(x,y)*y*y,lambda x,y: x*y*y*y)
    def test_div(self):
        test_helper([(45,65),(45,65)],Tensor.div,lambda x,y: x/y)
        test_helper([(45,65),(45,65)],lambda x,y: Tensor.div(x,y)/y/x,lambda x,y: x/y/y/x)
    def test_pow(self):
        test_helper([(45,65),(45,65)],Tensor.pow,lambda x,y: x**y)
    def test_sum(self):
        test_helper([(45,45,65)],lambda x: Tensor.sum(x,axis=-1),lambda x: x.sum(axis=-1))
        test_helper([(45,45,65)],lambda x: Tensor.sum(x,axis=(0,1)),lambda x: x.sum(axis=(0,1)))
        test_helper([(45,45,65)],lambda x: Tensor.sum(x,axis=(0,1,2)),lambda x: x.sum(axis=(0,1,2)))
    
    def test_broadcast_sum(self):
        test_helper([(45,45,65),(45,45,65)],lambda x,y: (x+y).sum(axis=-1),lambda x,y: (x+y).sum(axis=-1))
        test_helper([(45,45,65),(45,65)],lambda x,y: (x+y).sum(axis=-1),lambda x,y: (x+y).sum(axis=-1))
        test_helper([(45,45,65),(45,65)],lambda x,y: (x+y-y).sum(axis=-1),lambda x,y: (x+y-y).sum(axis=-1))
        test_helper([(45,45,65),(45,65),(65,)],lambda x,y,z: (x+y-z).sum(axis=-1),lambda x,y,z: (x+y-z).sum(axis=-1))
        test_helper([(45,45,65),(45,65),(45,45,65)],lambda x,y,z: (x+y*z).sum(axis=-1),lambda x,y,z: (x+y*z).sum(axis=-1))
        test_helper([(45,45,65),(45,65),(45,45,65)],lambda x,y,z: (x+y*z/x).sum(axis=-1),lambda x,y,z: (x+y*z/x).sum(axis=-1))

    def test_dot(self):
        test_helper([(45,65), (65,100)], Tensor.dot, lambda x,y: x.matmul(y), atol=1e-4)
    def test_matmul_simple(self):
        test_helper([(2,), (2,2)], Tensor.dot, lambda x,y: x.matmul(y), atol=1e-4)
    def test_matmul(self):
        test_helper([(65,), (65,99)], Tensor.dot, lambda x,y: x.matmul(y), atol=1e-4)
    def test_gemm(self):
        test_helper([(256,256), (256,256)] ,Tensor.dot, lambda x,y: x.matmul(y), atol=1e-3)

    def test_relu(self):
        test_helper([(45,45,65)],lambda x: Tensor.relu(x),lambda x: x.relu())
    def test_sign(self):
        test_helper([(45,45,65)],lambda x: Tensor.sign(x),lambda x: x.sign())
    def test_exp(self):
        test_helper([(45,45,65)],lambda x: Tensor.exp(x),lambda x: x.exp())
    def test_log(self):
        test_helper([(45,45,65)],lambda x: Tensor.log(x),lambda x: x.log())
    def test_neg(self):
        test_helper([(45,45,65)],lambda x: Tensor.__neg__(x),lambda x: -x)
    def test_sigmoid(self):
        test_helper([(45,65)],Tensor.sigmoid, lambda x: x.sigmoid())
    def test_tanh(self):
        test_helper([(45,65)], Tensor.tanh,lambda x: x.tanh(),  atol=1e-6, grad_atol=1e-6)
    def test_unary_sum(self):
        test_helper([(45,45,65),(45,45,65)],lambda x,y: (x+y).relu().sum(axis=-1),lambda x,y: (x+y).relu().sum(axis=-1))
        test_helper([(45,45,65),(45,65)],lambda x,y: (x+y).sum(axis=-1),lambda x,y: (x+y).sum(axis=-1))
        test_helper([(45,45,65),(45,65)],lambda x,y: (x+y-(y.log())).sum(axis=-1),lambda x,y: (x+y-(y.log())).sum(axis=-1))
        test_helper([(45,45,65),(45,65),(65,)],lambda x,y,z: (x+y-z).sum(axis=-1),lambda x,y,z: (x+y-z).sum(axis=-1))
        test_helper([(45,45,65),(45,65),(45,45,65)],lambda x,y,z: (x+y*z.exp()).sum(axis=-1),lambda x,y,z: (x+y*z.exp()).sum(axis=-1))
        test_helper([(45,45,65),(45,65),(45,45,65)],lambda x,y,z: (x+y*z/x.sign()).sum(axis=(0,1)),lambda x,y,z: (x+y*z/x.sign()).sum(axis=(0,1)))

    def test_multi_reduce(self):
        test_helper([(45,45,65),(45,45,65)],lambda x,y: (x+y).relu().sum(axis=-1)+x.sum(axis=-1),lambda x,y: (x+y).relu().sum(axis=-1)+x.sum(axis=-1))
        test_helper([(45,45,65),(45,65)],lambda x,y: (x+y).sum(axis=-1).sum(axis=-1),lambda x,y: (x+y).sum(axis=-1).sum(axis=-1))
        test_helper([(45,45,65),(45,65)],lambda x,y: (x+y-(y.log())).sum(axis=-1)+(x*y).sum(-1),lambda x,y: (x+y-(y.log())).sum(axis=-1)+(x*y).sum(axis=-1))
        test_helper([(45,45,65),(45,65),(45,65)],lambda x,y,z: (x+y).max()*z.sum(axis=-1),lambda x,y,z: (x+y).max()*z.sum(axis=-1))
        test_helper([(45,45,65),(45,65),(45,45,65)],lambda x,y,z: (x+y*z.exp()).sum(axis=-1).sum(),lambda x,y,z: (x+y*z.exp()).sum(axis=-1).sum())
        test_helper([(45,45,65),(45,65),(45,45,65)],lambda x,y,z: (x+y*z/x.sign()).sum(axis=(0,1)),lambda x,y,z: (x+y*z/x.sign()).sum(axis=(0,1)))

    def test_matmul_sum(self):
        test_helper([(45,65), (65,100),(100,)],lambda x,y,a: Tensor.dot(x,y)+a, lambda x,y,a: x.matmul(y)+a, atol=1e-4)

    def test_mean(self):
        test_helper([(45,45,65)],lambda x: Tensor.mean(x),lambda x: x.mean().unsqueeze(0))

    def test_transpose(self):
        test_helper([(3,3,3)],  lambda x: x.transpose(1,2),lambda x: x.transpose(1,2))
        test_helper([(3,3,3)], lambda x: x.transpose(0,2), lambda x: x.transpose(0,2))
        test_helper([(1,2,3,4)], lambda x: x.permute(order=(3,0,2,1)), lambda x: x.movedim((3,0,2,1),(0,1,2,3)),)
        test_helper([(3,4,5,6)], lambda x: x.permute(order=(3,2,1,0)), lambda x: x.movedim((3,2,1,0),(0,1,2,3)),)

    def test_reshape(self):
        test_helper([(4,3,6,6)], lambda x: x.reshape(*(-1,3,6,6)), lambda x: torch.reshape(x, (-1,3,6,6)))
        test_helper([(4,3,6,6)],  lambda x: x.reshape(*(-1,1,6,6)),lambda x: torch.reshape(x, (-1,1,6,6)),)
    def test_expand(self):
        arg = (4,3,2,6)
        test_helper([(4,3,1,6)], lambda x: x.expand(*arg), lambda x: x.expand(arg),)
    def test_flatten(self):
        for axis in range(3):
            test_helper([(4,3,6,6)],  lambda x: x.flatten(start_dim=axis),lambda x: torch.flatten(x, start_dim=axis),atol=1e-7,grad_atol=1e-7)
    # def test_logsoftmax(self):
    #     test_helper([(45,65)], Tensor.logsoftmax, lambda x: torch.nn.LogSoftmax(dim=1)(x),  atol=0.05, rtol=0.25,grad_atol=0.104, grad_rtol=0)
    
    # def test_softmax(self):
    #     test_helper([(45,65), (65,100),(100,)],lambda x,y,a: Tensor.softmax(x), lambda x,y,a: x.softmax(dim=-1),  atol=0.104, rtol=0.32,grad_atol=0.104, grad_rtol=0)

    def test_gelu(self):
        test_helper([(45,65)],  Tensor.gelu,lambda x: 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))))
       
if __name__=="__main__":
    unittest.main(verbosity=2)
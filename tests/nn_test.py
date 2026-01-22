import numpy as np
import torch
from minigrad import nn
import unittest
from minigrad.tensor import Tensor

x = np.random.randn()
w = np.random.randn()
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
b1 = np.random.randn()
b2 = np.random.randn()

def compare(x,y,atol=1e-4,rtol=1e-3):
    try:
        np.testing.assert_allclose(x,y,atol=atol,rtol=rtol)
    except Exception as e:
        raise Exception(f"Failed {e}")

class MinigradNet():
    pass

class TorchNet():
    pass

def embedding_test(input_idxs,vocab=100,d_model=512,atol=1e-4,rtol=1e-5):
    torch.manual_seed(1010)
    np.random.seed(1010)
    ne = np.random.randn(vocab,d_model)
    te = torch.tensor(ne)
    me = Tensor(ne)
    ee = nn.Embedding(vocab,d_model)
    ee.embedding = nn.Parameter(me)
    ret = ee(Tensor(input_idxs))
    out = te[torch.tensor(input_idxs)]
    compare(ret.numpy(),out.detach().numpy())


class TestNN(unittest.TestCase):

    def test_forward(self):
        embedding_test([[1,2,3,110],[5,6,7,99]])


if __name__=="__main__":
    np.random.seed(101)
    unittest.main(verbosity=2)

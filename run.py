from enum import Enum
from minigrad import tensor
import numpy as np

a = tensor.Tensor([1,2,3],requires_grad=True)
b = tensor.Tensor([1,2,3])
# c = a*b
z =a+b
# y = z*z
# y.backward()
print(z.realize())
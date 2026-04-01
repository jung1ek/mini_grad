from minigrad import tensor
import numpy as np
from minigrad import nn
from minigrad.tensor import Tensor
from typing import Callable
import math
from copy import deepcopy

def clones(module,N):
    return nn.ModuleList([module for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super().__init__()
        self.alpha = nn.Parameter(Tensor.ones(d_model))
        self.beta = nn.Parameter(Tensor.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.alpha.value * (x-mean)/(std+self.eps) + self.beta.value

def attention(query,key,value,mask=None,dropout=0.1):
    dk = query.size(-1)
    scores = query.matmul(key.transpose(-2,-1)) / math.sqrt(dk)

    if mask is not None:
        scores = scores.masked_fill_(mask,1e-9)
    attn = scores.softmax(-1)
    if dropout is not None:
        attn = attn.dropout(dropout)
    return attn.matmul(value), attn
     
class MultiHeadAttention(nn.Module):

    def __init__(self,d_model,heads,dropout=0.1):
        super().__init__()
        self.l1,self.l2,self.l3,self.l4 = [nn.Linear(d_model,d_model) for _ in range(4)]
        self.d_model = d_model
        self.heads = heads
        self.dk = d_model//heads
        self.dropout = dropout

        self.attn = None

    def forward(self,q,k,v,mask=None):
        n_batches = q.shape[0]
        q_porj,k_proj,v_proj = [lin(x).view(n_batches,-1,self.heads,self.dk).transpose(1,2) for lin,x in zip((self.l1,self.l2,self.l3),(q,k,v))]
        x, self.attn = attention(
            q_porj, k_proj, v_proj, mask=mask, dropout=self.dropout
        )
        x = (
            x.transpose(1, 2)
            .view(n_batches, -1, self.heads * self.dk)
        )
        return self.l4(x)


class PositionalFFN(nn.Module):

    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(d_model,d_ff)
        self.l2 = nn.Linear(d_ff,d_model)
        self.dropout = dropout
    
    def forward(self,x):
        return self.l2(self.l1(x).relu().dropout(self.dropout))

class SublayerConnection(nn.Module):

    def __init__(self,d_model,dropout=0.1):
        super().__init__()
        self.ln = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self,x: Tensor, sublayer: Callable):
        return x + sublayer(self.ln(x.dropout(self.dropout)))

class EncoderLayer(nn.Module):

    def __init__(self,d_model,attn,ffn,droput=0.1):
        super().__init__()
        self.ffn = ffn
        self.self_attn = attn
        self.sub1,self.sub2 = [SublayerConnection(d_model,droput) for _ in range(2)]
    
    def forward(self,x,mask=None):
        x = self.sub1(x,lambda x: self.self_attn(x,x,x,mask))
        return self.sub2(x,self.ffn)

class DecoderLayer(nn.Module):

    def __init__(self,d_model,self_attn,src_attn,ffn,dropout=0.1):
        super().__init__()
        self.src_attn = src_attn
        self.self_attn = self_attn
        self.ffn = ffn
        self.sl1,self.sl2,self.sl3 = [SublayerConnection(d_model,dropout) for _ in range(3)]

    def forward(self,x,m,src_mask=None,tgt_mask=None):
        x = self.sl1(x,lambda x: self.self_attn(x,x,x,tgt_mask))
        x = self.sl2(x,lambda x: self.src_attn(x,m,m,src_mask))
        return self.sl3(x,self.ffn)

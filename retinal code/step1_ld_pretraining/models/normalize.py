import torch
from torch.autograd import Variable
from torch import nn

def l2_normal(x,dim=1):
    norm = x.pow(2).sum(dim, keepdim=True).pow(1./2)+1e-10
    out = x.div(norm)
    return out
            
def l1_normal(x,dim=1):
    x=x+10e-10
    norm = x.sum(dim, keepdim=True)
    out = x.div(norm)
    return out  
        

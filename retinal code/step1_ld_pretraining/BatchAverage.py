import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np

class BatchCriterion(nn.Module):
    ''' 
        Compute the loss within each batch  
        This code refers to https://github.com/mangye16/Unsupervised_Embedding_Learning
    '''
    def __init__(self, negM, T):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        
    def forward(self, x):
        
        
        batchSize = x.size(0)
        diag_mat=1 - torch.eye(batchSize).cuda()
        
        #get positive innerproduct
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2),x.narrow(0,0,batchSize//2)), 0)
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_()
        del reordered_x

        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*diag_mat
        del x
        del diag_mat
        if self.negM==1:
            all_div = all_prob.sum(1)
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos

        lnPmt = torch.div(pos, all_div+1e-10)
        del pos
        # negative probability
        Pon_div = all_div.repeat(batchSize,1)
        del all_div
        lnPon = torch.div(all_prob, Pon_div.t()+1e-10)
        del all_prob,Pon_div
        lnPon = -lnPon.add(-1)
        
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum)/batchSize
        return loss

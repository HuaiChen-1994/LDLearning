# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:11:41 2020

@author: Administrator
"""
import numpy as np
from .normalize import l2_normal,l1_normal
import torch.nn as nn
import torch
import torch.nn.functional as F

class decoder(nn.Module):
    def __init__(self,channel_list=[128,128,64,32,16,32,8]):
        super(decoder, self).__init__()
        self.upsample=nn.Upsample(scale_factor=(2,2))
        
        
        self.conv_layer8x=nn.Sequential(
                                              nn.Conv2d(in_channels=channel_list[0]+channel_list[1],out_channels=channel_list[1],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[1]),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels=channel_list[1],out_channels=channel_list[2],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[2]),
                                              nn.ReLU(True))

        self.conv_layer4x=nn.Sequential(
                                              nn.Conv2d(in_channels=channel_list[2]*2,out_channels=channel_list[2],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[2]),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels=channel_list[2],out_channels=channel_list[3],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[3]),
                                              nn.ReLU(True)
                                                )
        self.conv_layer2x=nn.Sequential(
                                              nn.Conv2d(in_channels=channel_list[3]*2,out_channels=channel_list[3],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[3]),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels=channel_list[3],out_channels=channel_list[4],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[4]),
                                              nn.ReLU(True)
                                                )
        self.embedding_branch=nn.Sequential(
                                              nn.Conv2d(in_channels=channel_list[4]*2,out_channels=channel_list[5],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[5]),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels=channel_list[5],out_channels=channel_list[5],kernel_size=(3,3),padding=(1,1))
                                                )
        if len(channel_list)>6:
            self.clustering_branch=nn.Sequential(
                                                  nn.Conv2d(in_channels=channel_list[4]*2,out_channels=channel_list[6],kernel_size=(3,3),padding=(1,1)),
                                                  nn.BatchNorm2d(channel_list[6]),
                                                  nn.ReLU(True),
                                                  nn.Conv2d(in_channels=channel_list[6],out_channels=channel_list[6],kernel_size=(3,3),padding=(1,1)),
                                                  nn.Sigmoid()
                                                )
    def forward(self, x,region_output_size=(4,4),if_clustering=False,if_test=False):
        features=self.upsample(x[-1])
        features=torch.cat([features,x[-2]],dim=1)
        features=self.conv_layer8x(features)
        features=self.upsample(features)
        features=torch.cat([features,x[-3]],dim=1)
        features=self.conv_layer4x(features)
        features=self.upsample(features)
        features=torch.cat([features,x[-4]],dim=1)
        features=self.conv_layer2x(features)
        features=self.upsample(features)
        features=torch.cat([features,x[-5]],dim=1)
        del x
        if not if_clustering:
            features=self.embedding_branch(features)
            features=l2_normal(features)
            return features
        
        attention_map=self.clustering_branch(features)
        attention_map=l1_normal(attention_map, dim=1)
        features=self.embedding_branch(features)
        
        features=l2_normal(features)
        # return attention_map,features
        if if_test:
            return attention_map,features
        patch_features=nn.functional.adaptive_avg_pool2d(features,output_size=region_output_size)
        
        region_features=features[:,np.newaxis,:,:,:]*attention_map[:,:,np.newaxis,:,:]
        region_features=torch.mean(region_features,(3,4))
        region_features=l2_normal(region_features,2)
        return l2_normal(patch_features),region_features,attention_map
    

# def loss_entropy(attention_map):
#     log_attention_map=torch.log2(attention_map+ 1e-30)
#     entropy=(-attention_map*log_attention_map).mean()
#     return entropy
class Discriminator(nn.Module):
    def __init__(self,in_channels=1,n_B=8,n_C=8):
        super(Discriminator, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),padding=(1,1)),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=(1,1)),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding=(1,1)),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding=(1,1)),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding=(1,1)),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding=(0,0)),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding=(0,0)),
            nn.LeakyReLU(True)
            )
        self.classification1=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),padding=(0,0)),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=32,out_channels=1,kernel_size=(1,1),padding=(0,0)),
            nn.Sigmoid()
            )
    def forward(self, x):
        cnn1_out=self.cnn1(x[:,:1])
        return self.classification1(cnn1_out).flatten()
    
def loss_entropy(attention_map):
    entropy=(-attention_map*torch.log2(attention_map+ 1e-30)-(1-attention_map)*torch.log2(1-attention_map+ 1e-30)).mean()
    return entropy
    
def loss_distance(feature_map1,feanture_map2):
    return (feature_map1*feanture_map2).sum(1).mean()
    
def loss_area(attention_map,limit_rate=1/16,if_l2=True):
    area_list=torch.mean(attention_map,(2,3))-limit_rate
    if if_l2:
        loss=((-area_list)**2).to(torch.float32).mean()
    else:
        loss=torch.abs(area_list).to(torch.float32).mean()
    return loss
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:41:14 2019

@author: Administrator
"""
import torch
from collections import OrderedDict
import numpy as np

import torch
from torch import nn

def dice_loss(pre_y, true_y):
    smooth = 1.0
    true_y = true_y.view([-1])
    pre_y = pre_y.view([-1])
    if torch.max(true_y) == 0:
        true_y = 1 - true_y
        pre_y = 1 - pre_y
        TP=torch.dot(true_y, pre_y)
        G_sum=torch.sum(true_y)
        P_sum=torch.sum(pre_y)
        return -2*TP/(G_sum+P_sum+smooth)
    else:
        TP=torch.dot(true_y, pre_y)
        G_sum=torch.sum(true_y)
        P_sum=torch.sum(pre_y)
        return -2*TP/(G_sum+P_sum+smooth)

        
def load_weights_2_newmodel(new_model,old_model):
    new_name=[]
    for k,v in new_model.state_dict().items():
        print(k,v.size())
        new_name.append(k)
    old_model_weights=[]
    for k,v in old_model.items():
        print(k,v.size())
        old_model_weights.append(v)
        
    new_state_dict = OrderedDict()  
    for layer_index in range(len(new_name)):
        new_state_dict[new_name[layer_index]]=old_model_weights[layer_index]
    new_model.load_state_dict(new_state_dict)
    return new_model

def load_weights_from_datapallel_new_model(new_model,old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    new_model.load_state_dict(new_state_dict)
    return new_model

class conv_bn_relu_block(nn.Module):
    """ CONV->BN_RELU"""

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1,
                 padding=1, dilation=1, groups=1, bias=True):
        super(conv_bn_relu_block, self).__init__()
        self.base_processing = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.base_processing(x)
        return x
        
        
class vgg16_decoder(nn.Module):
    #
    def __init__(self,channel_num=[128,128,64,32,16],init_weights=True):
        super(vgg16_decoder,self).__init__()
        self.up_sample=nn.Upsample(scale_factor=2)
        self.base_bloack0=nn.Sequential(conv_bn_relu_block(in_channels=channel_num[0]+channel_num[1],out_channels=channel_num[1]),
                                   conv_bn_relu_block(in_channels=channel_num[1],out_channels=channel_num[2]))
        self.base_bloack1=nn.Sequential(conv_bn_relu_block(in_channels=channel_num[2]*2,out_channels=channel_num[2]),
                                   conv_bn_relu_block(in_channels=channel_num[2],out_channels=channel_num[3]))
        self.base_bloack2=nn.Sequential(conv_bn_relu_block(in_channels=channel_num[3]*2,out_channels=channel_num[3]),
                                   conv_bn_relu_block(in_channels=channel_num[3],out_channels=channel_num[4]))
        self.base_bloack3=nn.Sequential(conv_bn_relu_block(in_channels=channel_num[4]*2,out_channels=channel_num[4]),
                                   conv_bn_relu_block(in_channels=channel_num[4],out_channels=channel_num[4]))
        self.base_bloack4=nn.Sequential(
                                        nn.Conv2d(in_channels=channel_num[4], out_channels=1, kernel_size=(3,3),padding=1, dilation=1, groups=1, bias=True),
                                        nn.Sigmoid())
        
        if init_weights:
            self._initialize_weights()
    def forward(self,featuresx1,featuresx2,featuresx4,featuresx8,featuresx16):
        x=self.up_sample(featuresx16)
        del featuresx16
        x=torch.cat([x,featuresx8],dim=1)
        del featuresx8
        x=self.base_bloack0(x)
        
        x=self.up_sample(x)
        x=torch.cat([x,featuresx4],dim=1)
        del featuresx4
        x=self.base_bloack1(x)
        
        x=self.up_sample(x)
        x=torch.cat([x,featuresx2],dim=1)
        del featuresx2
        x=self.base_bloack2(x)
        
        x=self.up_sample(x)
        x=torch.cat([x,featuresx1],dim=1)
        del featuresx1
        x=self.base_bloack3(x)
        
        x=self.base_bloack4(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
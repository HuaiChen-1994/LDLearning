#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:11:11 2021

@author: sjtu439
"""
import os

#===========================================
#augmentation for pd_mixup
# Args: 
#     lr (float): Initinal learning rate.
#     resume (str): Load historical checkpoint if it is not null.
#     checkpoint_path (str): Path to save checkpoint.
#     low_dim (int): The number of low-dimension features.
#     batch_size (int): Setting batch size.
#     gpu (str): Setting the gpus to train this model
#     dataset_path(str): the path of CXR8
#     workers (int): The number of workers for dateloader.
#     arch (str): The architecture for encoder. It is set as 'vgg16_bn_tiny' by defaut
#     train_epoch (int): The number of training epoch.
#     region_output_size ((int,int)): The output size of final regions. It is set as (4,4) by defuat to divide the image evenly into 4 by 4 regions.
#===========================================
args_pd_mixup={
          'lr':0.001,
          'resume':'',
          'checkpoint_path':'./checkpoint_pd_mixup',
          'low_dim':32,
          'batch_size':4,
          'gpu':'0',
          'dataset_path':'../../datasets/X-ray/CXR8/images',
          'workers':2,
          'arch':'vgg16_bn_tiny',
          'train_epoch':20,
          'region_output_size':(8,8)
          }


#===========================================
#augmentation for ld
# Args: 
#     lr (float): Initinal learning rate.
#     resume (str): Load historical checkpoint if it is not null.
#     checkpoint_path (str): Path to save checkpoint.
#     low_dim (int): The number of low-dimension features.
#     batch_size (int): Setting batch size.
#     gpu (str): Setting the gpus to train this model
#     dataset_path(str): the path of kaggle dataset
#     workers (int): The number of workers for dateloader.
#     arch (str): The architecture for encoder. It is set as 'vgg16_bn_tiny' by defaut
#     train_epoch (int): The number of training epoch.
#     region_output_size ((int,int)): The output size of final regions. It is set as (4,4) by defuat to divide the image evenly into 4 by 4 regions.
#===========================================
args_ld={
        'lr':0.001,
          'resume':'',
          'checkpoint_path':'./checkpoint_ld',
          'low_dim':32,
          'cluster_num':16,
          'batch_size':4,
          'gpu':'3,2,1,0',
          'dataset_path':'../../datasets/X-ray/CXR8/images',
          'workers':3,
          'arch':'vgg16_bn_tiny',
          'train_epoch':80,
          'region_output_size':(8,8),
          }
args_ld['load_model_path']=os.path.join(args_pd_mixup['checkpoint_path']+'_'+str(args_ld['region_output_size'])+'_'+str(args_ld['low_dim'])+'/checkpoint_vgg16_bn_tiny_20.pth.tar')
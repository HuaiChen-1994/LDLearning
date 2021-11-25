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
#     dataset_path(str): the path of kaggle dataset
#     workers (int): The number of workers for dateloader.
#     arch (str): The architecture for encoder. It is set as 'vgg16_bn_tiny' by defaut
#     train_epoch (int): The number of training epoch.
#     region_output_size ((int,int)): The output size of final regions. It is set as (8,8) by defuat to divide the image evenly into 8 by 8 regions.
#     training_list_data_path: the list of high quality dataset 
#===========================================
args_pd_mixup={
          'lr':0.001,
          'resume':'',
          'checkpoint_path':'./checkpoint_pd_mixup',
          'low_dim':32,
          'batch_size':4,
          'gpu':'0',
          'dataset_path':'../../datasets/retinal_images/kaggle_training_512',
          'workers':2,
          'arch':'vgg16_bn_tiny',
          'train_epoch':20,
          'region_output_size':(8,8),
          'training_list_data_path':'../../datasets/retinal_images/kaggle_good_quality.npy'
          }


#===========================================
#augmentation for ld_prior
# Args: 
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
#     region_output_size ((int,int)): The output size of final regions. It is set as (8,8) by defuat to divide the image evenly into 8 by 8 regions.
#     training_list_data_path: the list of high quality dataset 
#     check_image_path: visualization the historical segmentation result
#     prior_label_path: the path of shape prior mask. '../../datasets/retinal_images/prior knowledge/COTA_vessel_mask' or '../../datasets/retinal_images/prior knowledge/DSA_mask'
#=============================================================
args_ld_prior={'lr':0.001,
          'resume':'',
          'checkpoint_path':'./checkpoint_ld',
          'low_dim':32,
          'cluster_num':8,
          'batch_size':3,
          'gpu':'2,3',
          'dataset_path':'../../datasets/retinal_images/kaggle_training_512',
          'workers':3,
          'arch':'vgg16_bn_tiny',
          'train_epoch':80,
          'region_output_size':(8,8),
          'training_list_data_path':'../../datasets/retinal_images/kaggle_good_quality.npy',
          'check_image_path':'check_image',
          'prior_label_path':'../../datasets/retinal_images/prior knowledge/COTA_vessel_mask'
          }
args_ld_prior['load_model_path']=os.path.join(args_pd_mixup['checkpoint_path']+'_'+str(args_ld_prior['region_output_size'])+'_'+str(args_ld_prior['low_dim'])+'/checkpoint_vgg16_bn_tiny_20.pth.tar')
args_ld_prior['check_image_path']=args_ld_prior['check_image_path']+'_'+str(args_ld_prior['region_output_size'])+'_'+str(args_ld_prior['low_dim'])+'_'+str(args_ld_prior['cluster_num'])
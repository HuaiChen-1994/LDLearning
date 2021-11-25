# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:22:09 2019

@author: Administrator
"""
import numpy as np
import random
import os
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms
def check_and_create_fold(path):
    if not os.path.exists(path):
        os.makedirs(path)
class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count        
def check_image(img1_1,img1_2,img2_1,img2_2,img3,mean=[0.5],std=[1]):
    if not os.path.exists('./temp'):
        os.makedirs('./temp')
    img1_1=img1_1.numpy()
    img1_2=img1_2.numpy()
    img2_1=img2_1.numpy()
    img2_2=img2_2.numpy()
    img3=img3.numpy()
    for image_index in range(len(img1_1)):
        image_instance=img1_1[image_index][0]
        image_instance=(image_instance*std+mean)*255
        image_instance=Image.fromarray(image_instance.astype('uint8'))#.convert('RGB')
        image_instance.save('./temp/'+str(image_index)+'1_1.jpg')
        
        image_instance=img1_2[image_index][0]
        image_instance=(image_instance*std+mean)*255
        image_instance=Image.fromarray(image_instance.astype('uint8'))#.convert('RGB')
        image_instance.save('./temp/'+str(image_index)+'1_2.jpg')
        
        image_instance=img2_1[image_index][0]
        image_instance=(image_instance*std+mean)*255
        image_instance=Image.fromarray(image_instance.astype('uint8'))#.convert('RGB')
        image_instance.save('./temp/'+str(image_index)+'2_1.jpg')
        
        image_instance=img2_2[image_index][0]
        image_instance=(image_instance*std+mean)*255
        image_instance=Image.fromarray(image_instance.astype('uint8'))#.convert('RGB')
        image_instance.save('./temp/'+str(image_index)+'2_2.jpg')
        
        image_instance=img3[image_index][0]
        image_instance=(image_instance*std+mean)*255
        image_instance=Image.fromarray(image_instance.astype('uint8'))#.convert('RGB')
        image_instance.save('./temp/'+str(image_index)+'3.jpg')

class MyDataset(data.Dataset):
    '''
    Data generater
    Args:
        total_num (int): The data number of one epoch.
        data_list (list or array): The list of data.
        data_path (str): The path where the data is.
        if_rot_90 (bool): If True, images will be randomly rotated by 0, 90, 180 or 270.
        if_flip (bool): If True, images will by flipped.
        if_mixup (bool): If True, returns the mixup images.
        transform_list: The list of transformation operations. 
                        It consists three part, respectively for augmenting, generating similar paris and normalization. 
    Returns:
        img1_1,img1_2,img2_1,img2_2 are two similar pairs ().
        img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot are corresponding augmentation operations of flip and rotation for img1_2 and img2_2.
        img3,img1_weight are mixup image and corresponding fusion weights and only will be returned if if_mixup is True.
    '''
    
    def __init__(self,
                 total_num,
                 data_list,
                 data_path,
                 if_rot_90=True,
                 if_flip=True,
                 if_mixup=True,
                 transform_list=None):
       self.data_list=data_list
       self.data_path=data_path
       self.transform_list=transform_list
       self.if_mixup=if_mixup
       self.if_flip=if_flip
       self.if_rot_90=if_rot_90
       self.total_num=total_num
    
    def __getitem__(self,idex):
        # Randomly choose two images
        choosen_index=random.randint(0, len(self.data_list)-1)
        img1=Image.open(os.path.join(self.data_path,self.data_list[choosen_index])).convert('L')
        choosen_index=random.randint(0, len(self.data_list)-1)
        img2=Image.open(os.path.join(self.data_path,self.data_list[choosen_index])).convert('L')
        img1=self.transform_list[0](img1)
        img2=self.transform_list[0](img2)
        
        # Generate similar pairs
        img1_1=self.transform_list[1](img1)
        img2_1=self.transform_list[1](img2)
        img1_2=self.transform_list[1](img1)
        img2_2=self.transform_list[1](img2)
        
        img1_2_flip=random.randint(0,1)
        img1_2_rot=random.randint(0,3)
        if self.if_flip:
            if img1_2_flip:
                img1_2=torch.flip(img1_2,dims=(1,2))
        if self.if_rot_90:
            if img1_2_rot:
                img1_2=torch.rot90(img1_2,k=img1_2_rot,dims=(1,2))
        
        img2_2_flip=random.randint(0,1)
        img2_2_rot=random.randint(0,3)
        if self.if_flip:
            if img2_2_flip:
                img2_2=torch.flip(img2_2,dims=(1,2))
        if self.if_rot_90:
            if img2_2_rot:
                img2_2=torch.rot90(img2_2,k=img2_2_rot,dims=(1,2))
            
        
        if self.if_mixup:
            # IF if_mixup is True, mixup images will be created.
            img1_weight=random.random()*0.6+0.2
            img3=img1_weight*img1_1+(1-img1_weight)*img2_1    
            img3=self.transform_list[2](img3)
            img1_weight=torch.tensor(img1_weight)    
            
            img1_1=self.transform_list[2](img1_1)
            img1_2=self.transform_list[2](img1_2)
            img2_1=self.transform_list[2](img2_1)
            img2_2=self.transform_list[2](img2_2)
                
            return img1_1,img1_2,img2_1,img2_2,img3,img1_weight,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot
        else:
            img1_1=self.transform_list[2](img1_1)
            img1_2=self.transform_list[2](img1_2)
            img2_1=self.transform_list[2](img2_1)
            img2_2=self.transform_list[2](img2_2)
                
            return img1_1,img1_2,img2_1,img2_2,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot
            
    def __len__(self):
        return self.total_num
    
if __name__=='__main__':
    dataset_path='../../datasets/X-ray/CXR8/images'
    child_path_list=os.listdir(dataset_path)
    train_data_list=[]
    for child_path in child_path_list:
        train_data_list=train_data_list+[os.path.join(child_path,data_instance) for data_instance in os.listdir(os.path.join(dataset_path,child_path))]
    val_data_list=train_data_list[100000:]
    train_data_list=train_data_list[:100000]
    normalize = transforms.Normalize(mean=[0.5],
                                         std=[1])
    trainset = MyDataset(
                        total_num=1000,
                          if_rot_90=True,
                          if_flip=True,
                          if_mixup=True,
                          data_list=train_data_list,
                          data_path=dataset_path,
                          transform_list=[transforms.Compose([
                                                             
                                                              transforms.RandomHorizontalFlip(p=0.5),
                                                              transforms.RandomVerticalFlip(p=0.5),
                                                              transforms.RandomRotation([-180,180]),
                                                              transforms.RandomResizedCrop(size=448, scale=(0.9,1.0)),]),
                                          transforms.Compose([
                                                                transforms.RandomRotation([-2,2]),
                                                              transforms.ColorJitter(0.2, 0.2, 0.2, 0.02), 
                                                              transforms.RandomGrayscale(p=0.2),
                                                              transforms.ToTensor(),
                                                              ]),
                                        transforms.Compose([normalize])]        )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True,num_workers=0, pin_memory=True)
    for batch_idx, (img1_1,img1_2,img2_1,img2_2,img3,img1_weight,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot) in enumerate(trainloader):
        check_image(img1_1,img1_2,img2_1,img2_2,img3)
        break
    
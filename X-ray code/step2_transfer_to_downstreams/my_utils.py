# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:22:09 2019

@author: Administrator
"""
import numpy as np
import random
import os
from PIL import Image
from torch.utils import data
 
def check_and_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

class MyDataset(data.Dataset):
    def __init__(self,
                 data_list,
                 data_path,
                 if_augumentation=True,
                 mean=[0.5],
                 std=[0.5]):
       self.data_list=data_list
       self.data_path=data_path
       self.if_augmentation=if_augumentation
       self.mean=mean
       self.std=std
    
    def __getitem__(self,idex):
        data_instance=self.data_list[idex]
        #read image
        image_instance=Image.open(os.path.join(self.data_path,data_instance[0])).convert('L')
        label_instance=Image.open(os.path.join(self.data_path,data_instance[1])).convert('L')
        #augmentation
        if self.if_augmentation:
            # # flip vertical
            # if random.randint(0,1):
            #     image_instance=image_instance.transpose(Image.FLIP_LEFT_RIGHT)
            #     label_instance=label_instance.transpose(Image.FLIP_LEFT_RIGHT)
            # # flip horizontal
            # if random.randint(0,1):
            #     image_instance=image_instance.transpose(Image.FLIP_TOP_BOTTOM)
            #     label_instance=label_instance.transpose(Image.FLIP_TOP_BOTTOM)
            # rotation
            rotation_degree=random.uniform(-30,30)
            image_instance = image_instance.rotate(rotation_degree)
            label_instance=label_instance.rotate(rotation_degree)
        image_instance=np.asarray(image_instance).astype('float32')
        image_instance=(image_instance/255.-self.mean)/self.std#image_instance/255.#(image_instance/255.-self.mean)/self.std
        image_instance=image_instance[np.newaxis,:,:]
  
        label_instance=np.asarray(label_instance).astype('float32')
        label_instance=((label_instance>(0.3*255))*1).astype('float32')
        return image_instance.astype('float32'),label_instance
        
    def __len__(self):
        return len(self.data_list)
                
if __name__=='__main__':
    import cv2
    main_path='../../../datasets/X-ray'
    training_data_list,test_data_list=np.load(os.path.join(main_path,'rightLung_MontgomerySet.npy'),allow_pickle=True)
    
    #create dataLoader
    train_set=MyDataset(data_list=training_data_list,
                        data_path=main_path,
                        if_augumentation=True)
    train_loader=data.DataLoader(train_set,batch_size=2,shuffle=True,num_workers=0)
    
    for iter_index,(batch_data,batch_label) in enumerate(train_loader):
        break
    batch_data=batch_data.cpu().numpy()
    batch_label=batch_label.cpu().numpy()
    check_and_create_folder('./temp')
    for index in range(len(batch_data)):
        data_instance=batch_data[index]
        label_instance=batch_label[index]
        data_instance=np.transpose(data_instance,(1,2,0))
        data_instance=np.floor((data_instance*0.5+0.5)*255)
        data_instance=np.clip(data_instance,0,255)#.astype('uint8')
        label_instance=np.floor(label_instance*255).astype('uint8')
        
        # data_instance=data_instance[:,:,[2,1,0]]
        cv2.imwrite(os.path.join('./temp',str(index)+'.jpg'),data_instance)
        cv2.imwrite(os.path.join('./temp',str(index)+'label.jpg'),label_instance)
    

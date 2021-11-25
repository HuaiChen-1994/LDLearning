# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 16:02:42 2020

@author: 439
"""
import numpy as np
import os
import random
import cv2
from skimage.util import invert
import imageio
import csv
import SimpleITK as sitk
def check_and_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

def preprocessing_SCR():
    #convert to png
    ori_path='./SCR/oriIMG/'
    save_path='./SCR/png_512'
    check_and_create_folder(save_path)
    data_list=os.listdir(ori_path)
    for data_instance in data_list[:]:
        raw_image = np.fromfile(os.path.join(ori_path,data_instance), dtype=">i2").reshape((2048, 2048))
        img = invert(raw_image)
        img=np.floor((img-img.min())/(img.max()-img.min())*255).astype('uint8')
        img=cv2.resize(img,(512,512))
        cv2.imwrite(os.path.join(save_path,data_instance.split('.')[0]+'.png'),img)
    
    #merge label
    ori_path_list=['./SCR/scratch/fold1/masks/',
                    './SCR/scratch/fold2/masks/']
    save_path_list=['./SCR/fold1','./SCR/fold2']
    target_list=['heart','lung','clavicle']
    for ori_path,save_path in zip(ori_path_list,save_path_list):
        for target_instance in target_list:
            check_and_create_folder(os.path.join(save_path,target_instance))
            if target_instance=='heart':
                data_list=os.listdir(os.path.join(ori_path,'heart'))
            if target_instance=='lung':
                data_list=os.listdir(os.path.join(ori_path,'left lung'))
            if target_instance=='clavicle':
                data_list=os.listdir(os.path.join(ori_path,'left clavicle'))
            for data_instance in data_list[:]:
                if target_instance=='heart':
                    image_data=imageio.mimread(os.path.join(ori_path,'heart',data_instance))[0]
                    image_data=cv2.resize(image_data,(512,512))
                    cv2.imwrite(os.path.join(save_path,target_instance,data_instance.split('.')[0]+'.png'),image_data)
                if target_instance=='lung':
                    image_data=imageio.mimread(os.path.join(ori_path,'left lung',data_instance))[0]
                    image_data=image_data+imageio.mimread(os.path.join(ori_path,'right lung',data_instance))[0]
                    image_data=cv2.resize(image_data,(512,512))
                    cv2.imwrite(os.path.join(save_path,target_instance,data_instance.split('.')[0]+'.png'),image_data)
                if target_instance=='clavicle':
                    image_data=imageio.mimread(os.path.join(ori_path,'left clavicle',data_instance))[0]
                    image_data=image_data+imageio.mimread(os.path.join(ori_path,'right clavicle',data_instance))[0]
                    image_data=cv2.resize(image_data,(512,512))
                    cv2.imwrite(os.path.join(save_path,target_instance,data_instance.split('.')[0]+'.png'),image_data)
    #split train and val data
    data_path='./SCR/png_512/'
    label1_path='./SCR/fold1/'
    label2_path='./SCR/fold2/'
    target_list=os.listdir(label1_path)
    for target_instance in target_list:
        data_list=os.listdir(os.path.join(label1_path,target_instance))
        training_list=[[data_path+data_instance,label1_path+target_instance+'/'+data_instance] for data_instance in data_list]
        data_list=os.listdir(os.path.join(label2_path,target_instance))
        test_list=[[data_path+data_instance,label2_path+target_instance+'/'+data_instance] for data_instance in data_list]
        np.save('./'+target_instance+'_SCR.npy',[training_list,test_list])
    
    #heart location data
    ori_path_list=['./SCR/scratch/fold1/masks/',
                    './SCR/scratch/fold2/masks/']
    save_path='./SCR'
    target_list=['heart']
    for target_instance in target_list:
        info_list=[]
        for path_instance in ori_path_list:
            data_list=os.listdir(os.path.join(path_instance,target_instance))
            for data_instance in data_list[:]:
                print(target_instance,data_instance)
                image_data=imageio.mimread(os.path.join(path_instance,target_instance,data_instance))[0]
                image_data=cv2.resize(image_data,(512,512))
                temp=image_data.sum(0)
                center_w=int(np.where(temp>0)[0].mean())
                center_h=int(np.where(image_data[:,center_w]>0)[0].mean())
                info_list.append([data_instance.split('.')[0],center_w,center_h])
        np.save(os.path.join(save_path,target_instance+'_center.npy'),info_list)

def preprocessing_SIIM():
    image_save_path='./SIIM-ACR/image_png/'
    mask_save_path='./SIIM-ACR/mask_png/'
    check_and_create_folder(mask_save_path)
    check_and_create_folder(image_save_path)
    main_path='./SIIM-ACR/siim/dicom-images-train'
    dcm_list=[]
    child_path_list=os.listdir(main_path)
    for child_path in child_path_list:
        dcm_path=os.path.join(main_path,child_path)
        dcm_path=os.path.join(dcm_path,os.listdir(dcm_path)[0])
        dcm_list.append([os.listdir(dcm_path)[0],dcm_path])
    dcm_list=np.asarray(dcm_list)
    f=csv.reader(open('./SIIM-ACR/siim/train-rle.csv','r'))
    csv_data=[]
    for info in f:
        csv_data.append(info)
    csv_data=csv_data[1:]
    csv_data=np.asarray(csv_data)
    
    for dcm_instance in dcm_list[:]:
        csv_index_list=np.where(csv_data[:,0]==dcm_instance[0][:-4])[0]
        if len(csv_index_list)==0:
            continue
        if csv_data[csv_index_list[0],1]=='-1':
            continue
        dcm_data=sitk.ReadImage(os.path.join(dcm_instance[1],dcm_instance[0]))
        dcm_data=sitk.GetArrayFromImage(dcm_data)[0]
        mask_binary=np.zeros_like(dcm_data)
        for csv_index in csv_index_list:
            mask_binary=mask_binary+rle2mask(csv_data[csv_index,1], mask_binary.shape[1], mask_binary.shape[0])
        mask_binary=((mask_binary>0)*255).astype('uint8')
        dcm_data=cv2.resize(dcm_data,(512,512))
        mask_binary=cv2.resize(mask_binary,(512,512)).transpose(1,0)
        cv2.imwrite(os.path.join(image_save_path,dcm_instance[0][:-4]+'.png'),dcm_data)
        cv2.imwrite(os.path.join(mask_save_path,dcm_instance[0][:-4]+'.png'),mask_binary)
    
    data_list=os.listdir(image_save_path)
    train_num=len(data_list)//2
    test_num=len(data_list)-train_num
    
    training_list=[[image_save_path+data_instance,mask_save_path+data_instance] for data_instance in data_list[:train_num]]
    test_list=[[image_save_path+data_instance,mask_save_path+data_instance] for data_instance in data_list[train_num:train_num+test_num]]
    np.save('./Pneumothorax_SIIM.npy',[training_list,test_list])
    
if __name__=='__main__':
    preprocessing_SCR()
    preprocessing_SIIM()
    
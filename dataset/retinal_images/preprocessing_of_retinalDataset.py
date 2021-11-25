# -*- coding: utf-8 -*-
"""
Created on Thu May 13 19:36:59 2021

@author: 439
"""
import numpy as np
import os
from PIL import Image
import cv2
from skimage import measure
import gzip
import shutil
import csv
import xlrd
def check_and_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def identify_roi(image_data):
    th1 =image_data[:,:,0]>20
    th2 =image_data[:,:,1]>20
    th3 =image_data[:,:,2]>20
    th_total=(((th1+th2+th3)>0)*1).astype('uint8')
    label_image=measure.label(th_total, connectivity = 2)
    props = measure.regionprops(label_image)
    numPix=[]
    for ia in range(len(props)):
        numPix += [props[ia].area]
    maxnum = max(numPix)
    index = numPix.index(maxnum)
    minr, minc, maxr, maxc = props[index].bbox#[minr, maxr),[minc, maxc)
    minr=np.max((minr-16,0))
    minc=np.max((minc-16,0))
    maxr=np.min((maxr+16,image_data.shape[0]))
    maxc=np.min((maxc+16,image_data.shape[1]))
    return minr, minc, maxr, maxc

def preprocessing_RITE():
    ori_data_path_list=['./original_data/RITE/training/images','./original_data/RITE/test/images']
    ori_label_path_list=['./original_data/RITE/training/vessel','./original_data/RITE/test/vessel']
    data_out_path_list=['./RITE_512/training/images','./RITE_512/test/images']
    label_out_path_list=['./RITE_512/training/vessel','./RITE_512/test/vessel']
    data_name_list=['training','test']
    out_size=512
    for ori_data_path,ori_label_path,data_out_path,label_out_path,data_name in zip(ori_data_path_list,ori_label_path_list,data_out_path_list,label_out_path_list,data_name_list):
        check_and_create_folder(data_out_path)
        check_and_create_folder(label_out_path)
        data_list=os.listdir(ori_data_path)
        data_list=[data_instance.split('_')[0] for data_instance in data_list]
        for data_instance in data_list:
            image_data=cv2.imread(os.path.join(ori_data_path,data_instance+'_'+data_name+'.tif'))
            label_data=Image.open(os.path.join(ori_label_path,data_instance+'_'+data_name+'.png')).convert('L')
            minr, minc, maxr, maxc=identify_roi(image_data)
            crop_image=image_data[minr:maxr,minc:maxc,:]
            crop_image=cv2.resize(crop_image,(out_size,out_size))
            cv2.imwrite(os.path.join(data_out_path,data_instance+'.png'),crop_image)
            
            label_data=((np.asarray(label_data)>0)*255).astype('uint8')
            crop_label=label_data[minr:maxr,minc:maxc]
            crop_label=cv2.resize(crop_label,(out_size,out_size))
            crop_label=((crop_label>(0.3*255))*255).astype('uint8')
            cv2.imwrite(os.path.join(label_out_path,data_instance+'.png'),crop_label) 
            
def preprocessing_STARE():
    ori_data_path='./original_data/STARE/stare-images'
    ori_label_path1='./original_data/STARE/labels-ah'
    ori_label_path2='./original_data/STARE/labels-vk'
    out_data_path='./STARE_512/stare-images'
    out_label_path1='./STARE_512/labels-ah'
    out_label_path2='./STARE_512/labels-vk'
    out_size=512
    check_and_create_folder(out_data_path)
    check_and_create_folder(out_label_path1)
    check_and_create_folder(out_label_path2)
    #unzip files
    for gz_path in [ori_data_path,ori_label_path1,ori_label_path2]:
        gz_files_list=os.listdir(gz_path)
        out_path=gz_path+'_unzip'
        check_and_create_folder(out_path)
        for gz_file_instance in gz_files_list:
            gz_file =  gzip.GzipFile(os.path.join(gz_path,gz_file_instance))  
            open(os.path.join(out_path,gz_file_instance[:-3]), "wb+").write(gz_file.read())  
            gz_file.close() 
    #crop and resize
    data_list=os.listdir(ori_data_path+'_unzip')
    data_list=[data_instance.split('.')[0] for data_instance in data_list]
    for data_instance in data_list:
        image_data=Image.open(os.path.join(ori_data_path+'_unzip',data_instance+'.ppm'))
        label_data1=Image.open(os.path.join(ori_label_path1+'_unzip',data_instance+'.ah.ppm')).convert('L')
        label_data2=Image.open(os.path.join(ori_label_path2+'_unzip',data_instance+'.vk.ppm')).convert('L')
        image_data=np.asarray(image_data)
        image_data=cv2.cvtColor(image_data,cv2.COLOR_RGB2BGR)
        label_data1=np.asarray(label_data1)
        label_data2=np.asarray(label_data2)
        minr, minc, maxr, maxc=identify_roi(image_data)
        crop_image=image_data[minr:maxr,minc:maxc,:]
        crop_image=cv2.resize(crop_image,(out_size,out_size))
        cv2.imwrite(os.path.join(out_data_path,data_instance+'.png'),crop_image)
        
        label_data=((np.asarray(label_data1)>0)*255).astype('uint8')
        crop_label=label_data[minr:maxr,minc:maxc]
        crop_label=cv2.resize(crop_label,(out_size,out_size))
        crop_label=((crop_label>(0.3*255))*255).astype('uint8')
        cv2.imwrite(os.path.join(out_label_path1,data_instance+'.png'),crop_label) 
        
        label_data=((np.asarray(label_data2)>0)*255).astype('uint8')
        crop_label=label_data[minr:maxr,minc:maxc]
        crop_label=cv2.resize(crop_label,(out_size,out_size))
        crop_label=((crop_label>(0.3*255))*255).astype('uint8')
        cv2.imwrite(os.path.join(out_label_path2,data_instance+'.png'),crop_label)
        
def preprocessing_CHASEDB():
    out_size=512
    ori_data_path='./original_data/CHASEDB'
    out_data_path='./CHASEDB_512'
    check_and_create_folder(out_data_path)
    data_list=os.listdir(ori_data_path)
    data_list=[data_instance[:9] for data_instance in data_list] 
    data_list=np.asarray(data_list)
    for data_instance in data_list:
        image_data=cv2.imread(os.path.join(ori_data_path,data_instance+'.jpg'))
        label_data1=Image.open(os.path.join(ori_data_path,data_instance+'_1stHO.png')).convert('L')
        label_data1=np.asarray(label_data1)
        label_data2=Image.open(os.path.join(ori_data_path,data_instance+'_2ndHO.png')).convert('L')
        label_data2=np.asarray(label_data2)
    
        minr, minc, maxr, maxc=identify_roi(image_data)
        crop_image=image_data[minr:maxr,minc:maxc,:]
        crop_image=cv2.resize(crop_image,(out_size,out_size))
        cv2.imwrite(os.path.join(out_data_path,data_instance+'.png'),crop_image)
        
        label_data=((np.asarray(label_data1)>0)*255).astype('uint8')
        crop_label=label_data[minr:maxr,minc:maxc]
        crop_label=cv2.resize(crop_label,(out_size,out_size))
        crop_label=((crop_label>(0.3*255))*255).astype('uint8')
        cv2.imwrite(os.path.join(out_data_path,data_instance+'_1st.png'),crop_label) 
        
        label_data=((np.asarray(label_data2)>0)*255).astype('uint8')
        crop_label=label_data[minr:maxr,minc:maxc]
        crop_label=cv2.resize(crop_label,(out_size,out_size))
        crop_label=((crop_label>(0.3*255))*255).astype('uint8')
        cv2.imwrite(os.path.join(out_data_path,data_instance+'_2nd.png'),crop_label) 
        
def preprocessing_HRF():
    out_size=512
    ori_data_path='./original_data/HRF/images'
    ori_label_path='./original_data/HRF/manual1'
    out_data_path='./HRF_512/images'
    out_label_path='./HRF_512/manual1'
    check_and_create_folder(out_data_path)
    check_and_create_folder(out_label_path)
    #read xls
    workbook=xlrd.open_workbook('./original_data/HRF/optic_disk_centers.xls')
    worksheet=workbook.sheet_by_index(0)
    worksheet1=workbook.sheet_by_index(1)
    nrows=worksheet.nrows
    xls_info=[]
    for i in range(nrows): #循环打印每一行
        xls_info.append(worksheet.row_values(i))
        if i==4:
            xls_info[-1]=[xls_info[-1][0]]+worksheet1.row_values(i)[1:]#there is a wrong center in sheet0
    xls_info=xls_info[1:]
    out_info=[]
    for xls_instance in xls_info[:]:
        if os.path.exists(os.path.join(ori_data_path,xls_instance[0]+'.jpg')):
            image_data=cv2.imread(os.path.join(ori_data_path,xls_instance[0]+'.jpg'))
        else:
            image_data=cv2.imread(os.path.join(ori_data_path,xls_instance[0]+'.JPG'))
        label_data=cv2.imread(os.path.join(ori_label_path,xls_instance[0]+'.tif'),2)
        label_data=np.asarray(label_data)
        
        
        minr, minc, maxr, maxc=identify_roi(image_data)
        out_info.append([xls_instance[0],int((float(xls_instance[1])-minc)*out_size/(maxc-minc)),int((float(xls_instance[2])-minr)*out_size/(maxr-minr))])
        crop_image=image_data[minr:maxr,minc:maxc,:]
        crop_image=cv2.resize(crop_image,(out_size,out_size))
        cv2.imwrite(os.path.join(out_data_path,xls_instance[0]+'.png'),crop_image)
        
        label_data=((np.asarray(label_data)>0)*255).astype('uint8')
        crop_label=label_data[minr:maxr,minc:maxc]
        crop_label=cv2.resize(crop_label,(out_size,out_size))
        crop_label=((crop_label>(0.01*255))*255).astype('uint8')
        cv2.imwrite(os.path.join(out_label_path,xls_instance[0]+'.png'),crop_label)
    np.save('./HRF_512/optic_disk_centers.npy',out_info)

def preprocessing_kaggle():
    input_path='./original_data/kaggle_training'
    out_size=512
    out_path='./kaggle_training'+'_'+str(out_size)
    check_and_create_folder(out_path)
    data_list=os.listdir(input_path)
    for data_index,data_instance in enumerate(data_list):
        print('\r',data_index,end='')
        if os.path.exists(os.path.join(out_path,data_instance)):
            continue
        image_data=cv2.imread(os.path.join(input_path,data_instance))
        th1 =image_data[:,:,0]>20
        th2 =image_data[:,:,1]>20
        th3 =image_data[:,:,2]>20
        th_total=(((th1+th2+th3)>0)*1).astype('uint8')
        label_image=measure.label(th_total, connectivity = 2)
        props = measure.regionprops(label_image)
        numPix=[]
        for ia in range(len(props)):
            numPix += [props[ia].area]
        if len(numPix)==0:
            print(data_instance,'no roi')
            continue
        maxnum = max(numPix)
        if maxnum<label_image.size/4:
            print(data_instance,'roi region is too small')
            continue
        index = numPix.index(maxnum)
        minr, minc, maxr, maxc = props[index].bbox#[minr, maxr),[minc, maxc)
        minr=np.max((minr-16,0))
        minc=np.max((minc-16,0))
        maxr=np.min((maxr+16,image_data.shape[0]))
        maxc=np.min((maxc+16,image_data.shape[1]))
        crop_image=image_data[minr:maxr,minc:maxc,:]
        crop_image=cv2.resize(crop_image,(out_size,out_size))
        cv2.imwrite(os.path.join(out_path,data_instance),crop_image)
def preprocessing_Drishti():
    ori_data_path_list=['./original_data/Drishti-GS1_files/Drishti-GS1_files/Training/Images/',
                        './original_data/Drishti-GS1_files/Drishti-GS1_files/Test/Images/']
    ori_label_path_list=['./original_data/Drishti-GS1_files/Drishti-GS1_files/Training/GT/',
                        './original_data/Drishti-GS1_files/Drishti-GS1_files/Test/Test_GT/']
    out_data_path_list=['./Drishti-GS1_512/Training/Images',
                         './Drishti-GS1_512/Test/Images']
    out_label_path_list=['./Drishti-GS1_512/Training/GT',
                         './Drishti-GS1_512/Test/GT']
    out_size=512
    for ori_data_path,ori_label_path,out_data_path,out_label_path in zip(ori_data_path_list,ori_label_path_list,out_data_path_list,out_label_path_list):
        check_and_create_folder(out_data_path)
        check_and_create_folder(out_label_path)
        data_list=os.listdir(ori_data_path)
        data_list=[data_instance.split('.')[0] for data_instance in data_list]
        for data_instance in data_list:
            image_data=cv2.imread(os.path.join(ori_data_path,data_instance+'.png'))
            label_data_OD=cv2.imread(os.path.join(ori_label_path,data_instance,'SoftMap',data_instance+'_ODsegSoftmap.png'),2)
            label_data_cup=cv2.imread(os.path.join(ori_label_path,data_instance,'SoftMap',data_instance+'_cupsegSoftmap.png'),2)
            
            minr, minc, maxr, maxc=identify_roi(image_data)
            crop_image=image_data[minr:maxr,minc:maxc,:]
            crop_image=cv2.resize(crop_image,(out_size,out_size))
            cv2.imwrite(os.path.join(out_data_path,data_instance+'.png'),crop_image)
            
            crop_label=label_data_OD[minr:maxr,minc:maxc]
            crop_label=cv2.resize(crop_label,(out_size,out_size))
            cv2.imwrite(os.path.join(out_label_path,data_instance+'_OD.png'),crop_label) 
            
            crop_label=label_data_cup[minr:maxr,minc:maxc]
            crop_label=cv2.resize(crop_label,(out_size,out_size))
            cv2.imwrite(os.path.join(out_label_path,data_instance+'_cup.png'),crop_label) 
def preprocessing_IDRID():
    # segmentation part
    ori_data_path_list=['./original_data/IDRID/A. Segmentation/1. Original Images/a. Training Set',
                        './original_data/IDRID/A. Segmentation/1. Original Images/b. Testing Set']
    ori_label_path_list=['./original_data/IDRID/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set',
                        './original_data/IDRID/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set']
    label_item_list=['1. Microaneurysms','2. Haemorrhages','3. Hard Exudates','4. Soft Exudates','5. Optic Disc']
    out_data_path_list=['./IDRID_512/Segmentation/Training/Images',
                        './IDRID_512/Segmentation/Test/Images']
    out_label_path_list=['./IDRID_512/Segmentation/Training/GTs',
                        './IDRID_512/Segmentation/Test/GTs']
    out_size=512
    for ori_data_path,ori_label_path,out_data_path,out_label_path in zip(ori_data_path_list,ori_label_path_list,out_data_path_list,out_label_path_list):
        check_and_create_folder(out_data_path)
        for label_item in label_item_list:
            check_and_create_folder(os.path.join(out_label_path,label_item))
        check_and_create_folder(out_label_path)
        data_list=os.listdir(ori_data_path)
        data_list=[data_instance.split('.')[0] for data_instance in data_list]
        for data_instance in data_list:
            image_data=cv2.imread(os.path.join(ori_data_path,data_instance+'.jpg'))
            minr, minc, maxr, maxc=identify_roi(image_data)
            crop_image=image_data[minr:maxr,minc:maxc,:]
            crop_image=cv2.resize(crop_image,(out_size,out_size))
            cv2.imwrite(os.path.join(out_data_path,data_instance+'.png'),crop_image)
            
            for label_item in label_item_list:
                if label_item=='1. Microaneurysms':
                    if not os.path.isfile(os.path.join(ori_label_path,label_item,data_instance+'_MA.tif')):
                        continue
                    label_data=cv2.imread(os.path.join(ori_label_path,label_item,data_instance+'_MA.tif'),2)
                if label_item=='2. Haemorrhages':
                    if not os.path.isfile(os.path.join(ori_label_path,label_item,data_instance+'_HE.tif')):
                        continue
                    label_data=cv2.imread(os.path.join(ori_label_path,label_item,data_instance+'_HE.tif'),2)
                if label_item=='3. Hard Exudates':
                    if not os.path.isfile(os.path.join(ori_label_path,label_item,data_instance+'_EX.tif')):
                        continue
                    label_data=cv2.imread(os.path.join(ori_label_path,label_item,data_instance+'_EX.tif'),2)
                if label_item=='4. Soft Exudates':
                    if not os.path.isfile(os.path.join(ori_label_path,label_item,data_instance+'_SE.tif')):
                        continue
                    label_data=cv2.imread(os.path.join(ori_label_path,label_item,data_instance+'_SE.tif'),2)
                if label_item=='5. Optic Disc':
                    if not os.path.isfile(os.path.join(ori_label_path,label_item,data_instance+'_OD.tif')):
                        continue
                    label_data=cv2.imread(os.path.join(ori_label_path,label_item,data_instance+'_OD.tif'),2)
                label_data=((label_data>0)*255).astype('uint8')
                crop_label=label_data[minr:maxr,minc:maxc]
                crop_label=cv2.resize(crop_label,(out_size,out_size))
                crop_label=((crop_label>(0.3*255))*255).astype('uint8')
                cv2.imwrite(os.path.join(out_label_path,label_item,data_instance+'.png'),crop_label)
    
    #grading part
    ori_data_path_list=['./original_data/IDRID/B. Disease Grading/1. Original Images/a. Training Set',
                        './original_data/IDRID/B. Disease Grading/1. Original Images/b. Testing Set']
    out_data_path_list=['./IDRID_512/Disease_Grading/Training',
                        './IDRID_512/Disease_Grading/Test']
    out_size=512
    if os.path.exists('./IDRID_512/Disease_Grading/GTs'):
          shutil.rmtree('./IDRID_512/Disease_Grading/GTs')
    shutil.copytree('./original_data/IDRID/B. Disease Grading/2. Groundtruths', './IDRID_512/Disease_Grading/GTs')
    for ori_data_path,out_data_path in zip(ori_data_path_list,out_data_path_list):
        check_and_create_folder(out_data_path)
        data_list=os.listdir(ori_data_path)
        data_list=[data_instance.split('.')[0] for data_instance in data_list]
        for data_instance in data_list:
            image_data=cv2.imread(os.path.join(ori_data_path,data_instance+'.jpg'))
            minr, minc, maxr, maxc=identify_roi(image_data)
            crop_image=image_data[minr:maxr,minc:maxc,:]
            crop_image=cv2.resize(crop_image,(out_size,out_size))
            cv2.imwrite(os.path.join(out_data_path,data_instance+'.png'),crop_image)
    
    #location part
    ori_data_path_list=['./original_data/IDRID/C. Localization/1. Original Images/a. Training Set',
                        './original_data/IDRID/C. Localization/1. Original Images/b. Testing Set']
    OD_info_path_list=['./original_data/IDRID/C. Localization/2. Groundtruths/1. Optic Disc Center Location/a. IDRiD_OD_Center_Training Set_Markups.csv',
                       './original_data/IDRID/C. Localization/2. Groundtruths/1. Optic Disc Center Location/b. IDRiD_OD_Center_Testing Set_Markups.csv']
    fovea_info_path_list=['./original_data/IDRID/C. Localization/2. Groundtruths/2. Fovea Center Location/IDRiD_Fovea_Center_Training Set_Markups.csv',
                       './original_data/IDRID/C. Localization/2. Groundtruths/2. Fovea Center Location/IDRiD_Fovea_Center_Testing Set_Markups.csv']
    out_data_path_list=['./IDRID_512/Localization/Training',
                        './IDRID_512/Localization/Test']
    out_size=512
    for ori_data_path,out_data_path,OD_info_path,fovea_info_path in zip(ori_data_path_list,out_data_path_list,
                                                                        OD_info_path_list,fovea_info_path_list):
        check_and_create_folder(out_data_path)
        OD_info=[]
        with open(OD_info_path) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                if len(row[0]):
                    OD_info.append(row)
        OD_info=OD_info[1:]
        fovea_info=[]
        with open(fovea_info_path) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                if len(row[0]):
                    fovea_info.append(row)
        fovea_info=fovea_info[1:]
        for instance_index in range(len(OD_info)):
            image_data=cv2.imread(os.path.join(ori_data_path,OD_info[instance_index][0]+'.jpg'))
            minr, minc, maxr, maxc=identify_roi(image_data)
            OD_info[instance_index][1]=int((int(OD_info[instance_index][1])-minc)*out_size/(maxc-minc))
            OD_info[instance_index][2]=int((int(OD_info[instance_index][2])-minr)*out_size/(maxr-minr))
            fovea_info[instance_index][1]=int((int(fovea_info[instance_index][1])-minc)*out_size/(maxc-minc))
            fovea_info[instance_index][2]=int((int(fovea_info[instance_index][2])-minr)*out_size/(maxr-minr))
            
            crop_image=image_data[minr:maxr,minc:maxc,:]
            crop_image=cv2.resize(crop_image,(out_size,out_size))
            cv2.imwrite(os.path.join(out_data_path,OD_info[instance_index][0]+'.png'),crop_image)
        np.save(os.path.join('./IDRID_512/Localization',OD_info_path.split('/')[-1][:-4]+'.npy'),OD_info)
        np.save(os.path.join('./IDRID_512/Localization',fovea_info_path.split('/')[-1][:-4]+'.npy'),fovea_info)
            
def split_training_and_test_dataset():
    ########split training and test dataset
    ##Retinal vessel segmentation
    #CHASEDB 20 training and 8 test
    data_list=os.listdir('./CHASEDB_512')
    data_list=[data_instance[:9] for data_instance in data_list]
    data_list=np.asarray(data_list)
    data_list=np.unique(data_list)
    training_list=data_list[:20]
    test_list=data_list[20:]
    training_list=[[os.path.join('./CHASEDB_512/'+data_instance+'.png'),os.path.join('./CHASEDB_512/'+data_instance+'_1st.png')] for data_instance in training_list]
    test_list=[[os.path.join('./CHASEDB_512/'+data_instance+'.png'),os.path.join('./CHASEDB_512/'+data_instance+'_1st.png')] for data_instance in test_list]
    np.save('./Vessel_CHASEDB.npy',[training_list,test_list])
    
    #HRF 25 training and 20 test
    data_list=os.listdir('./HRF_512/images')
    training_list=data_list[:25]
    test_list=data_list[25:]
    training_list=[[os.path.join('./HRF_512/images/'+data_instance),os.path.join('./HRF_512/manual1/'+data_instance)] for data_instance in training_list]
    test_list=[[os.path.join('./HRF_512/images/'+data_instance),os.path.join('./HRF_512/manual1/'+data_instance)] for data_instance in test_list]
    np.save('./Vessel_HRF.npy',[training_list,test_list])
    
    #RITE 20 training and 20 test
    training_list=os.listdir('./RITE_512/training/images')
    test_list=os.listdir('./RITE_512/test/images')
    training_list=[[os.path.join('./RITE_512/training/images/'+data_instance),os.path.join('./RITE_512/training/vessel/'+data_instance)] for data_instance in training_list]
    test_list=[[os.path.join('./RITE_512/test/images/'+data_instance),os.path.join('./RITE_512/test/vessel/'+data_instance)] for data_instance in test_list]
    np.save('./Vessel_RITE.npy',[training_list,test_list])
    
    #STARE 10 training and 10 test
    data_list=os.listdir('./STARE_512/stare-images')
    training_list=data_list[:10]
    test_list=data_list[10:]
    training_list=[[os.path.join('./STARE_512/stare-images/'+data_instance),os.path.join('./STARE_512/labels-ah/'+data_instance)] for data_instance in training_list]
    test_list=[[os.path.join('./STARE_512/stare-images/'+data_instance),os.path.join('./STARE_512/labels-ah/'+data_instance)] for data_instance in test_list]
    np.save('./Vessel_STARE.npy',[training_list,test_list])
    
    ##OD and cup segmentation
    #Drishti-GS1 50 training and 51 test
    training_list=os.listdir('./Drishti-GS1_512/Training/Images')
    test_list=os.listdir('./Drishti-GS1_512/Test/Images')
    training_list=[[os.path.join('./Drishti-GS1_512/Training/Images/'+data_instance),os.path.join('./Drishti-GS1_512/Training/GT/'+data_instance.split('.')[0]+'_OD.png')] for data_instance in training_list]
    test_list=[[os.path.join('./Drishti-GS1_512/Test/Images/'+data_instance),os.path.join('./Drishti-GS1_512/Test/GT/'+data_instance.split('.')[0]+'_OD.png')] for data_instance in test_list]
    np.save('./OD_Drishti-GS1.npy',[training_list,test_list])
    training_list=os.listdir('./Drishti-GS1_512/Training/Images')
    test_list=os.listdir('./Drishti-GS1_512/Test/Images')
    training_list=[[os.path.join('./Drishti-GS1_512/Training/Images/'+data_instance),os.path.join('./Drishti-GS1_512/Training/GT/'+data_instance.split('.')[0]+'_cup.png')] for data_instance in training_list]
    test_list=[[os.path.join('./Drishti-GS1_512/Test/Images/'+data_instance),os.path.join('./Drishti-GS1_512/Test/GT/'+data_instance.split('.')[0]+'_cup.png')] for data_instance in test_list]
    np.save('./CUP_Drishti-GS1.npy',[training_list,test_list])
    
    #IDRID 54 training and 27 test
    training_list=os.listdir('./IDRID_512/Segmentation/Training/GTs/5. Optic Disc')
    test_list=os.listdir('./IDRID_512/Segmentation/Test/GTs/5. Optic Disc')
    training_list=[[os.path.join('./IDRID_512/Segmentation/Training/Images/'+data_instance),os.path.join('./IDRID_512/Segmentation/Training/GTs/5. Optic Disc/'+data_instance)] for data_instance in training_list]
    test_list=[[os.path.join('./IDRID_512/Segmentation/Test/Images/'+data_instance),os.path.join('./IDRID_512/Segmentation/Test/GTs/5. Optic Disc/'+data_instance)] for data_instance in test_list]
    np.save('./OD_IDRID.npy',[training_list,test_list])
    
    ##Lesions segmentation
    for gt_path in ['1. Microaneurysms','2. Haemorrhages','3. Hard Exudates','4. Soft Exudates']:
        training_list=os.listdir('./IDRID_512/Segmentation/Training/GTs/'+gt_path)
        test_list=os.listdir('./IDRID_512/Segmentation/Test/GTs/'+gt_path)
        training_list=[[os.path.join('./IDRID_512/Segmentation/Training/Images/'+data_instance),os.path.join('./IDRID_512/Segmentation/Training/GTs/'+gt_path+'/'+data_instance)] for data_instance in training_list]
        test_list=[[os.path.join('./IDRID_512/Segmentation/Test/Images/'+data_instance),os.path.join('./IDRID_512/Segmentation/Test/GTs/'+gt_path+'/'+data_instance)] for data_instance in test_list]
        np.save('./'+gt_path[3:]+'_IDRID.npy',[training_list,test_list])
            
if __name__=='__main__':
    # ###########Kaggle
    # preprocessing_kaggle()
    
    # ###########RITE
    # preprocessing_RITE()
        
    # ###########STARE
    # preprocessing_STARE() 
    
    # ##########CHASEDB
    # preprocessing_CHASEDB()
    
    # ##########HRF
    # preprocessing_HRF()
    
    #########Drishti-GS1
    preprocessing_Drishti()
    
    #########IDRID
    preprocessing_IDRID()
            
    #######split training and test dataset
    split_training_and_test_dataset()
    
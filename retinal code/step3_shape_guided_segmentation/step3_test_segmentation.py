# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:53:20 2021

@author: 439
"""
import numpy as np
import os
import models
import os
import torch
import xlwt
from PIL import Image
from collections import OrderedDict
os.environ['CUDA_VISIBLE_DEVICES']='0'
def load_weights_from_datapallel_new_model(new_model,old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    new_model.load_state_dict(new_state_dict)
    return new_model

def test(encoder_net,decoder_net,test_list,main_data_path,test_out_path):
    if not os.path.exists(test_out_path):
        os.makedirs(test_out_path)
    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('DSC')
    DSC_attention_list=[]
    DSC_cos_list=[]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    for instance_index,(image_instance,label_instance) in enumerate(test_list):
        use_gpu=torch.cuda.is_available()
        #read image
        image=Image.open(os.path.join(main_data_path,image_instance))
        image=np.asarray(image).astype('float32')
        field_mask=torch.tensor((((image>20).sum(2))>0)*1)
        
        image=(image/255.-mean)/std
        image=np.transpose(image,(2,0,1))[np.newaxis,:,:,:].astype('float32')
        image=torch.tensor(image)
        if use_gpu:
            image=image.cuda()
            field_mask=field_mask.cuda()
        featuresx1,featuresx2,featuresx4,featuresx8,featuresx16=encoder_net(image)
        attention_map,embedded_features=decoder_net([featuresx1,featuresx2,featuresx4,featuresx8,featuresx16],if_clustering=True,if_test=True)
        
        attention_map=attention_map*field_mask[np.newaxis,np.newaxis,:,:]
        region_features=embedded_features[:,np.newaxis,:,:,:]*attention_map[:,:,np.newaxis,:,:]
        region_features=torch.mean(region_features,(3,4))
        region_features=models.normalize.l2_normal(region_features,2)[0]
        
        cos_similar=(region_features[:,:,np.newaxis,np.newaxis]*embedded_features[:,:,:,:]).sum(1)
        cos_similar=(cos_similar/0.2).exp()
        cos_similar=cos_similar/(cos_similar.sum(0)[np.newaxis,:,:])
        
        cos_similar=(cos_similar[0]*field_mask).data.cpu().numpy()
        binary_out_cos_similar=cos_similar>0.5
        
        attention_map=(attention_map[0,0]*field_mask).data.cpu().numpy()
        binary_out_attention_map=attention_map>0.5
        
        #save predict image
        cos_similar=np.floor(cos_similar*255).astype('uint8')
        cos_similar=Image.fromarray(cos_similar)
        cos_similar.save(os.path.join(test_out_path,'cos_similar_'+image_instance.split('/')[-1]))
        attention_map=np.floor(attention_map*255).astype('uint8')
        attention_map=Image.fromarray(attention_map)
        attention_map.save(os.path.join(test_out_path,'attention_'+image_instance.split('/')[-1]))
        
        label=Image.open(os.path.join(main_data_path,label_instance))
        label=(np.asarray(label)[:,:]>(0.3*255)).astype('float32')
        DSC_attention=2*np.sum(label*binary_out_attention_map)/(np.sum(label)+np.sum(binary_out_attention_map))
        DSC_attention_list.append(DSC_attention)
        DSC_cos=2*np.sum(label*binary_out_cos_similar)/(np.sum(label)+np.sum(binary_out_cos_similar))
        DSC_cos_list.append(DSC_cos)
        worksheet.write(instance_index,0,image_instance)
        worksheet.write(instance_index,1,DSC_attention)
        worksheet.write(instance_index,2,DSC_cos)
    DSC_attention_list=np.asarray(DSC_attention_list)
    DSC_cos_list=np.asarray(DSC_cos_list)
    worksheet.write(instance_index+1,0,'Mean')
    worksheet.write(instance_index+1,1,np.mean(DSC_attention_list))
    worksheet.write(instance_index+1,2,np.mean(DSC_cos_list))
    worksheet.write(instance_index+2,0,'std')
    worksheet.write(instance_index+2,1,np.std(DSC_attention_list))
    worksheet.write(instance_index+2,2,np.std(DSC_cos_list))
    
    print('mean:'+str(np.mean(DSC_attention_list))+',std:'+str(np.std(DSC_attention_list))+\
          'mean_cos:'+str(np.mean(DSC_cos_list))+',std_cos:'+str(np.std(DSC_cos_list)) )
    workbook.save(os.path.join(test_out_path,'test_DSC.xls'))
if __name__=='__main__':
    main_data_path='../../datasets/retinal_images'
    npy_name_list=['Vessel_CHASEDB.npy','Vessel_HRF.npy','Vessel_RITE.npy','Vessel_STARE.npy']
    weights_path='./checkpoint_ld_(8, 8)_32_8'
    
    #build models
    encoder_net = models.__dict__['vgg16_bn_tiny']().cuda().eval()
    decoder_net=models.my_models.decoder(channel_list=[128,128,64,32,16,32,8]).cuda().eval()
    checkpoint = torch.load(os.path.join(weights_path,'model_best_vgg16_bn_tiny.pth.tar'))
    encoder_net=load_weights_from_datapallel_new_model(encoder_net, checkpoint['encoder_net'])
    decoder_net=load_weights_from_datapallel_new_model(decoder_net, checkpoint['decoder_net'])
    use_gpu=torch.cuda.is_available()
    if use_gpu:
        encoder_net=encoder_net.cuda()
        decoder_net=decoder_net.cuda()
    for npy_name in npy_name_list[:]:
        print(npy_name)
        test_out_path=os.path.join('./segmentation',npy_name.split('.')[0])
        training_list,test_list=np.load(os.path.join(main_data_path,npy_name),allow_pickle=True)
        test(encoder_net,decoder_net,test_list,main_data_path,test_out_path)
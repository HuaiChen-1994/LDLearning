# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:16:35 2019

@author: Administrator
"""
import sys
sys.path.append('./')
import os
import numpy as np
import xlwt
import torch
from torch import optim
from torch.utils import data
from PIL import Image

import vgg
from my_utils import MyDataset,check_and_create_folder
from my_models import vgg16_decoder,dice_loss,load_weights_from_datapallel_new_model
def training(training_loader,val_loader,trainable_nets,encoder_net,decoder_net,epoch,optimizer,checkpoint_save_path):
    use_gpu=torch.cuda.is_available()
    best_eval_loss=np.inf
    train_iter=training_loader.__len__()
    for epoch_index in range(epoch):
        print('\nepoch:'+str(epoch_index))
        # training stage
        #set parameters trainable
        for net_insatnce in trainable_nets:
            net_insatnce.train()
            for param in net_insatnce.parameters():
                param.requires_grad = True
        # bp to updata
        running_loss=[]
        for iter_index,(batch_data,batch_label) in enumerate(training_loader):
            if len(running_loss) > 0:
                print('\r','iter:'+str(iter_index)+'/'+str(train_iter)+',train loss:'+str(np.mean(running_loss[-50:], axis=0)),end='')
            if use_gpu:
                batch_data=batch_data.cuda()
                batch_label=batch_label.cuda()
            featuresx1,featuresx2,featuresx4,featuresx8,featuresx16=encoder_net(batch_data)
            out=decoder_net(featuresx1,featuresx2,featuresx4,featuresx8,featuresx16)
            loss = dice_loss(out, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.data.cpu().numpy())
            del out,featuresx1,featuresx2,featuresx4,featuresx8,featuresx16

        #val stage
        #set parameters frozen
        for net_instance in trainable_nets:
            net_instance.eval()
            for param in net_instance.parameters():
                param.requires_grad = False
        eval_loss=[]
        for iter_index,(batch_data,batch_label) in enumerate(val_loader):
            if use_gpu:
                batch_data=batch_data.cuda()
                batch_label=batch_label.cuda()
            featuresx1,featuresx2,featuresx4,featuresx8,featuresx16=encoder_net(batch_data)
            out=decoder_net(featuresx1,featuresx2,featuresx4,featuresx8,featuresx16)
            loss = dice_loss(out, batch_label)
            eval_loss.append(loss.data.cpu().numpy())
            del out,featuresx1,featuresx2,featuresx4,featuresx8,featuresx16
        print('\nVal loss:' + str(np.mean(eval_loss, axis=0)))
        if (np.mean(eval_loss, axis=0)) < best_eval_loss:
            best_eval_loss = np.mean(eval_loss, axis=0)
            torch.save(encoder_net.state_dict(), './'+checkpoint_save_path+'/best_encoder_net.pkl')
            torch.save(decoder_net.state_dict(), './'+checkpoint_save_path+'/best_decoder_net.pkl')
            print('save best model,best loss:' + str(best_eval_loss))
        else:
            print('evaluate does not improve,best loss:' + str(best_eval_loss))
def test(encoder_net,decoder_net,checkpoint_save_path,test_list,main_data_path,test_out_path):
        workbook=xlwt.Workbook()
        worksheet=workbook.add_sheet('DSC')
        DSC_list=[]
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        encoder_net.load_state_dict(torch.load('./'+checkpoint_save_path+'/best_encoder_net.pkl'))
        decoder_net.load_state_dict(torch.load('./'+checkpoint_save_path+'/best_decoder_net.pkl'))
        encoder_net.eval()
        decoder_net.eval()
        for instance_index,(image_instance,label_instance) in enumerate(test_list):
            use_gpu=torch.cuda.is_available()
            #read image
            image=Image.open(os.path.join(main_data_path,image_instance))
            image=np.asarray(image).astype('float32')
            image=(image/255.-mean)/std
            image=np.transpose(image,(2,0,1))[np.newaxis,:,:,:].astype('float32')
            image=torch.tensor(image)
            if use_gpu:
                image=image.cuda()
            featuresx1,featuresx2,featuresx4,featuresx8,featuresx16=encoder_net(image)
            out=decoder_net(featuresx1,featuresx2,featuresx4,featuresx8,featuresx16)
            out=out.data.cpu().numpy()[0,0]
            binary_out=out>0.5
            out=np.floor(out*255).astype('uint8')
            out=Image.fromarray(out)
            out.save(os.path.join(test_out_path,image_instance.split('/')[-1]))
            
            label=Image.open(os.path.join(main_data_path,label_instance))
            label=(np.asarray(label)[:,:]>(0.3*255)).astype('float32')
            DSC=2*np.sum(label*binary_out)/(np.sum(label)+np.sum(binary_out))
            DSC_list.append(DSC)
            worksheet.write(instance_index,0,image_instance)
            worksheet.write(instance_index,1,DSC)
        DSC_list=np.asarray(DSC_list)
        worksheet.write(instance_index+1,0,'Mean')
        worksheet.write(instance_index+1,1,np.mean(DSC_list))
        worksheet.write(instance_index+2,0,'std')
        worksheet.write(instance_index+2,1,np.std(DSC_list))
        print('mean:'+str(np.mean(DSC_list))+',std:'+str(np.std(DSC_list)))
        workbook.save(os.path.join(test_out_path,'test_DSC.xls'))
        
if __name__=='__main__':
    #######################
    #parameters setting
    #######################
    npy_name_list=[
                        'OD_IDRID.npy',
                        'Vessel_RITE.npy',
                        'CUP_Drishti-GS1.npy',
                        'OD_Drishti-GS1.npy',
                        'Vessel_STARE.npy',
                        'Hard Exudates_IDRID.npy',
                        'Vessel_CHASEDB.npy',
                        'Vessel_HRF.npy',
                        ]#downstream tasks
    pre_trained_models_list=[
                            ['../step1_ld_pretraining/checkpoint_ld_(8, 8)_32_16','ld_(8, 8)_32_16','vgg16_bn_tiny'],
                            ]#test representation models [weight path, model name, backbone architecture]
    gpu='0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    val_rate=0.2
    batch_size=2
    epoch=100
    main_data_path='../../datasets/retinal_images'
    
    ######################
    #train models
    ######################
    for pre_trained_checkpoint_path,pre_trained_model_name,arch in pre_trained_models_list[:]:
        print(pre_trained_model_name)
        for npy_name in npy_name_list[:]:
            segmentation_object=npy_name.split('_')[0]
            dataset_name=npy_name[len(segmentation_object)+1:].split('.')[0]
            checkpoint_save_path=os.path.join(pre_trained_model_name,dataset_name+'_'+segmentation_object)
            check_and_create_folder(checkpoint_save_path+'_frozen')
            check_and_create_folder(checkpoint_save_path+'_fine')
            #######create data loader
            training_list,test_list=np.load(os.path.join(main_data_path,npy_name),allow_pickle=True)
            val_data_list=training_list[:int(val_rate*len(training_list))]
            training_list=training_list[int(val_rate*len(training_list)):]
            training_set=MyDataset(data_list=training_list,
                                data_path=main_data_path,
                                if_augumentation=True)
            training_loader=data.DataLoader(training_set,batch_size=batch_size,shuffle=True,num_workers=0)
            val_set=MyDataset(data_list=val_data_list,
                                data_path=main_data_path,
                                if_augumentation=False)
            val_loader=data.DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=0)
            
            ######build models
            encoder_net=vgg.__dict__[arch]()
            if arch=='vgg16_bn_tiny':
                decoder_net=vgg16_decoder(channel_num=[128,128,64,32,16])
            if arch=='vgg16_bn_small':
                decoder_net=vgg16_decoder(channel_num=[256,256,128,64,32])
            if arch=='vgg16_bn':
                decoder_net=vgg16_decoder(channel_num=[512,512,256,128,64])
            
            ######training frozen
            trainable_nets=[decoder_net]
            frozen_nets=[encoder_net]
            use_gpu=torch.cuda.is_available()
            if use_gpu:
                for net_instance in trainable_nets+frozen_nets:
                    net_instance=net_instance.cuda()
            pre_trained_checkpoint=torch.load(os.path.join(pre_trained_checkpoint_path,'model_best_'+arch+'.pth.tar'))
            encoder_net=load_weights_from_datapallel_new_model(encoder_net,pre_trained_checkpoint['encoder_net'])
            params=[{'params':net_instance.parameters(),'lr':1e-3} for net_instance in trainable_nets]
            optimizer=optim.Adam(params)
            for net_instance in frozen_nets:
                net_instance.eval()
                for param in net_instance.parameters():
                    param.requires_grad = False
            training(training_loader,val_loader,trainable_nets,encoder_net,decoder_net,epoch,optimizer,checkpoint_save_path+'_frozen')
            ######training fine-tune
            encoder_net.load_state_dict(torch.load('./'+checkpoint_save_path+'_frozen/best_encoder_net.pkl'))
            decoder_net.load_state_dict(torch.load('./'+checkpoint_save_path+'_frozen/best_decoder_net.pkl'))
            trainable_nets=[encoder_net,decoder_net]
            frozen_nets=[]
            params=[{'params':net_instance.parameters(),'lr':1e-3} for net_instance in trainable_nets]
            optimizer=optim.Adam(params)
            for net_instance in frozen_nets:
                net_instance.eval()
                for param in net_instance.parameters():
                    param.requires_grad = False
            training(training_loader,val_loader,trainable_nets,encoder_net,decoder_net,epoch,optimizer,checkpoint_save_path+'_fine')
            ######test 
            test_out_path=os.path.join('./output',pre_trained_model_name,'frozen',dataset_name+'_'+segmentation_object)
            check_and_create_folder(test_out_path)
            test(encoder_net,decoder_net,checkpoint_save_path+'_frozen',test_list,main_data_path,test_out_path)
            
            test_out_path=os.path.join('./output',pre_trained_model_name,'fine',dataset_name+'_'+segmentation_object)
            check_and_create_folder(test_out_path)
            test(encoder_net,decoder_net,checkpoint_save_path+'_fine',test_list,main_data_path,test_out_path)
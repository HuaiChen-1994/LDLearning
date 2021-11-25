'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.append('./')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import os
import time
import random
import numpy as np
import models
import shutil
from args_setting import *
from my_utils import MyDataset,AverageMeter
from BatchAverage import BatchCriterion

def adjust_learning_rate(optimizer, lr):
    '''Sets the learning rate to the initial LR'''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr   
    print('setting lr:'+str(lr))  
    
def save_checkpoint(state, is_best, arch,main_path):
    '''save historical checkpoint and best checkpoint'''
    filename=os.path.join(main_path,'checkpoint_'+arch+'_'+str(state['epoch'])+'.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(main_path,'model_best_'+arch+'.pth.tar'))
        torch.save(state['encoder_net'],os.path.join(main_path,'model_best_'+arch+'.pkl'))
        torch.save(state['decoder_net'],os.path.join(main_path,'model_best_decoder_net'+arch+'.pkl'))
        
        
def compute_patch_loss(features,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,img1_weight,args,criterion):
    '''
    Compute the loss_pd and loss_mixup of patch regions.
    '''
    target_mix_feature=torch.zeros((args['batch_size'],features.size()[1],features.size()[2],features.size()[3])).cuda()
    for instance_index in range(len(target_mix_feature)):
        target_mix_feature[instance_index]=img1_weight[instance_index]*features[instance_index]+(1-img1_weight[instance_index])*features[args['batch_size']+instance_index]
    target_mix_feature=target_mix_feature.detach()
    target_mix_feature=models.normalize.l2_normal(target_mix_feature,dim=1)
    features=rearrange_features(features,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,args)
    
    # compute patch loss
    features_input=features[:4*args['batch_size']].permute(0,2,3,1)
    features_input=features_input.reshape((features_input.size()[0]*features_input.size()[1]*features_input.size()[2],features_input.size()[3]))
    loss_pd=criterion(features_input)
    del features_input
    
    # compute mixup loss
    features_input=torch.cat((features[4*args['batch_size']:],target_mix_feature),0).permute(0,2,3,1)
    del features,target_mix_feature
    features_input=features_input.reshape((features_input.size()[0]*features_input.size()[1]*features_input.size()[2],features_input.size()[3]))
    loss_mixup=criterion(features_input)
    return loss_pd,loss_mixup

def rearrange_features(features,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,args):
    '''
    Reorder the features to align featrues of the corresponding similar pixels.
    '''
    for instance_index,(if_flip,rot_index) in enumerate(zip(img1_2_flip,img1_2_rot)):
        if if_flip:
            features[2*args['batch_size']+instance_index]=torch.flip(features[2*args['batch_size']+instance_index],dims=(1,2))
        if rot_index:
            features[2*args['batch_size']+instance_index]=torch.rot90(features[2*args['batch_size']+instance_index],k=-rot_index,dims=(1,2))
            
    for instance_index,(if_flip,rot_index) in enumerate(zip(img2_2_flip,img2_2_rot)):
        if if_flip:
            features[3*args['batch_size']+instance_index]=torch.flip(features[3*args['batch_size']+instance_index],dims=(1,2))
        if rot_index:
            features[3*args['batch_size']+instance_index]=torch.rot90(features[3*args['batch_size']+instance_index],k=-rot_index,dims=(1,2))
    return features

if __name__=='__main__':
    args=args_pd_mixup
    lr=args['lr']
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_loss = np.inf  # best test loss
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    check_point_out_path=args['checkpoint_path']+'_'+str(args['region_output_size'])+'_'+str(args['low_dim'])
    if not os.path.exists(check_point_out_path):
        os.makedirs(check_point_out_path)
    
    #============================================
    # Prepare data generator
    #============================================
    print('==> Preparing data..')
    train_data_list=os.listdir(os.path.join(args['dataset_path']))
    val_data_list=train_data_list[30000:]
    train_data_list=train_data_list[:30000]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    trainset = MyDataset(
                        total_num=1000*args['batch_size'],
                          if_rot_90=True,
                          if_flip=True,
                          if_mixup=True,
                          data_list=train_data_list,
                          data_path=args['dataset_path'],
                          transform_list=[transforms.Compose([
                                                             
                                                              transforms.RandomHorizontalFlip(p=0.5),
                                                              transforms.RandomVerticalFlip(p=0.5),
                                                              transforms.RandomRotation([-180,180]),
                                                              transforms.RandomResizedCrop(size=448, scale=(0.9,1.0)),]),
                                          transforms.Compose([
                                                               transforms.RandomRotation([-2,2]),
                                                              transforms.ColorJitter(0.4, 0.4, 0.4, 0.02), 
                                                              transforms.RandomGrayscale(p=0.2),
                                                              transforms.ToTensor(),
                                                              ]),
                                        transforms.Compose([normalize])]        )
    train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args['batch_size'], shuffle=True,
            num_workers=args['workers'], pin_memory=True)
    
    valset = MyDataset(
                        total_num=100*args['batch_size'],
                          if_rot_90=True,
                          if_flip=True,
                          if_mixup=True,
                          data_list=val_data_list,
                          data_path=args['dataset_path'],
                          transform_list=[transforms.Compose([
                                                              transforms.RandomHorizontalFlip(p=0.5),
                                                              transforms.RandomVerticalFlip(p=0.5),
                                                              transforms.RandomRotation([-180,180]),
                                                              transforms.RandomResizedCrop(size=448, scale=(0.9,1.0)),]),
                                          transforms.Compose([
                                                               transforms.RandomRotation([-2,2]),
                                                              transforms.ColorJitter(0.4, 0.4, 0.4, 0.02), 
                                                              transforms.RandomGrayscale(p=0.2),
                                                              transforms.ToTensor(),
                                                              ]),
                                        transforms.Compose([normalize])]        )
    val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args['batch_size'], shuffle=False,
            num_workers=args['workers'], pin_memory=True) 
    
    
    #===============================================
    #Build models
    #===============================================
    print('==> Building model..')
    encoder_net = models.__dict__[args['arch']]()
    decoder_net=models.my_models.decoder(channel_list=[128,128,64,32,16,args['low_dim']])                
    
    criterion = BatchCriterion(1,0.1)
    if device == 'cuda':
        encoder_net = torch.nn.DataParallel(encoder_net, device_ids=range(torch.cuda.device_count()))
        decoder_net=torch.nn.DataParallel(decoder_net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    encoder_net.to(device)
    criterion.to(device)
    decoder_net.to(device)
    trainable_nets=[encoder_net,decoder_net]
    params=[{'params':net_instance.parameters()} for net_instance in trainable_nets]
    optimizer = optim.Adam(params, lr=lr)
    
    # Load historical weights if needed 
    if len(args['resume'])>0:
        # Load checkpoint.
        model_path =os.path.join(check_point_out_path,args['resume'])
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(model_path)
        encoder_net.load_state_dict(checkpoint['encoder_net'])
        decoder_net.load_state_dict(checkpoint['decoder_net'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr=checkpoint['lr']
        adjust_learning_rate(optimizer, lr)
    
    
    for epoch in range(start_epoch, args['train_epoch']):
        #===========================================================
        #Training 
        #===========================================================
        print('\nEpoch: %d' % epoch)        
        train_loss_pd = AverageMeter()
        train_loss_mixup = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        for net_instance in trainable_nets:
            net_instance.train()
            for param in net_instance.parameters():
                param.requires_grad=True            
        end = time.time()
        
        for batch_idx, (img1_1,img1_2,img2_1,img2_2,img3,img1_weight,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot) in enumerate(train_loader):
            if img1_1.size()[0]!=args['batch_size']:
                continue
            #load data and get patch features
            data_time.update(time.time() - end)
            img1_1,img1_2,img2_1,img2_2,img3=img1_1.to(device),img1_2.to(device),img2_1.to(device),img2_2.to(device),img3.to(device)
            inputs=torch.cat((img1_1,img2_1,img1_2,img2_2,img3),0)
            del img1_1,img1_2,img2_1,img2_2,img3
            embedded_features = decoder_net(encoder_net(inputs))    
            patch_features=nn.functional.adaptive_avg_pool2d(embedded_features,output_size=args['region_output_size'])
            patch_features=models.normalize.l2_normal(patch_features)
            del inputs,embedded_features
        
            #compute loss_pd and loss_mixup
            loss_pd,loss_mixup=compute_patch_loss(patch_features,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,img1_weight,args,criterion)
            del patch_features,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,img1_weight
            
            loss =loss_pd+loss_mixup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_pd.update(loss_pd.item(), args['batch_size']) 
            train_loss_mixup.update(loss_mixup.item(), args['batch_size']) 
            del loss_pd,loss,loss_mixup
     
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx%10 ==0:
                print('\r','Epoch: [{}][{}/{}] '
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Loss_pd: {train_loss_pd.val:.4f} ({train_loss_pd.avg:.4f})'
                      'Loss_mixup: {train_loss_mixup.val:.4f} ({train_loss_mixup.avg:.4f})'.format(
                      epoch, batch_idx, len(train_loader), batch_time=batch_time, 
                      data_time=data_time, train_loss_pd=train_loss_pd,train_loss_mixup=train_loss_mixup),end='')        
                
        #===========================================================
        #Evaluating
        #===========================================================
        print('#########evaluation########')
        val_loss = AverageMeter()
        val_loss_pd = AverageMeter()
        val_loss_mixup = AverageMeter()
        for net_instance in trainable_nets:
            net_instance.eval()
            for param in net_instance.parameters():
                param.requires_grad=False
                
        for batch_idx, (img1_1,img1_2,img2_1,img2_2,img3,img1_weight,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot) in enumerate(val_loader):
            if img1_1.size()[0]!=args['batch_size']:
                continue
            img1_1,img1_2,img2_1,img2_2,img3=img1_1.to(device),img1_2.to(device),img2_1.to(device),img2_2.to(device),img3.to(device)
            inputs=torch.cat((img1_1,img2_1,img1_2,img2_2,img3),0)
            del img1_1,img1_2,img2_1,img2_2,img3
            embedded_features = decoder_net(encoder_net(inputs))    
            patch_features=nn.functional.adaptive_avg_pool2d(embedded_features,output_size=args['region_output_size'])
            patch_features=models.normalize.l2_normal(patch_features)
            del inputs,embedded_features
            
            loss_pd,loss_mixup=compute_patch_loss(patch_features,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,img1_weight,args,criterion)
            del patch_features,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,img1_weight
            
            loss =loss_pd+loss_mixup
            
            val_loss.update(loss.item(), args['batch_size'])
            val_loss_pd.update(loss_pd.item(), args['batch_size'])
            val_loss_mixup.update(loss_mixup.item(), args['batch_size'])
            
            if batch_idx%10 ==0:
                print('\r','Epoch: [{}][{}/{}] '
                      'Loss_pd: {val_loss_pd.val:.4f} ({val_loss_pd.avg:.4f})'
                      'Loss_mixup: {val_loss_mixup.val:.4f} ({val_loss_mixup.avg:.4f})'.format(
                      epoch, batch_idx, len(val_loader), val_loss_pd=val_loss_pd,val_loss_mixup=val_loss_mixup),end='')        
    
        #=====================================
        #Adjust lr
        #=====================================
        if val_loss.avg<best_loss:
            is_best=True
            best_loss=val_loss.avg
        else:
            is_best=False
        if epoch%10==9:
            lr=lr/2.
            adjust_learning_rate(optimizer, lr)
        #=======================================
        #Save models
        #=======================================        
        save_checkpoint({
                'epoch': epoch + 1,
                'encoder_net': encoder_net.state_dict(),
                'decoder_net':decoder_net.state_dict(),
                'best_loss':best_loss,
                'optimizer':optimizer.state_dict(),
                'lr':lr
            }, is_best=is_best,arch=args['arch'],main_path=check_point_out_path)
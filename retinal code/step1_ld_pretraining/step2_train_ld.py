'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.append('./')
import  cv2
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
from my_utils import MyDataset,AverageMeter
from BatchAverage import BatchCriterion
from collections import OrderedDict
from args_setting import *

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

def compute_clustering_loss(region_features,attention_map):
    '''
    Compute loss_ld, loss_entropy and loss_area.
    '''
    mean_features=region_features.mean(dim=0)
    mean_features=models.normalize.l2_normal(mean_features,dim=1)
    #loss ld
    diag_mat=torch.eye(mean_features.size()[0]).cuda()
    cosine_similarity=((region_features[:,:,np.newaxis,:]*mean_features[np.newaxis,np.newaxis,:,:]).sum(3))/0.2#[x,8,8,32] -> [x,8,8] 
    del region_features
    cosine_similarity=(cosine_similarity).exp_()
    cosine_similarity=cosine_similarity/((cosine_similarity.sum(2))[:,:,np.newaxis])
    loss_ld=0
    for image_index in range(len(cosine_similarity)):
        loss_ld=loss_ld-(cosine_similarity[image_index].log_()*diag_mat).sum()/(diag_mat.sum())
    loss_ld=loss_ld/len(cosine_similarity)
    del cosine_similarity
    #loss entropy loss area
    loss_entropy=models.my_models.loss_entropy(attention_map)
    del attention_map
    return loss_ld,loss_entropy

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
def load_weights_2_newmodel(new_model,old_model):
    '''
    Load weights from old model to new model. 
    The new decoder in local discrimination has additional clustering branch compared to decoder in patch discrimination, therefore, only the same part will be initialized by pre-trained model.
    '''
    new_name=[]
    for k,v in new_model.state_dict().items():
        new_name.append(k)
    old_model_weights=[]
    for k,v in old_model.items():
        old_model_weights.append(v)
        
    new_state_dict = OrderedDict()  
    for layer_index in range(len(old_model_weights)):
        new_state_dict[new_name[layer_index]]=old_model_weights[layer_index]
    for layer_index,(k,v) in enumerate(new_model.state_dict().items()):
        if layer_index>=(len(old_model_weights)):
            new_state_dict[new_name[layer_index]]=v
    new_model.load_state_dict(new_state_dict)
    return new_model
if __name__=='__main__':
    args=args_ld
    lr=args['lr']
    no_improve_time=0
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_loss = np.inf  # best test loss
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    check_point_out_path=args['checkpoint_path']+'_'+str(args['region_output_size'])+'_'+str(args['low_dim'])+'_'+str(args['cluster_num'])
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
    decoder_net=models.my_models.decoder(channel_list=[128,128,64,32,16,args['low_dim'],args['cluster_num']])                
    
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
    #load pd weights
    
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args_ld['load_model_path'])
    encoder_net=load_weights_2_newmodel(encoder_net,checkpoint['encoder_net'])
    decoder_net=load_weights_2_newmodel(decoder_net,checkpoint['decoder_net'])
    del checkpoint
    
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
        train_loss_ld = AverageMeter()
        train_loss_entropy = AverageMeter()
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
            patch_features,region_features,attention_map= decoder_net(encoder_net(inputs),args['region_output_size'],if_clustering=True)  
            del inputs
            attention_map=attention_map[:4*args['batch_size']]
            region_features=region_features[:4*args['batch_size']]
            #compute loss_pd and loss_mixup
            loss_pd,loss_mixup=compute_patch_loss(patch_features,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,img1_weight,args,criterion)
            del patch_features 
            
            #compute loss_ld, loss_entropy
            loss_ld, loss_entropy=compute_clustering_loss(region_features,attention_map)
            del img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,img1_weight,region_features,attention_map,
            
            loss =loss_pd+loss_mixup+10*loss_ld+0.1*loss_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_pd.update(loss_pd.item(), args['batch_size']) 
            train_loss_mixup.update(loss_mixup.item(), args['batch_size']) 
            train_loss_ld.update(loss_ld.item(), args['batch_size'])
            train_loss_entropy.update(loss_entropy.item(), args['batch_size'])
            del loss_pd,loss,loss_mixup,loss_ld,loss_entropy
     
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx%10 ==0:
                print('\r','Epoch: [{}][{}/{}] '
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Loss_pd: {train_loss_pd.val:.4f} ({train_loss_pd.avg:.4f})'
                      'Loss_mixup: {train_loss_mixup.val:.4f} ({train_loss_mixup.avg:.4f})'
                      'Loss_ld: {train_loss_ld.val:.4f} ({train_loss_ld.avg:.4f})'
                      'Loss_entropy: {train_loss_entropy.val:.4f} ({train_loss_entropy.avg:.4f})'.format(
                      epoch, batch_idx, len(train_loader), batch_time=batch_time, 
                      data_time=data_time, train_loss_pd=train_loss_pd,train_loss_mixup=train_loss_mixup,
                      train_loss_ld=train_loss_ld,train_loss_entropy=train_loss_entropy),end='')        
                
        #===========================================================
        #Evaluating
        #===========================================================
        print('#########evaluation########')
        val_loss = AverageMeter()
        val_loss_pd = AverageMeter()
        val_loss_mixup = AverageMeter()
        val_loss_ld = AverageMeter()
        val_loss_entropy = AverageMeter()
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
            patch_features,region_features,attention_map= decoder_net(encoder_net(inputs),args['region_output_size'],if_clustering=True)    
            del inputs
            
            attention_map=attention_map[:4*args['batch_size']]
            region_features=region_features[:4*args['batch_size']]
            #compute loss_pd and loss_mixup
            loss_pd,loss_mixup=compute_patch_loss(patch_features,img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,img1_weight,args,criterion)
            del patch_features
            
            #compute loss_ld, loss_entropy, loss_area
            loss_ld, loss_entropy=compute_clustering_loss(region_features,attention_map)
            del img1_2_flip,img1_2_rot,img2_2_flip,img2_2_rot,img1_weight,region_features,attention_map,
            
            loss =loss_pd+loss_mixup+10*loss_ld+0.1*loss_entropy
            
            val_loss.update(loss.item(), args['batch_size'])
            val_loss_pd.update(loss_pd.item(), args['batch_size'])
            val_loss_mixup.update(loss_mixup.item(), args['batch_size'])
            val_loss_ld.update(loss_ld.item(), args['batch_size'])
            val_loss_entropy.update(loss_entropy.item(), args['batch_size'])
            if batch_idx%10 ==0:
                print('\r','Epoch: [{}][{}/{}] '
                      'Loss_pd: {val_loss_pd.val:.4f} ({val_loss_pd.avg:.4f})'
                      'Loss_mixup: {val_loss_mixup.val:.4f} ({val_loss_mixup.avg:.4f})'
                      'Loss_ld: {val_loss_ld.val:.4f} ({val_loss_ld.avg:.4f})'
                      'Loss_entropy: {val_loss_entropy.val:.4f} ({val_loss_entropy.avg:.4f})'.format(
                      epoch, batch_idx, len(val_loader), 
                      val_loss_pd=val_loss_pd,val_loss_mixup=val_loss_mixup,
                      val_loss_ld=val_loss_ld,val_loss_entropy=val_loss_entropy
                      ),end='')            
        
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
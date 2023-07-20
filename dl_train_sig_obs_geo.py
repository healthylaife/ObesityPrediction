#!/usr/bin/env python
# coding: utf-8

import pickle
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
import numpy as np
import torch as T
import torch
import math
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
import torch.nn as nn
from torch import optim
import importlib
import torch.nn.functional as F
import import_ipynb
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
import evaluation
import callibrate_output
import fairness
import parameters
from parameters import *
#import model as model
import mimic_model_sig_obs as model
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from pickle import dump,load
from sklearn.model_selection import train_test_split
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution,LayerDeepLift,DeepLift

#import torchvision.utils as utils
import argparse
from torch.autograd import Variable
from argparse import ArgumentParser
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

#save_path = "saved_models/model.tar"
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")


importlib.reload(model)
import mimic_model_sig_obs as model
importlib.reload(parameters)
import parameters
from parameters import *
importlib.reload(evaluation)
import evaluation
importlib.reload(fairness)
import fairness


class DL_models():
    def __init__(self,model_name,train,feat_vocab_size,age_vocab_size,demo_vocab_size,obs):
        
        self.model_name=model_name
        self.feat_vocab_size=feat_vocab_size
        self.age_vocab_size=age_vocab_size
        self.demo_vocab_size=demo_vocab_size
        self.obs=obs
        self.save_path="saved_models/"+model_name+"_"+str(self.obs)
        if torch.cuda.is_available():
            self.device='cuda:0'
        else:
            self.device='cpu'
        self.device='cpu'
        #print(self.device)
        self.loss=evaluation.Loss(self.device,True,False,False,False,False,False,True,True,True,True,True,True)
        if train:
            print("===============MODEL TRAINING===============")
            
            self.dl_train()
            
        else:
            print("===============MODEL TESTING===============")
            
            self.dl_test()
            
        
        
        
        
    def dl_test(self):
        #test_ids=pd.read_csv('./data/4/tt/test_id.csv',header=0)
        test_ids=pd.read_csv('./data/geo/test_id.csv',header=0)
        for i in range(1):
            print("==================={0:2d} FOLD=====================".format(i))
            path=self.save_path+".tar"
            self.net=torch.load(path)
            print("[ MODEL LOADED ]")
            print(self.net)
            
#             print("TESTING BATHC 1")
            test_hids=list(test_ids['person_id'].unique())
            self.tp=0
            self.fp=0
            self.fn=0
            self.tn=0
            self.tp_non=0
            self.tp_obese=0
            self.fp_non=0
            self.fp_obese=0
            
            #self.save_dependance_meas(test_hids,i)
            #self.read_dependance_meas()
            #self.model_test_full(test_hids,i)
#             print(self.tp)
#             print(self.fp)
#             print(self.fn)
#             print(self.tn)
#             print(self.tp_non)
#             print(self.tp_obese)
#             print(self.fp_non)
#             print(self.fp_obese)
            #self.interpret_read()
            self.model_test(test_hids,i)
            self.test_read(i)
            #self.interpret_embed()
            #self.model_test_full(test_hids[300:310],i)
    

            

    def create_kfolds(self):
    
#         train_ids=pd.read_csv('./data/4/tt/train_id.csv',header=0)
#         test_ids=pd.read_csv('./data/4/tt/test_id.csv',header=0)
        train_ids=pd.read_csv('./data/geo/train_id.csv',header=0)
        test_ids=pd.read_csv('./data/geo/test_id.csv',header=0)
        return list(train_ids['person_id'].unique()),list(test_ids['person_id'].unique())
    
    def dl_train(self):
        #norm_hids,over_hids,ob_hids,test_hids=self.create_kfolds()
        train_hids,test_hids=self.create_kfolds()
#         print(len(k_hids))
#         print(k_hids[0].shape)
              
        for i in range(1):
            self.create_model()
            print("[ MODEL CREATED ]")
            print(self.net)
            path=self.save_path+".tar"

            print("==================={0:2d} FOLD=====================".format(i))

            val_hids=random.sample(train_hids,int(len(train_hids)*0.05))
            train_hids=list(set(train_hids)-set(val_hids))
            with open('./saved_models/'+'val_hids_sig_'+str(i), 'wb') as fp:
                  pickle.dump(val_hids, fp)
            #train_hids=list(set(train_hids)-set(val_hids))
            min_loss=100
            counter=0
            for epoch in range(args.num_epochs):
                if counter==args.patience:
                    print("STOPPING THE TRAINING BECAUSE VALIDATION ERROR DID NOT IMPROVE FOR {:.1f} EPOCHS".format(args.patience))
                    break
                
                train_loss=0.0
                self.net.train()
            
                print("======= EPOCH {:.1f} ========".format(epoch))
                for nbatch in range(int(len(train_hids)/(args.batch_size))):
                    #print(len(train_hids))
                    enc_feat,enc_len,enc_age,enc_demo,_=self.encXY(train_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size],train_data=True)
                    dec_feat,dec_labels,mask=self.decXY(train_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size],train_data=True)
                    
                    output,prob,batch_loss = self.train_model(enc_feat,enc_len,enc_age,enc_demo,dec_feat,dec_labels,mask)
                    
                    train_loss+=batch_loss

                train_loss=train_loss/(nbatch+1)        
                print("Total Train Loss: ", train_loss)
                #print("Done training")
                val_loss1=self.model_val(val_hids[0:200])
                #print("Done val1")
                val_loss2=self.model_val(val_hids[200:400])
                val_loss3=self.model_val(val_hids[400:600])
                val_loss4=self.model_val(val_hids[600:800])
                val_loss5=self.model_val(val_hids[800:1000])
                val_loss6=self.model_val(val_hids[1000:1200])
                val_loss7=self.model_val(val_hids[1200:])
             
                val_loss=(val_loss1+val_loss2+val_loss3+val_loss4+val_loss5+val_loss6+val_loss7)/7
            
                print("Total Val Loss: ", val_loss)
               

                if(val_loss<=min_loss+0.01):
                    print("Validation results improved")
                    print("Updating Model")
                    T.save(self.net,path)
                    min_loss=val_loss
                    counter=0
                else:
                    print("No improvement in Validation results")
                    counter=counter+1
 
    
    def train_model(self,enc_feat,enc_len,enc_age,enc_demo,dec_feat,dec_labels,mask):
        self.optimizer.zero_grad()
        obs=self.obs
        pred_mask=np.zeros((mask.shape[0],mask.shape[1]))
        #print(mask.shape)
        if obs>0:
            pred_mask[:,0:obs]=mask[:,0:obs]#mask right
#             print(pred_mask[2])
        pred_mask=torch.tensor(pred_mask)
        pred_mask=pred_mask.type(torch.LongTensor)
        #dec_feat
        #print(dec_labels.shape)
        #print(pred_mask.shape)
        dec_labels_pred=dec_labels*pred_mask#hide output labels
        #print(dec_labels_pred)
        pred_mask_feat=pred_mask.unsqueeze(2)
        pred_mask_feat=pred_mask_feat.repeat(1,1,dec_feat.shape[2])
        pred_mask_feat=pred_mask_feat.type(torch.DoubleTensor)
#             print(pred_mask_feat.shape)
#             print(pred_mask_feat[2])
#             print(dec_feat[7,:,:])

        dec_feat_pred=dec_feat*pred_mask_feat#hide future features
#             print(dec_feat[7,:,:])
        if obs>0:
#                 print(pred_mask.shape)
            obs_idx=pred_mask[:,obs-1]#take last entry before prediction window
#                 print(obs_idx.shape)
            #obs_idx=torch.add(obs_idx,-obs)
            #print("obs_idx",obs_idx.shape)
            obs_idx=torch.nonzero(obs_idx>0)
            #print("obs_idx",obs_idx.shape)
            obs_idx=obs_idx.squeeze()
            dec_feat_pred=dec_feat_pred[obs_idx]
            dec_labels_pred=dec_labels_pred[obs_idx]
            pred_mask=pred_mask[obs_idx]
            mask=mask[obs_idx]
            dec_labels=dec_labels[obs_idx]
            
            enc_feat,enc_len,enc_age,enc_demo,enc_ob=enc_feat[obs_idx],enc_len[obs_idx],enc_age[obs_idx],enc_demo[obs_idx],enc_ob[obs_idx]

        output,prob,disc_input,_ = self.net(False,False,enc_feat,enc_len,enc_age,enc_demo,dec_feat_pred,dec_labels_pred,pred_mask) 


            

        
        

        out_loss=0.0
        x_loss=0.0
        
        obs_mask=np.zeros((mask.shape[0],mask.shape[1]))
        obs_mask[:,obs:]=mask[:,obs:]#mask left side
        obs_mask=torch.tensor(obs_mask)
        obs_mask=obs_mask.type(torch.LongTensor)
#         print(prob[0][0:5])#[200,2]
#         print(mask[0:5,:])#[200,8]
        for i in range(obs,len(prob)):#time
#             print("================================")
#             print(mask[0:10])
            idx=torch.nonzero(obs_mask[:,i]>0)
            idx=idx.squeeze()
            #print(idx)
            m=obs_mask[idx]
            m_prob=prob[i][idx]
            m_output=output[i][idx]

            dec_labels_dec=dec_labels[idx]
#             print(dec_labels_dec[:,i])
            dec_labels_dec=torch.tensor(dec_labels_dec[:,i])
  
            dec_labels_dec=dec_labels_dec.type(torch.FloatTensor)
            dec_labels_dec=dec_labels_dec.to(self.device)
            m_prob=m_prob[:,1].type(torch.FloatTensor)
            
            out_loss+=(self.criterion(m_prob,dec_labels_dec))/(m[:,i].sum()+ 1e-5)
#             
        out_loss=out_loss/len(prob)

        total_loss=out_loss

        total_loss.backward()

        self.optimizer.step()
        
        return output,prob,total_loss
    
    def model_val(self,val_hids):
        #print("======= VALIDATION ========")
        
        val_prob=[]
        val_truth=[]
        val_logits=[]
        val_loss=0.0
        
        self.net.eval()
        

        enc_feat,enc_len,enc_age,enc_demo,_=self.encXY(val_hids,train_data=True)
        dec_feat,dec_labels,mask=self.decXY(val_hids,train_data=True)
        
        
        obs=self.obs
        pred_mask=np.zeros((mask.shape[0],mask.shape[1]))
        if obs>0:
            pred_mask[:,0:obs]=mask[:,0:obs]#mask right
#             print(pred_mask[2])
        pred_mask=torch.tensor(pred_mask)
        pred_mask=pred_mask.type(torch.LongTensor)
        #dec_feat
        #print(dec_labels)
        dec_labels_pred=dec_labels*pred_mask#hide output labels
        #print(dec_labels_pred)
        pred_mask_feat=pred_mask.unsqueeze(2)
        pred_mask_feat=pred_mask_feat.repeat(1,1,dec_feat.shape[2])
        pred_mask_feat=pred_mask_feat.type(torch.DoubleTensor)
#             print(pred_mask_feat.shape)
#             print(pred_mask_feat[2])
#             print(dec_feat[7,:,:])

        dec_feat_pred=dec_feat*pred_mask_feat#hide future features
#             print(dec_feat[7,:,:])
        if obs>0:
#                 print(pred_mask.shape)
            obs_idx=pred_mask[:,obs-1]#take last entry before prediction window
#                 print(obs_idx.shape)
            #obs_idx=torch.add(obs_idx,-obs)
            #print("obs_idx",obs_idx.shape)
            obs_idx=torch.nonzero(obs_idx>0)
            #print("obs_idx",obs_idx.shape)
            obs_idx=obs_idx.squeeze()
            dec_feat_pred=dec_feat_pred[obs_idx]
            dec_labels_pred=dec_labels_pred[obs_idx]
            pred_mask=pred_mask[obs_idx]
            mask=mask[obs_idx]
            dec_labels=dec_labels[obs_idx]
            
            enc_feat,enc_len,enc_age,enc_demo,enc_ob=enc_feat[obs_idx],enc_len[obs_idx],enc_age[obs_idx],enc_demo[obs_idx],enc_ob[obs_idx]

        output,prob,disc_input,_ = self.net(False,False,enc_feat,enc_len,enc_age,enc_demo,dec_feat_pred,dec_labels_pred,pred_mask) 


            

        
        

        out_loss=0.0
        x_loss=0.0
        
        obs_mask=np.zeros((mask.shape[0],mask.shape[1]))
        obs_mask[:,obs:]=mask[:,obs:]#mask left side
        obs_mask=torch.tensor(obs_mask)
        obs_mask=obs_mask.type(torch.LongTensor)
#         print(prob[0][0:5])#[200,2]
#         print(mask[0:5,:])#[200,8]
        for i in range(obs,len(prob)):#time
#             print("================================")
#             print(mask[0:10])
            idx=torch.nonzero(obs_mask[:,i]>0)
            idx=idx.squeeze()
            #print(idx)
            m=obs_mask[idx]
            m_prob=prob[i][idx]
            m_output=output[i][idx]

            dec_labels_dec=dec_labels[idx]
#             print(dec_labels_dec[:,i])
            dec_labels_dec=torch.tensor(dec_labels_dec[:,i])
            dec_labels_dec=dec_labels_dec.type(torch.FloatTensor)
            dec_labels_dec=dec_labels_dec.to(self.device)
            m_prob=m_prob[:,1].type(torch.FloatTensor)
            
            out_loss+=(self.criterion(m_prob,dec_labels_dec))/(m[:,i].sum()+ 1e-5)
#             
        out_loss=out_loss/len(prob)

        val_loss=out_loss
                    
        return val_loss.item()
    
    def model_test_full(self,total_test_hids,k):
        print("======= TESTING ========")
        
        val_prob=[]
        val_truth=[]
        val_logits=[]
        val_loss=0.0
        
        self.net.eval()
     
        n_batches=int(len(total_test_hids)/(args.batch_size))
        
        for nbatch in range(n_batches):
            print("==================={0:2d} BATCH=====================".format(nbatch))
            test_hids=total_test_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size]
            
            enc_feat,enc_len,enc_age,enc_demo,enc_ob=self.encXY(test_hids,train_data=False)
            dec_feat,dec_labels,mask=self.decXY(test_hids,train_data=False)
            
            obs=self.obs
            pred_mask=np.zeros((mask.shape[0],mask.shape[1]))
            if obs>0:
                pred_mask[:,0:obs]=mask[:,0:obs]#mask right
    #             print(pred_mask[2])
            pred_mask=torch.tensor(pred_mask)
            pred_mask=pred_mask.type(torch.LongTensor)
            #dec_feat
            #print(dec_labels)
            dec_labels_pred=dec_labels*pred_mask#hide output labels
            #print(dec_labels_pred)
            pred_mask_feat=pred_mask.unsqueeze(2)
            pred_mask_feat=pred_mask_feat.repeat(1,1,dec_feat.shape[2])
            pred_mask_feat=pred_mask_feat.type(torch.DoubleTensor)
    #             print(pred_mask_feat.shape)
    #             print(pred_mask_feat[2])
    #             print(dec_feat[7,:,:])

            dec_feat_pred=dec_feat*pred_mask_feat#hide future features
    #             print(dec_feat[7,:,:])
            if obs>0:
    #                 print(pred_mask.shape)
                obs_idx=pred_mask[:,obs-1]#take last entry before prediction window
    #                 print(obs_idx.shape)
                #obs_idx=torch.add(obs_idx,-obs)
                #print("obs_idx",obs_idx.shape)
                obs_idx=torch.nonzero(obs_idx>0)
                #print("obs_idx",obs_idx.shape)
                obs_idx=obs_idx.squeeze()
                dec_feat_pred=dec_feat_pred[obs_idx]
                dec_labels_pred=dec_labels_pred[obs_idx]
                pred_mask=pred_mask[obs_idx]
                mask=mask[obs_idx]
                dec_labels=dec_labels[obs_idx]

                enc_feat,enc_len,enc_age,enc_demo,enc_ob=enc_feat[obs_idx],enc_len[obs_idx],enc_age[obs_idx],enc_demo[obs_idx],enc_ob[obs_idx]

            output,prob,disc_input,contri = self.net(False,True,enc_feat,enc_len,enc_age,enc_demo,dec_feat_pred,dec_labels_pred,pred_mask) 
            
            obs_mask=np.zeros((mask.shape[0],mask.shape[1]))
            obs_mask[:,obs:]=mask[:,obs:]#mask left side
            obs_mask=torch.tensor(obs_mask)
            obs_mask=obs_mask.type(torch.LongTensor)
    #         print(prob[0][0:5])#[200,2]
    #         print(mask[0:5,:])#[200,8]
            i=2
    #             print("================================")
    #             print(mask[0:10])
            idx=torch.nonzero(obs_mask[:,i]>0)
            idx=idx.squeeze()
            #print(idx)
            m=obs_mask[idx]
            m_prob=prob[i][idx]
            m_output=output[i][idx]

            dec_labels_dec=dec_labels[idx]
#             print(dec_labels_dec[:,i])
            #dec_labels_dec=torch.tensor(dec_labels_dec[:,i])
            dec_labels_dec=dec_labels_dec.type(torch.FloatTensor)
            dec_labels_dec=dec_labels_dec.to(self.device)
            
            disc_input=disc_input[idx]
#             print(dec_labels_dec[:,i])
            #disc_input=torch.tensor(disc_input[:,i])
            disc_input=disc_input.type(torch.FloatTensor)
            disc_input=disc_input.to(self.device)
            
            enc_ob=enc_ob[idx]
#             print(dec_labels_dec[:,i])
            
            enc_ob=enc_ob.to(self.device)
            
            m_prob=m_prob[:,1].type(torch.FloatTensor)
        
#             print(dec_labels_dec.shape)
#             print(m_prob.shape)
#             print(disc_input.shape)
#             print(enc_ob.shape)
    
    
            #output,prob,disc_input,contri = self.net(False,True,enc_feat,enc_len,enc_age,enc_demo,dec_feat,dec_labels,mask) 
    
            contri=contri.cpu().detach().numpy()
            contri=contri[idx]
            print(contri.shape)
            self.interpret(contri,dec_labels_dec,disc_input,m_prob,nbatch,enc_ob.data.cpu().numpy())
            #self.interpret(contri[3],dec_labels,disc_input)
        

    
        
            
        #print(contri[0,:,0:2])
    def model_test(self,total_test_hids,k):
        print("======= TESTING ========")
        
        n_batches=int(len(total_test_hids)/(args.batch_size))
        self.n_batches=n_batches
#         print(n_batches)
#         n_batches=2
        total_auroc_mat=np.zeros((8,8,args.labels))
        total_auprc_mat=np.zeros((8,8,args.labels))
        total_auprc_base=np.zeros((8,8,args.labels))
        total_samples=np.zeros((8,8,args.labels))
        total_curve=np.zeros((8,8,args.labels,9))
        total_all_curve=np.zeros((8,8,args.labels,9))
        
        
        
        for nbatch in range(n_batches):
            print("==================={0:2d} BATCH=====================".format(nbatch))
            test_hids=total_test_hids[nbatch*args.batch_size:(nbatch+1)*args.batch_size]
            
            
        
            test_loss=0.0

            self.net.eval()

    #         print(dec_feat.shape)
    #         print(dec_labels.shape)
    #         print(mask.shape)
            
            for obs in range(self.obs+1):
#                 print("======= OBS {:.1f} ========".format(obs+2))       
                enc_feat,enc_len,enc_age,enc_demo,enc_ob=self.encXY(test_hids,train_data=False)
                dec_feat,dec_labels,mask=self.decXY(test_hids,train_data=False)
                

                pred_mask=np.zeros((mask.shape[0],mask.shape[1]))

                if obs>0:
                    pred_mask[:,0:obs]=mask[:,0:obs]#mask right
    #             print(pred_mask[2])
                pred_mask=torch.tensor(pred_mask)
                pred_mask=pred_mask.type(torch.DoubleTensor)
                #dec_feat
                
                dec_labels_pred=dec_labels*pred_mask
                #print(dec_labels_pred)
                pred_mask_feat=pred_mask.unsqueeze(2)
                pred_mask_feat=pred_mask_feat.repeat(1,1,dec_feat.shape[2])
                pred_mask_feat=pred_mask_feat.type(torch.DoubleTensor)
    #             print(pred_mask_feat.shape)
    #             print(pred_mask_feat[2])
    #             print(dec_feat[7,:,:])

                dec_feat_pred=dec_feat*pred_mask_feat
    #             print(dec_feat[7,:,:])
                if obs>0:
                    obs_idx=pred_mask[:,obs-1]#take last entry before prediction window
                    obs_idx=torch.nonzero(obs_idx>0)
                    #print("obs_idx",obs_idx.shape)
                    obs_idx=obs_idx.squeeze()
                    dec_feat_pred=dec_feat_pred[obs_idx]
                    dec_labels_pred=dec_labels_pred[obs_idx]
                    pred_mask=pred_mask[obs_idx]
                    mask=mask[obs_idx]
                    dec_labels=dec_labels[obs_idx]
                    enc_feat,enc_len,enc_age,enc_demo,enc_ob=enc_feat[obs_idx],enc_len[obs_idx],enc_age[obs_idx],enc_demo[obs_idx],enc_ob[obs_idx]
                    #print(enc_demo)


    #             print("dec_feat",dec_feat_pred[0,0:2,:])
    #             print("dec_labels",dec_labels_pred[0])    
                output,prob,disc_input,logits = self.net(False,False,enc_feat,enc_len,enc_age,enc_demo,dec_feat_pred,dec_labels_pred,pred_mask)
   
                mask=np.asarray(mask)

                #print(mask.shape)

                auroc_mat=np.zeros((8,args.labels))
                auprc_mat=np.zeros((8,args.labels))
                auprc_base=np.zeros((8,args.labels))
                n_samples=np.zeros((8,args.labels))
                
                
                obs_mask=np.zeros((mask.shape[0],mask.shape[1]))
                obs_mask[:,obs:]=mask[:,obs:]#mask left side
                #print(enc_ob.shape)
                eth,race,gender,payer=enc_demo[:,0],enc_demo[:,1],enc_demo[:,2],enc_demo[:,3]
                eth,race,gender,payer,ob_stat=np.asarray(eth),np.asarray(race),np.asarray(gender),np.asarray(payer),np.asarray(enc_ob)
                    
                for i in range(obs,len(prob)):#time
                    dec_labels_dec=torch.tensor(dec_labels[:,i])
                    m=obs_mask[:,i]
                    #print(m.shape)
                    idx=list(np.where(m == 0)[0])
                    #print(len(idx))
                    logits_dec=logits[:,i,:]
                    logits_dec=logits_dec.squeeze()
                    for l in range(args.labels):#class
                        dec_labels_l=[1 if y==l else 0 for y in dec_labels_dec]
                        #print("dec_labels_l",len(dec_labels_l))
                        #print("=========================================================")
                        prob_l=prob[i][:,l].data.cpu().numpy()
                        #print(logits_dec.shape)
#                         if l==1 or l==0:
#                             logits_l=logits_dec[:,0].data.cpu().numpy()
#                         elif l==2:
#                             logits_l=logits_dec[:,1].data.cpu().numpy()
                        logits_l=logits_dec.data.cpu().numpy()
                        #print(len(prob_l))
                        #print(len(dec_labels_l))
                        
                        prob_l,dec_labels_l,logits_l=np.asarray(prob_l),np.asarray(dec_labels_l),np.asarray(logits_l)
                        prob_l=np.delete(prob_l, idx)
                        dec_labels_l=np.delete(dec_labels_l, idx)
                        logits_l=np.delete(logits_l, idx)
                        race_l=np.delete(race, idx)
                        eth_l=np.delete(eth, idx)
                        gender_l=np.delete(gender, idx)
                        payer_l=np.delete(payer, idx)
                        ob_l=np.delete(ob_stat, idx)
#                         print("prob",prob_l.shape)
#                         print("dec",dec_labels_l.shape)
#                         print("logits_l",logits_l.shape)
                        n_samples[i,l]=prob_l.shape[0]
                        
                        path='data/outputgeo/'+str(k)+'/'+str(obs)+'/'+str(i)+'/'+str(l)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        with open('./'+path+'/'+'ob_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(ob_l, fp)
                        with open('./'+path+'/'+'race_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(race_l, fp)
                        with open('./'+path+'/'+'eth_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(eth_l, fp)
                        with open('./'+path+'/'+'gender_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(gender_l, fp)
                        with open('./'+path+'/'+'payer_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(payer_l, fp)
                        with open('./'+path+'/'+'logits_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(logits_l, fp)                                
                        with open('./'+path+'/'+'prob_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(prob_l, fp)
                        with open('./'+path+'/'+'dec_list'+'_'+str(nbatch), 'wb') as fp:
                                   pickle.dump(dec_labels_l, fp)
                        

        
        


        
    
    def test_read(self,k):
        print("======= TESTING ========")
        for obs in range(self.obs+1):
#                 print("======= OBS {:.1f} ========".format(obs+2))
            
            auroc_mat=np.zeros((8,args.labels))
            auprc_mat=np.zeros((8,args.labels))
            auprc_base=np.zeros((8,args.labels))
            micro=np.zeros((8,args.labels-1))
            
            for i in range(obs,8):#time

                for l in range(args.labels):#class
                    prob_l=[]
                    race_l=[]
                    ob_l=[]
                    eth_l=[]
                    gender_l=[]
                    payer_l=[]
                    logits_l=[]
                    dec_labels_l=[]
                    for nbatch in range(15):#45
                        path='data/outputgeo/'+str(k)+'/'+str(obs)+'/'+str(i)+'/'+str(l)
                        
                        with open('./'+path+'/'+'ob_list'+'_'+str(nbatch), 'rb') as fp:
                                   ob_l.extend(pickle.load(fp))
                        with open('./'+path+'/'+'race_list'+'_'+str(nbatch), 'rb') as fp:
                                   race_l.extend(pickle.load(fp))
                        with open('./'+path+'/'+'eth_list'+'_'+str(nbatch), 'rb') as fp:
                                   eth_l.extend(pickle.load(fp))
                        with open('./'+path+'/'+'gender_list'+'_'+str(nbatch), 'rb') as fp:
                                   gender_l.extend(pickle.load(fp))
                        with open('./'+path+'/'+'payer_list'+'_'+str(nbatch), 'rb') as fp:
                                   payer_l.extend(pickle.load(fp))        
                        with open('./'+path+'/'+'logits_list'+'_'+str(nbatch), 'rb') as fp:
                                   logits_l.extend(pickle.load(fp))
                                
                        with open('./'+path+'/'+'prob_list'+'_'+str(nbatch), 'rb') as fp:
                                   prob_l.extend(pickle.load(fp))
                        with open('./'+path+'/'+'dec_list'+'_'+str(nbatch), 'rb') as fp:
                                   dec_labels_l.extend(pickle.load(fp))
                    #print(prob_l)
                    
                    
                    prob_l=np.asarray(prob_l)
                    logits_l=np.asarray(logits_l)
                    dec_labels_l=np.asarray(dec_labels_l)
                    ob_l=np.asarray(ob_l)
                    if l>0:
                        micro[i,l-1]=dec_labels_l.sum()
                    
                    if len(prob_l)>0 and l>0:
#                         print(ob_l)                        
#                         ob_l[ob_l==0]=20
#                         ob_l[ob_l==1]=50
#                         ob_l[ob_l==2]=80
#                         ob_l=ob_l/100
#                         ob_l[ob_l==1]=0
#                         ob_l[ob_l==2]=1
#                         print(ob_l)
                        auroc_mat[i,l],auprc_mat[i,l],auprc_base[i,l]=self.loss(prob_l,dec_labels_l,logits_l,ob_l)   
                        #fairness.fairness_evaluation(race_l,eth_l,gender_l,ob_l,payer_l,prob_l,dec_labels_l)
                    else:
                        auroc_mat[i,l],auprc_mat[i,l],auprc_base[i,l]=np.nan,np.nan,np.nan
                        
                    #print(curve)
#                     plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],curve,color='blue',marker='o')
#                     plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],all_curve,color='red',marker='o')
#                     plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0,0,0,0,0,0,0,0,0],color='red',marker='o')
#                     plt.show()
            self.display_output(auroc_mat,auprc_mat,auprc_base,obs,k,micro,False)
            

       
            
#             if time==0:
#                 print(total_curve[time,2,9])

#                 plt.show()
#             print(np.mean(auroc_mat,axis=1))
#             print(np.mean(auprc_mat,axis=1))
#             print(np.mean(auprc_base,axis=1))   
        
        
    def encXY(self,ids,train_data):

        if train_data:
            enc=pd.read_csv('./data/geo/enc_train.csv',header=0)
            enc_len=pd.read_csv('./data/geo/enc_len_train.csv',header=0)
            demo=pd.read_csv('./data/geo/demo_train.csv',header=0)  
            enc_ob=pd.read_csv('./data/geo/bmi_2_train.csv',header=0)
        else:
            enc=pd.read_csv('./data/geo/enc_test.csv',header=0)
            enc_len=pd.read_csv('./data/geo/enc_len_test.csv',header=0)
            demo=pd.read_csv('./data/geo/demo_test.csv',header=0)
            enc_ob=pd.read_csv('./data/geo/bmi_2_test.csv',header=0)
        
        enc=enc[enc['person_id'].isin(ids)]
        demo=demo[demo['person_id'].isin(ids)]
        enc_len=enc_len[enc_len['index'].isin(ids)]
        enc_ob=enc_ob[enc_ob['person_id'].isin(ids)]
        
        enc_ob.loc[enc_ob.value=='Normal','label']=0
        enc_ob.loc[enc_ob.value=='Obesity','label']=2
        enc_ob.loc[enc_ob.value=='Overweight','label']=1
        
    
        
        enc_feat=enc['feat_dict'].values
        #print(enc_feat.shape)
        enc_eth=demo['Eth_dict'].values
        enc_race=demo['Race_dict'].values
        enc_sex=demo['Sex_dict'].values
        enc_payer=demo['Payer_dict'].values
        enc_coi=demo['COI_dict'].values
        enc_len=enc_len['person_id'].values
        enc_ob=enc_ob['label'].values
        
        enc_age=enc['age_dict'].values

               
        #Reshape to 3-D
        #print(enc_feat.shape)
        enc_feat=torch.tensor(enc_feat)
        enc_feat=torch.reshape(enc_feat,(len(ids),-1))
        enc_feat=enc_feat.type(torch.LongTensor)
        
        enc_len=torch.tensor(enc_len)
        #enc_len=torch.reshape(enc_len,(len(ids),-1))
        enc_len=enc_len.type(torch.LongTensor)
        
        enc_ob=torch.tensor(enc_ob)
        enc_ob=enc_ob.type(torch.LongTensor)
        
        enc_age=torch.tensor(enc_age)
        enc_age=torch.reshape(enc_age,(len(ids),-1))
        enc_age=enc_age.type(torch.LongTensor)
        
        enc_eth=torch.tensor(enc_eth)
        enc_eth=enc_eth.unsqueeze(1)
        enc_race=torch.tensor(enc_race)
        enc_race=enc_race.unsqueeze(1)
        enc_sex=torch.tensor(enc_sex)
        enc_sex=enc_sex.unsqueeze(1)
        enc_payer=torch.tensor(enc_payer)
        enc_payer=enc_payer.unsqueeze(1)
        enc_coi=torch.tensor(enc_coi)
        enc_coi=enc_coi.unsqueeze(1)
        #print(enc_eth.shape)
        #print(enc_sex)
        enc_demo=torch.cat((enc_eth,enc_race),1)
        enc_demo=torch.cat((enc_demo,enc_sex),1)
        enc_demo=torch.cat((enc_demo,enc_payer),1)
        enc_demo=torch.cat((enc_demo,enc_coi),1)
        enc_demo=enc_demo.type(torch.LongTensor)
        #print(enc_demo.shape)
        #print(enc_demo)

        return enc_feat,enc_len, enc_age, enc_demo,enc_ob
            
           
    def decXY(self,ids,train_data):
        #print("decoder")
        if train_data:
#             dec=pd.read_csv('./data/5/tt/dec_train.csv',header=0)
#             labels=pd.read_csv('./data/4/tt/labels_train.csv',header=0)    
#             mask=pd.read_csv('./data/4/tt/mask_train.csv',header=0)  
            dec=pd.read_csv('./data/geo/dec_train.csv',header=0)
            labels=pd.read_csv('./data/geo/labels_train.csv',header=0)    
            mask=pd.read_csv('./data/geo/mask_train.csv',header=0)  
            
        else:
#             dec=pd.read_csv('./data/5/tt/dec_test.csv',header=0)
#             labels=pd.read_csv('./data/4/tt/labels_test.csv',header=0)
#             mask=pd.read_csv('./data/4/tt/mask_test.csv',header=0) 
            dec=pd.read_csv('./data/geo/dec_test.csv',header=0)
            labels=pd.read_csv('./data/geo/labels_test.csv',header=0)
            mask=pd.read_csv('./data/geo/mask_test.csv',header=0) 
        
        dec=dec.fillna(0)
        dec = dec.apply(pd.to_numeric)
        del dec['age_dict']
        dec=dec[dec['person_id'].isin(ids)]
        labels=labels[labels['person_id'].isin(ids)]
        mask=mask[mask['person_id'].isin(ids)]
        
        dec_feat=dec.iloc[:,2:].values
        #print(list(dec['person_id']))
        
        dec_labels=labels['value'].values
        mask=mask['value'].values

        
        #Reshape to 3-D
#         print(dec_feat.shape)
        dec_feat=torch.tensor(dec_feat)
        dec_feat=torch.reshape(dec_feat,(len(ids),8,dec_feat.shape[1]))
        
        dec_labels=torch.tensor(dec_labels)
        dec_labels=torch.reshape(dec_labels,(len(ids),-1))
        
        mask=torch.tensor(mask)
        mask=torch.reshape(mask,(len(ids),-1))
        
        #print(dec_feat.shape)
        #print(dec_feat[0:5])
#         print(dec_labels.shape)
#         print(dec_labels)
        
#         print(mask)
        return dec_feat,dec_labels,mask
    
    
    
    def create_model(self):
        self.net = model.EncDec2(self.device,
                           self.feat_vocab_size,
                           self.age_vocab_size,
                           self.demo_vocab_size,
                           embed_size=args.embedding_size,rnn_size=args.rnn_size,
                           batch_size=args.batch_size) 
        
            
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lrn_rate)#,weight_decay=1e-5
        #self.criterion = model.BCELossWeight(self.device)
        self.criterion = nn.BCELoss(reduction='sum')
        self.kl_loss = nn.KLDivLoss()
        self.net.to(self.device)
    
    def save_output(self,auroc_mat,auprc_mat,auprc_base,obs,k,full):
        if full:
            with open('./data/output/'+str(k)+'/'+'auroc_'+'full', 'wb') as fp:
                   pickle.dump(auroc_mat, fp)
            with open('./data/output/'+str(k)+'/'+'auprc_'+'full', 'wb') as fp:
                   pickle.dump(auprc_mat, fp)
            with open('./data/output/'+str(k)+'/'+'base_'+'full', 'wb') as fp:
                   pickle.dump(auprc_base, fp)
        else:
            with open('./data/output/'+str(k)+'/'+'auroc_'+str(obs), 'wb') as fp:
                   pickle.dump(auroc_mat, fp)
            with open('./data/output/'+str(k)+'/'+'auprc_'+str(obs), 'wb') as fp:
                   pickle.dump(auprc_mat, fp)
            with open('./data/output/'+str(k)+'/'+'base_'+str(obs), 'wb') as fp:
                   pickle.dump(auprc_base, fp)
            
    
    def display_output(self,auroc_mat,auprc_mat,auprc_base,obs,k,micro,full):
        
#         print(auroc_mat)
        
        def plot(mat,base,full,title):
            #label=['3%','5%','10%','25%','50%','75%','85%','90%','95%','97%','>97%']
            #print(mat)
            #print(micro)
            temp_mat=mat[:,1:]
            normed_matrix = normalize(micro, axis=1, norm='l1')
            #print(normed_matrix)
            normed_matrix=np.multiply(temp_mat,normed_matrix)
            #print(normed_matrix)
            print(np.sum(normed_matrix,axis=1))
            
            x = np.arange(8)  # the label locations
            barWidth = 0.2  # the width of the bars

            plt.rcParams["figure.figsize"] = (10,3)
           
        
            r1 = np.arange(len(mat[:,0]))
            n_class_labels=['Not Overweight/Obese','Overweight','Obese']
            plt.bar(r1, mat[:,0], width=barWidth, label=n_class_labels[0])
            
            for n_class in range(1,args.labels):
                r1=[x + barWidth for x in r1]
                plt.bar(r1, mat[:,n_class], width=barWidth, label=n_class_labels[n_class])
            
                
                
                
            
            
            if title=='AUPRC':
                r1 = np.arange(len(base[:,0]))                
                plt.bar(r1, base[:,0], color='black', width=barWidth)
                for n_class in range(1,args.labels):
                    r1=[x + barWidth for x in r1]
                    plt.bar(r1, base[:,n_class], color='black', width=barWidth)
           
            #for x, y in zip(list(np.arange(0,24,0.2)), mat.reshape(-1)):
            #    plt.annotate(str(round(y,2)), (x,y+0.06))
            print(mat)   
            #print(zip(list(range(0,24)), mat.reshape(-1)))
            #plt.plot(np.nanmean(mat,axis=1),marker='o')
            for x, y in zip(list(range(0,8)), np.nanmean(temp_mat,axis=1)):
                plt.annotate(str(round(y,2)), (x,y+0.09))

            # Add some text for labels, title and custom x-axis tick labels, etc.
            plt.ylabel(title, fontsize=12)
            if full:
                plt.title('No obs and pred defined', fontsize=12)
            else:
                plt.title('Observation Window 0 to '+str(obs+2), fontsize=12)
            plt.xticks([r + barWidth for r in range(len(mat[:,0]))], ['3', '4', '5', '6', '7','8','9','10'], fontsize=18)
            plt.legend(loc="center right", bbox_to_anchor=(1.1,0.5), fontsize=8)
            
            if full:
                plt.savefig('./data/output/'+str(k)+'/'+title+'_'+'full'+'.png')
            else:
                plt.savefig('./data/output/'+str(k)+'/'+title+'_'+str(obs)+'.png')
            plt.show()
            
        
        plot(auroc_mat,auprc_base,full,title="AUROC")
        plot(auprc_mat,auprc_base,full,title="AUPRC")
    
    
    
          


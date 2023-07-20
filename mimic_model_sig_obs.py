import os
#import jsondim
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
from sklearn import metrics
import importlib
import numpy as np
from collections import defaultdict
import sys
import parameters
from parameters import *
import argparse
import pandas as pd
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from argparse import ArgumentParser

importlib.reload(parameters)
import parameters
from parameters import *
import math

class BCELossWeight(nn.Module):
    def __init__(self,device):
        super(BCELossWeight, self).__init__()
        
        self.device=device
        
    def forward(self,m_prob,dec_labels_sig):
        #print(dec_labels_sig.shape)
        #print(dec_labels_sig[0:10])
#         pos1=1/math.sqrt(dec_labels_sig[:,0].sum())
#         pos2=1/math.sqrt(dec_labels_sig[:,1].sum())
#         pos3=1/math.sqrt(dec_labels_sig[:,2].sum())
        
        pos1=dec_labels_sig[:,0].sum()
        pos2=dec_labels_sig[:,1].sum()
        pos3=dec_labels_sig[:,2].sum()
        
        #print(pos1,pos2,pos3)
        
#         neg1=1/math.sqrt(dec_labels_sig.shape[0]-(dec_labels_sig[:,0].sum()))
#         neg2=1/math.sqrt(dec_labels_sig.shape[0]-(dec_labels_sig[:,1].sum()))
#         neg3=1/math.sqrt(dec_labels_sig.shape[0]-(dec_labels_sig[:,2].sum()))
        neg1=dec_labels_sig.shape[0]-(dec_labels_sig[:,0].sum())
        neg2=dec_labels_sig.shape[0]-(dec_labels_sig[:,1].sum())
        neg3=dec_labels_sig.shape[0]-(dec_labels_sig[:,2].sum())
        
        
        
#         tot1=pos1+neg1
#         tot2=pos2+neg2
#         tot3=pos3+neg3
        
        pos1=1
        pos2=neg2/pos2
        pos3=neg3/pos3
        
#         neg1=neg1/(tot1)
#         neg2=neg2/(tot2)
#         neg3=neg3/(tot3)
        #print(pos1,pos2,pos3)
    
        element_weight_pos=torch.FloatTensor([1,pos2,pos3]).repeat(dec_labels_sig.shape[0],1)
        element_weight_pos=element_weight_pos.to(self.device)
        element_weight_pos=element_weight_pos*dec_labels_sig
        
        element_weight_neg=torch.FloatTensor([1,1,1]).repeat(dec_labels_sig.shape[0],1)
        element_weight_neg=element_weight_neg.to(self.device)
        element_weight_neg=element_weight_neg*(1-dec_labels_sig)
        
        element_weight=element_weight_pos+element_weight_neg
        
        #print(element_weight_pos[0:10,:])
        #print(element_weight_neg[0:10,:])
#         print(element_weight[0:10])
        
#         bce=nn.BCELoss(reduction='none')
#         loss=bce(m_prob,dec_labels_sig)
#         print(loss)
#         print(torch.sum(loss*element_weight))
        
        bce=nn.BCELoss(weight=element_weight,reduction='sum')
#         print(m_prob)
#         print(dec_labels_sig)
        loss=bce(m_prob,dec_labels_sig)
#         print(loss)
        return loss
        
        


    
class EncDec2(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,demo_vocab_size,embed_size,rnn_size,batch_size):
        super(EncDec2, self).__init__()
        self.embed_size=embed_size
        self.latent_size=args.latent_size
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        self.demo_vocab_size=demo_vocab_size
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):
        
        self.emb_feat=FeatEmbed(self.device,self.feat_vocab_size,self.embed_size,self.batch_size) 
        
        self.emb_age=AgeEmbed(self.device,self.age_vocab_size,self.embed_size,self.batch_size) 
        
        self.emb_demo=DemoEmbed(self.device,self.demo_vocab_size,self.embed_size,self.batch_size) 
        
#         self.disc=Discriminator(self.device)
        self.enc=Encoder2(self.device,self.feat_vocab_size,self.age_vocab_size,self.embed_size,self.rnn_size,self.batch_size)
        self.dec=Decoder2(self.device,self.feat_vocab_size,self.age_vocab_size,self.embed_size,self.rnn_size,self.emb_feat,self.batch_size)
        

        
        
    def forward(self,visualize_embed,find_contri,enc_feat,enc_len,enc_age,enc_demo,dec_feat,labels,mask):   
        if visualize_embed:
            features=torch.tensor(np.arange(0,350))
            features=features.type(torch.LongTensor)
            features=features.to(self.device)
            emb=self.emb_feat(features)
            return emb
        
        contri=torch.cat((enc_feat.unsqueeze(2),enc_age.unsqueeze(2)),2)
        #print("contri",contri.shape)
        #print(contri[0])
#         print("Enc featEmbed",contri.shape)
#         print("Enc featEmbed",contri[0])
        enc_feat=self.emb_feat(enc_feat)
        enc_age=self.emb_age(enc_age)
        enc_demo=self.emb_demo(enc_demo)
#         print("Enc featEmbed",enc_feat.shape)
#         print("Enc ageEmbed",enc_age.shape)
   
        code_output,code_h_n,code_c_n=self.enc(enc_feat,enc_len,enc_age)
        #print("code_output",code_output.shape)
        #print("code_output_n",code_output_n.shape)
        #print("========================")
        
        #===========DECODER======================
#         print(dec_feat.shape)
        idx=torch.count_nonzero(dec_feat,dim=2)
        idx=idx.unsqueeze(2)
        idx=idx.repeat(1,1,self.embed_size)
#         print(idx.shape)
        dec_feat=self.emb_feat(dec_feat)
#         print(dec_feat.shape)
        dec_feat=torch.sum(dec_feat, 2)
#         print(dec_feat.shape)
#         dec_feat=torch.div(dec_feat,idx)
#         print(dec_feat.shape)
        
#         print("Dec featEmbed",dec_feat.shape)
        dec_labels=self.emb_feat(labels)
#         print("Dec LabelEmbed",dec_labels.shape)
        
        if find_contri: 
            dec_output,dec_prob,disc_input,kl_input,all_contri=self.dec(find_contri,contri,dec_feat,dec_labels,code_output,code_h_n,code_c_n,enc_demo,mask,labels)
        else:
            dec_output,dec_prob,disc_input,kl_input=self.dec(find_contri,contri,dec_feat,dec_labels,code_output,code_h_n,code_c_n,enc_demo,mask,labels)
         
            
        kl_input=torch.tensor(kl_input)
#         print(len(disc_input))
#         print(disc_input[0:5])
#         print("------------------")
#         print(disc_input[195:202])
        disc_input=torch.stack(disc_input)
#         print(disc_input.shape)
#         print(disc_input[0:5])
#         print(disc_input[198:205])
        disc_input=torch.reshape(disc_input,(args.time,-1,disc_input.shape[1]))
#         print(disc_input.shape)
#         print(disc_input[:,0,:])
        disc_input=disc_input.permute(1,0,2)
#         self.disc(disc_input,mask,labels)
    
#         print(disc_input.shape)
#         print(disc_input[0,:,:])
#         print(disc_input[0:10])
#         print(disc_input[195:205])
        kl_input=torch.reshape(kl_input,(args.time,-1))
#         print(disc_input)
        kl_input=kl_input.permute(1,0)
#         print(disc_input.shape)
#         print(disc_input[0])
#         print(disc_input[1])
        #print(len(dec_prob))
        #print(dec_prob)
        #dec_prob=torch.stack(dec_prob)
        #print(dec_prob)
        
        #print("===================================================")
        
#         print(dec_prob[0].shape)
        #dec_output=torch.tensor(dec_output)
#         print("dec_output",dec_output.shape)
#         dec_output=dec_output.permute(1,0)
        #print("dec_output",dec_output)
#         print("dec_output",dec_output[0])
        #dec_prob=torch.tensor(dec_prob)
#         print("dec_prob",dec_prob.shape)
#         print(dec_prob[:,0,:])

#         dec_prob=dec_prob.permute(1,0,2)
#         print("dec_prob",dec_prob.shape)
#         print(dec_prob[0])

#         print("dec_output",dec_output.shape)
        
        if find_contri: 
            return dec_output,dec_prob,kl_input,all_contri
        else:
            return dec_output,dec_prob,kl_input,disc_input
    
    
class Encoder2(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,batch_size):
        super(Encoder2, self).__init__()
        self.embed_size=embed_size
#         self.latent_size=args.latent_size
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):

        
        self.rnn=nn.LSTM(input_size=self.embed_size*2,hidden_size=self.rnn_size,num_layers = args.rnnLayers,batch_first=True)
        #self.code_max = nn.AdaptiveMaxPool1d(1, True)
        self.drop=nn.Dropout(p=0.2)
 
        
    def forward(self,featEmbed,lengths,ageEmbed):   

        out1=torch.cat((featEmbed,ageEmbed),2)
#         print("out",out1.shape)
        
        out1=out1.type(torch.FloatTensor)
        out1=out1.to(self.device)
        #print(out1)
        #out1=self.drop(out1)
        h_0, c_0 = self.init_hidden(featEmbed.shape[0])
        h_0, c_0 = h_0.to(self.device), c_0.to(self.device)
        
        lengths=lengths.type(torch.LongTensor)
        #lengths=lengths.to(self.device)
        #print("lengths",lengths)
        
        code_pack = torch.nn.utils.rnn.pack_padded_sequence(out1, lengths, batch_first=True,enforce_sorted=False)
        
        #Run through LSTM
        code_output, (code_h_n, code_c_n)=self.rnn(code_pack, (h_0, c_0))
        code_h_n=code_h_n[-1,:,:].squeeze()
        code_c_n=code_c_n[-1,:,:].squeeze()
        #unpack sequence
        code_output,_  = torch.nn.utils.rnn.pad_packed_sequence(code_output, batch_first=True)
#         print("code_output",code_output.shape)
#         print("code_h_n",code_h_n.shape)
#         print("code_c_n",code_h_n.shape)
       
        
        
        
        
        
        return code_output,code_h_n,code_c_n

    
    def init_hidden(self,batch_size):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(args.rnnLayers,batch_size, self.rnn_size)
        c=torch.zeros(args.rnnLayers,batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    
    
class SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, device,vocab_size,embed_size,batch_size):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            max_norm=1,
        )
        self.linear = nn.Linear(
            in_features=embed_size,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x
        
        
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, train, transform = None):
        
        enc1=pd.read_csv('./data/3/enc_train.csv',header=0)
        enc_len1=pd.read_csv('./data/3/lengths_train.csv',header=0)
        demo1=pd.read_csv('./data/3/demo_train.csv',header=0)

        enc2=pd.read_csv('./data/3/enc_test.csv',header=0)
        enc_len2=pd.read_csv('./data/3/lengths_test.csv',header=0)
        demo2=pd.read_csv('./data/3/demo_test.csv',header=0)
        
        enc=pd.concat([enc1,enc2],axis=0)
        enc_len=pd.concat([enc_len1,enc_len2],axis=0)
        demo=pd.concat([demo1,demo2],axis=0)
        
#         print(enc1.shape)
#         print(enc.shape)
        ids=enc['person_id'].unique()
        demo=demo.groupby('person_id').last().reset_index()
        #print(demo.shape)
        #print(enc_len.shape)
    
        
        enc_feat=enc['feat_dict'].values
        enc_eth=demo['Eth_dict'].values
        enc_race=demo['Race_dict'].values
        enc_sex=demo['Sex_dict'].values
        enc_len=enc_len['person_id'].values
        
        enc_age=enc['age_dict'].values
        
        dic={350:0,351:1,352:2,353:3,354:4,355:5,356:6,357:7,358:8}
        enc_eth=torch.tensor([dic[x.item()] for x in enc_eth])
        enc_race=torch.tensor([dic[x.item()] for x in enc_race])
        enc_sex=torch.tensor([dic[x.item()] for x in enc_sex])
        
        #Reshape to 3-D
        #print(enc_feat.shape)
        enc_feat=torch.tensor(enc_feat)
        enc_feat=torch.reshape(enc_feat,(len(ids),-1))
        enc_feat=enc_feat.type(torch.LongTensor)
        
        enc_len=torch.tensor(enc_len)
        #enc_len=torch.reshape(enc_len,(len(ids),-1))
        enc_len=enc_len.type(torch.LongTensor)
        
        enc_age=torch.tensor(enc_age)
        enc_age=torch.reshape(enc_age,(len(ids),-1))
        enc_age=enc_age.type(torch.LongTensor)
        
        enc_eth=torch.tensor(enc_eth)
        enc_eth=enc_eth.unsqueeze(1)
        enc_race=torch.tensor(enc_race)
        enc_race=enc_race.unsqueeze(1)
        enc_sex=torch.tensor(enc_sex)
        enc_sex=enc_sex.unsqueeze(1)
        #print(enc_eth.shape)
        #print(enc_sex)
        enc_demo=torch.cat((enc_eth,enc_race),1)
        enc_demo=torch.cat((enc_demo,enc_sex),1)
        enc_demo=enc_demo.type(torch.LongTensor)
        
        self.df=enc_feat

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        
        return self.df[index]
    
    
class Decoder2(nn.Module):
    def __init__(self,device,feat_vocab_size,age_vocab_size,embed_size,rnn_size,emb_feat,batch_size):
        super(Decoder2, self).__init__()
        self.embed_size=embed_size
        
        self.rnn_size=rnn_size
        self.feat_vocab_size=feat_vocab_size
 
        self.age_vocab_size=age_vocab_size
        
        
        self.padding_idx = 0
        self.device=device
        self.emb_feat=emb_feat
        self.build()
        
    def build(self):
        
        #self.fc_pool=nn.Linear(self.rnn_size, args.labels, False)
        #self.rnn_cell = nn.GRUCell(self.embed_size*2, self.rnn_size)
        self.linears = nn.ModuleList([nn.Linear((2*self.rnn_size)+5*int(self.embed_size/2)+4*int(self.embed_size), 2*(args.latent_size)) for i in range(args.time)])
        #+1*int(self.embed_size)
        self.linearsMed = nn.ModuleList([nn.Linear(2*(args.latent_size), args.latent_size) for i in range(args.time)])
        #self.regression = nn.Linear((2*self.rnn_size)+12, args.latent_size)
        self.linearsLast = nn.ModuleList([nn.Linear(args.latent_size, 1) for i in range(args.time)])
        
        self.leaky = nn.ModuleList([nn.LeakyReLU(0.1) for i in range(args.time)])
        
        self.drop=nn.Dropout(p=0.2)
        
        #self.regressionLast = nn.Linear(args.latent_size, 1)
        self.attn=LSTMAttention(self.rnn_size)
        #self.sig=nn.Sigmoid()
        #self.leaky = nn.LeakyReLU(0.1)
        
    def forward(self,find_contri,contri,featEmbed,labelsEmbed,encoder_outputs,h_n,c_n,enc_demo,mask,labels):   

        dec_output=[]
        kl_input=[]
        disc_input=[]
        dec_prob=[]
        for t in range(args.time):
                #print("===============",t,"======================")
                #predict bmi
                
#                 print("h_n",h_n.shape)
                #print("encoder_outputs",encoder_outputs.shape)
                #encoder_outputs = torch.cat((encoder_outputs, enc_demo), dim =1)
                #print("encoder_outputs",encoder_outputs.shape)
                #encoder_outputs=self.drop(encoder_outputs)
                a = self.attn(encoder_outputs)
#                 print("a",a.shape)
                if (find_contri) and (t==2):
                    all_contri=self.attention(contri,a)
                a = a.unsqueeze(1)
#                 print("a",a.shape)
#                 print("attn",a[0,0,0:5])
#                 print("encoder_outputs",encoder_outputs[0,0:5,:])

        #         print("fc_w",fc_w.shape)
                weighted = torch.bmm(a, encoder_outputs)
#                 print("weighted",weighted[0,0,:])
#                 print("weighted",weighted.shape)
                weighted = weighted.permute(1, 0, 2)
                weighted = weighted.squeeze()
                #print("weighted",weighted.shape)

                #print("h_n",h_n.shape)
                #print("dmeo",enc_demo.shape)
                reg_input = torch.cat((h_n, weighted), dim =1)
#                 print(reg_input.shape)
                reg_input = torch.cat((reg_input, enc_demo), dim =1)
#                 print(reg_input.shape)
#                 print(featEmbed[:,0,:].shape)
                reg_input = torch.cat((reg_input, featEmbed[:,0,:]), dim =1)
                reg_input = torch.cat((reg_input, featEmbed[:,1,:]), dim =1)
                reg_input = torch.cat((reg_input, featEmbed[:,2,:]), dim =1)
                reg_input = torch.cat((reg_input, featEmbed[:,3,:]), dim =1)
#                 print(reg_input.shape)
                bmi_h = self.linears[t](reg_input)
                bmi_h = self.leaky[t](bmi_h)
                bmi_h = self.linearsMed[t](bmi_h)
                bmi_h = self.linearsLast[t](bmi_h)
                bmi_prob=torch.sigmoid(bmi_h)
                bmi_prob_non=(1-bmi_prob[:,0]).unsqueeze(1)
                bmi_prob=torch.cat((bmi_prob_non,bmi_prob),axis=1)
#                 print("bmi_prob",bmi_prob.shape)
#                 bmi_prob=torch.max(bmi_label,dim=1).values
#                 print(bmi_prob[0:10])
#                 print(bmi_prob_non[0:10])
                bmi_label=torch.argmax(bmi_prob,dim=1)
                dec_output.append(bmi_label)
                #print("bmi_label",bmi_label.shape)

                d = {0:0,1:1}
                bmi_label_dict=torch.tensor([d[x.item()] for x in bmi_label])

          
                kl_input.extend(bmi_label_dict)
                disc_input.extend(bmi_h)
                dec_prob.append(bmi_prob)

        if find_contri:
            return dec_output,dec_prob,disc_input,kl_input,all_contri
        else:
            return dec_output,dec_prob,disc_input,kl_input
    
    def attention(self,contri,attn):
        #pool = pool.data.cpu().numpy()
        #print("attn in dec",attn.shape)
        #print("contri in dec",contri.shape)
        all_contri=[]
        contri=contri[:attn.shape[0],:attn.shape[1],:]
        attn=attn.unsqueeze(2)
        attn=attn.to('cpu')
        #print("contri in dec",contri.shape)
        #print(contri[0])
        #print("attn in dec",attn.shape)
        #print(attn[0])
        contri=contri.type(torch.FloatTensor)
        contri=torch.cat((contri,attn),2)
        contri=contri.type(torch.FloatTensor)
        #print("contri in dec",contri.shape)
#         all_contri.append(contri)
        return contri




class LSTMAttention(nn.Module):
    def __init__(self, rnn_size):
        super().__init__()
        
        self.attn = nn.Linear((rnn_size) , int(rnn_size/2))
        #self.attn = nn.Linear((enc_hid_dim * 2), dec_hid_dim)
        self.v = nn.Linear(int(rnn_size/2), 1, bias = False)
        
    def forward(self, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        #print("=====================inside attention======================")
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        
        
        energy = torch.tanh(self.attn(encoder_outputs)) 
        #energy = torch.tanh(self.attn(encoder_outputs)) 
        #print("energy",energy.shape)
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        #print("attention",attention.shape)
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)

    


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p: torch.tensor, q: torch.tensor):
        #p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        print(p)
        print(q)
        m = (0.5 * (p + q))
        print(m)
        return 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))
    
    
class FeatEmbed(nn.Module):
    def __init__(self,device,feat_vocab_size,embed_size,batch_size):
        super(FeatEmbed, self).__init__()
        self.embed_size=embed_size

        self.feat_vocab_size=feat_vocab_size

        
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):
        
        self.emb_feat=nn.Embedding(self.feat_vocab_size,self.embed_size,self.padding_idx) 

    def forward(self,feat):  
        feat=feat.type(torch.LongTensor)
        feat=feat.to(self.device)
        #print(self.emb_feat(torch.LongTensor([0,167])))
        featEmbed=self.emb_feat(feat)
        #print("enc feat",featEmbed[0])
        featEmbed=featEmbed.type(torch.FloatTensor)
        featEmbed=featEmbed.to(self.device)
            
        

        return featEmbed   

class AgeEmbed(nn.Module):
    def __init__(self,device,age_vocab_size,embed_size,batch_size):
        super(AgeEmbed, self).__init__()
        self.embed_size=embed_size

        self.age_vocab_size=age_vocab_size

        
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):
        
        self.emb_age=nn.Embedding(self.age_vocab_size,self.embed_size,self.padding_idx) 

    def forward(self,age):  
        
        age=age.type(torch.LongTensor)
        age=age.to(self.device)
        ageEmbed=self.emb_age(age)

        ageEmbed=ageEmbed.type(torch.FloatTensor)
        ageEmbed=ageEmbed.to(self.device)


        return ageEmbed  
        
class DemoEmbed(nn.Module):
    def __init__(self,device,demo_vocab_size,embed_size,batch_size):
        super(DemoEmbed, self).__init__()
        self.embed_size=embed_size

        self.demo_vocab_size=demo_vocab_size

        
        self.padding_idx = 0
        self.device=device
        self.build()
        
    def build(self):
        
        self.emb_demo=nn.Embedding(self.demo_vocab_size,self.embed_size,self.padding_idx) 
        self.fc=nn.Linear(self.embed_size, int(self.embed_size/2))
        #self.fc2=nn.Linear(int(self.embed_size/2), int(self.embed_size/2))

    def forward(self,demo):  
        #print(demo.shape)
        
        demo=demo.type(torch.LongTensor)
        demo=demo.to(self.device)
        demoEmbed=self.emb_demo(demo)
        demoEmbed=self.fc(demoEmbed)
#         print(demoEmbed.shape)
#         print(demoEmbed[0])
        demoEmbed=torch.reshape(demoEmbed,(demoEmbed.shape[0],-1))
#         print(demoEmbed.shape)
#         print(demoEmbed[0])
        demoEmbed=demoEmbed.type(torch.FloatTensor)
        demoEmbed=demoEmbed.to(self.device)
        
        #print(demoEmbed.shape)
        #demoEmbed=self.fc(demoEmbed)
        #demoEmbed=self.fc2(demoEmbed)
        #print(demoEmbed.shape)
        return demoEmbed  

    

            




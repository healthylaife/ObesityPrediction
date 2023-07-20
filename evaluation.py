import os
#import jsondim
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
from sklearn import metrics
from sklearn.calibration import calibration_curve
import pickle
import dcurves
import importlib
from matplotlib import pyplot
import import_ipynb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import callibrate_output
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

importlib.reload(callibrate_output)
import callibrate_output

if not os.path.exists("./data/output"):
    os.makedirs("./data/output")
    
class Loss(nn.Module):
    def __init__(self,device,curve,acc,ppv,sensi,tnr,npv,auroc,aurocPlot,auprc,auprcPlot,callb,callbPlot):
        super(Loss, self).__init__()
        
        self.device=device
        self.curve=curve
        self.acc=acc
        self.ppv=ppv
        self.sensi=sensi
        self.tnr=tnr
        self.npv=npv
        self.auroc=auroc
        self.aurocPlot=aurocPlot
        self.auprc=auprc
        self.auprcPlot=auprcPlot
        self.callb=callb
        self.callbPlot=callbPlot

    def forward(self, prob, labels,logits,ob):
        classify_loss='NA' 
        auc,apr='NA'
        base='NA'
        accur='NA'
        prec='NA'
        recall='NA'
        spec='NA'
        npv_val='NA'
        ECE='NA'
        MCE='NA'
        
        #print(prob)
        #print(labels)
#         if len(prob)==0:
#             return 0.00,0.00,0.00
        pos_ind = labels >= 0.5
        neg_ind = labels < 0.5
        pos_label = labels[pos_ind]
        neg_label = labels[neg_ind]
        pos_prob = prob[pos_ind]
        neg_prob = prob[neg_ind]
        pos_loss, neg_loss = 0, 0

        
        #################           AUROC            #######################
        
        #labels = labels.data.cpu().numpy()
        #prob = prob.data.cpu().numpy()
        #print(labels)
        #print(prob)
        if(self.auroc):
            fpr, tpr, threshholds = metrics.roc_curve(labels, prob)
            auc = metrics.auc(fpr, tpr)
        if(self.aurocPlot):
            self.auroc_plot(labels, prob,ob)
        
        #################           AUPRC            #######################
        if(self.auprc):
            base = ((labels==1).sum())/labels.shape[0]

            precision, recall, thresholds = metrics.precision_recall_curve(labels, prob)
            apr = metrics.auc(recall, precision)
            print("apr",apr)
        if(self.auprcPlot):
            self.auprc_plot(labels, prob,ob)
        
        # stati number
        prob1 = prob >= 0.5
        #print(prob)
        
        pos_l = (labels==1).sum()
        neg_l = (labels==0).sum()
        pos_p = (prob1 + labels == 2).sum()#how many positives are predicted positive#####TP
        neg_p = (prob1 + labels == 0).sum()#True negatives
        prob2 = prob < 0.5
        fn    = (prob2 + labels==2).sum()
        fp    = (prob2 + labels==0).sum()
        #print(classify_loss, pos_p, pos_l, neg_p, neg_l)
        
        #################           DECISION CURVE            #######################
        if(self.curve):
            self.decision_curve(prob,labels,ob)
            
        
        #################           Accuracy            #######################
        if(self.acc):
            accur=metrics.accuracy_score(labels,prob>=0.5)
        
        #################           Precision/PPV  (TP/(TP+FP))         #######################
        if(self.ppv):
            prec=metrics.precision_score(labels,prob>=0.5)
        
        #################           Recall/TPR/Sensitivity(TP/(TP+FN))          #######################
        if(self.sensi):
            recall=pos_p/(pos_p+fn)
        #################           Specificity/TNR  (TN/(TN+FP))         #######################
        if(self.tnr):
            spec=neg_p/(neg_p+fp)
        
        #################           NPV  (TN/(TN+FN))         #######################
        if(self.npv):
            npv_val=neg_p/(neg_p+fn)
        #################           Callibration         #######################
        if(self.callb):
            if(self.callbPlot):
                ECE, MCE = self.calb_metrics(prob,labels,logits,ob,True)
            else:
                ECE, MCE = self.calb_metrics(prob,labels,logits,False)
        
        #################           Fairness         #######################
        
        
#         print("BCE Loss: {:.2f}".format(classify_loss))
#         print("AU-ROC: {:.2f}".format(auc))
#         print("AU-PRC: {:.2f}".format(apr))
#         print("AU-PRC Baaseline: {:.2f}".format(base))
#         print("Accuracy: {:.2f}".format(accur))
#         print("Precision: {:.2f}".format(prec))
#         print("Recall: {:.2f}".format(recall))
#         print("Specificity: {:.2f}".format(spec))
#         print("NPV: {:.2f}".format(npv_val))
#         print("ECE: {:.2f}".format(ECE))
#         print("MCE: {:.2f}".format(MCE))
        
        #return [classify_loss, auc,apr,base,accur,prec,recall,spec,npv_val,ECE,MCE]
        return 1,2,3
        return round(auc,2),round(apr,2),round(base,2)
    
    def decision_curve(self,prob,labels,ob):
#         path='data/baseline/'+str(2)+'/'+str(5)
#         df_binary_baseline=pd.read_csv('./'+path+'/'+'label_list.csv',header=0)
# #         print(df_binary)
#         binary_output_df_baseline = dcurves.dca(
#                 data = df_binary_baseline,
#                 outcome = 'label',
#                 predictors = ['Baseline'],
#                 thresh_vals = [0.1, 0.8, 0.1]
#         )
#         binary_output_df_baseline=binary_output_df_baseline[binary_output_df_baseline['predictor']=='Baseline']
#         print(binary_output_df_baseline)
    
        d = {'Model': prob, 'label': labels}
        df_binary=pd.DataFrame(data=d)
        print(df_binary.shape)
#         print(df_binary.head())
#         df_binary = dcurves.load_test_data.load_binary_df()
#         print(df_binary.head())
        binary_output_df = dcurves.dca(
                data = df_binary,
                outcome = 'label',
                predictors = ['Model'],
                thresh_vals = [0.1, 0.8, 0.1]
        )
#         binary_output_df=pd.concat([binary_output_df,binary_output_df_baseline],axis=0)
        print("decision curve")    
        print(binary_output_df)
#         curve=dcurves.plot_net_benefit_graphs(binary_output_df, y_limits=[-0.05, 0.2], color_names=['orange', 'blue', 'green','red'])
#         binary_output_df = dcurves.dca(
#                 data = df_binary,
#                 outcome = 'label',
#                 predictors = ['ob'],
#                 thresh_vals = [0.1, 1.0, 0.1]
#         )
#         curve=dcurves.plot_net_benefit_graphs(binary_output_df, y_limits=[-0.05, 0.2], color_names=['green', 'blue', 'red','brown'])
#         plt.show()
 
        
        
        
        
      
            
    def auroc_plot(self,label, pred,ob):
        print("auroc")
#         path='data/baseline/'+str(2)+'/'+str(5)
#         with open('./'+path+'/'+'label_list', 'rb') as fp:
#                    base_label=pickle.load(fp)
#         with open('./'+path+'/'+'prob_list', 'rb') as fp:
#                    base_pred=pickle.load(fp)
        print(label.shape)       
        pos_l = (ob==1).sum()
        neg_l = (ob==0).sum()

        tp = (ob + label == 2).sum()#how many positives are predicted positive#####TP
        tn = (ob + label == 0).sum()#True negatives
        prob2 = ob < 0.5
        fn    = (prob2 + label==2).sum()
        fp    = (prob2 + label==0).sum()
        
        plt.figure(figsize=(8,6))
        plt.plot([0, 1], [0, 1],'r--')

        
        fpr, tpr, thresh = metrics.roc_curve(label, pred)
#         base_fpr, base_tpr, base_thresh = metrics.roc_curve(base_label, base_pred)
#         print(metrics.roc_auc_score(base_label, base_pred))
        auc = metrics.roc_auc_score(label, pred)
        #auc=0.81
        #print(list(thresh))
        d={'fpr':fpr,'tpr':tpr,'thresh':thresh}
        points=pd.DataFrame(d)
        points=points.sort_values(by=['thresh'])
        

        
        print(auc)
        print(points[(points['tpr']>0.79) & (points['tpr']<0.80)]['fpr'].max())
        print(points[(points['tpr']>0.84) & (points['tpr']<0.85)]['fpr'].max())
        print(points[(points['fpr']>0.19) & (points['fpr']<0.20)]['tpr'].max())
        print(points[(points['fpr']>0.14) & (points['fpr']<0.15)]['tpr'].max())
        
        print(points[(points['tpr']>0.89) & (points['tpr']<0.90)]['fpr'].max())
        print(points[(points['tpr']>0.94) & (points['tpr']<0.95)]['fpr'].max())
        print(points[(points['fpr']>0.09) & (points['fpr']<0.10)]['tpr'].max())
        print(points[(points['fpr']>0.04) & (points['fpr']<0.05)]['tpr'].max())
#         plt.plot(fpr, tpr,label='{:.3f}'.format(auc))
#         plt.plot(base_fpr, base_tpr,label='Baseline')
        
        #plt.plot(fp/(fp+tn), tp/(tp+fn),marker='^', color='red',label='Baseline')
        
#         for pt in range(1,10):
#             tp=0.1*pt
#             points=points[(points['thresh']>=tp)]
#             plt.plot(points.iloc[0,0], points.iloc[0,1],marker='o', color='blue')
            #plt.annotate(str(int(pt*10))+'%', (points.iloc[0,0], points.iloc[0,1]-0.04),fontsize=10)
            
#         plt.ylabel("True Positive Rate",fontsize=15)
#         plt.xlabel("False Positive Rate",fontsize=15)
#         plt.xticks(fontsize=15)
#         plt.yticks(fontsize=15)
#         #plt.legend()
#         plt.title("AUC-ROC")
        
#         #plt.savefig('./data/output/'+"auroc_plot.png")
#         plt.show()
    
    def auprc_plot(self,label, pred,ob):
        path='data/baseline/'+str(2)+'/'+str(5)
        with open('./'+path+'/'+'label_list', 'rb') as fp:
                   base_label=pickle.load(fp)
        with open('./'+path+'/'+'prob_list', 'rb') as fp:
                   base_pred=pickle.load(fp)
        
        pos_l = (ob==1).sum()
        neg_l = (ob==0).sum()
        tp = (ob + label == 2).sum()#how many positives are predicted positive#####TP
        tn = (ob + label == 0).sum()#True negatives
        prob2 = ob < 0.5
        fn    = (prob2 + label==2).sum()
        fp    = (prob2 + label==0).sum()
        
        base=(tp+fn)/(tp+fp+fn+tn)
        
        plt.figure(figsize=(8,6))
        plt.axhline(y=base,color='red',linestyle='--')
        
        
        
        precision, recall, thresholds = metrics.precision_recall_curve(label, pred)
        base_precision, base_recall, base_thresholds = metrics.precision_recall_curve(base_label, base_pred)
        apr = metrics.auc(recall, precision)
        precision=precision[0:len(thresholds)]
        recall=recall[0:len(thresholds)]
        print(len(precision))
        print(len(recall))
        print(len(thresholds))
        d={'fpr':recall,'tpr':precision,'thresh':thresholds}
        points=pd.DataFrame(d)
        points=points.sort_values(by=['thresh'])
        plt.plot(recall, precision,label='{:.3f}'.format(apr))
        plt.plot(base_recall, base_precision,label='Baseline')
        #plt.plot(tp/(tp+fn), tp/(tp+fp),marker='^', color='red',label='Baseline')
        
#         for pt in range(1,5):
#             tp=0.1*pt
#             points=points[(points['thresh']>=tp)]
#             #if points.sha
#             plt.plot(points.iloc[0,0], points.iloc[0,1],marker='o', color='blue')
#             #plt.annotate(str(int(pt*10))+'%', (points.iloc[0,0], points.iloc[0,1]-0.04),fontsize=10)
            
        plt.ylabel("Precision",fontsize=15)
        plt.xlabel("Recall",fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        #plt.legend()
        plt.title("AUC-PRC")
        
        #plt.savefig('./data/output/'+"auroc_plot.png")
        plt.show()
        
    def calb_curve_old(self,bins,bin_accs,ECE, MCE):
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()

        # x/y limits
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)

        # x/y labels
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        # Create grid
        ax.set_axisbelow(True) 
        ax.grid(color='gray', linestyle='dashed')

        # Error bars
        plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

        # Draw bars and identity line
        plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
        plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

        # Equally spaced axes
        plt.gca().set_aspect('equal', adjustable='box')

        # ECE and MCE legend
        ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
        MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
        plt.legend(handles=[ECE_patch, MCE_patch])
        #plt.savefig('./data/output/'+"callibration_plot.png")
        plt.show()
        
        
    def calb_curve(self,labels, prob,labels_cal, preds_cal):   
        
        x, y_plot = calibration_curve(labels, prob, n_bins = 10, normalize = True)
        x_cal, y_plot_cal = calibration_curve(labels_cal, preds_cal, n_bins = 10, normalize = True)
        plt.figure(figsize=[6, 6])
        # Plot a perfectly calibrated curve
        plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
        # Plot model's calibration curve
        plt.plot(y_plot, x, marker = '.', label = 'Before calibration')
        plt.plot(y_plot_cal, x_cal, marker = '.', label = 'calibrated')
        plt.title("Calibration Curves")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.legend(loc="upper left")
        #plt.savefig('./data/output/'+"callibration_plot.png")
        plt.show()
        
    def calb_bins(self,preds,labels):
        # Assign each prediction to a bin
        num_bins = 10
        bins = np.linspace(0.1, 1, num_bins)
        binned = np.digitize(preds, bins)

        # Save the accuracy, confidence and size of each bin
        bin_accs = np.zeros(num_bins)
        bin_confs = np.zeros(num_bins)
        bin_sizes = np.zeros(num_bins)

        for bin in range(num_bins):
            bin_sizes[bin] = len(preds[binned == bin])
            if bin_sizes[bin] > 0:
                bin_accs[bin] = (labels[binned==bin]).sum() / bin_sizes[bin]
                bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

        return bins, binned, bin_accs, bin_confs, bin_sizes


    def calb_metrics(self,preds,labels,logits,ob,curve):
        ECE = 0
        MCE = 0
        bins, _, bin_accs, bin_confs, bin_sizes = self.calb_bins(preds,labels)
        
        for i in range(len(bins)):
            abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
            ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
            MCE = max(MCE, abs_conf_dif)
        if curve:
            #self.calb_curve(bins,bin_accs,ECE, MCE)
            #self.decision_curve(preds,labels,ob)
            #self.calb_curve(labels, preds)
            
            #self.calb_curve(labels, ob)
            test_hids,prob_calb=callibrate_output.callibrate(preds,labels,logits)
            
            #test_hids_ob,prob_calb_ob=callibrate_output.callibrate(ob,labels,ob)
            print("after callibration")
            self.calb_curve(labels[test_hids], prob_calb,labels, preds)
            #self.calb_curve(labels[test_hids], prob_calb_ob)
            
            self.decision_curve(prob_calb,labels[test_hids],ob[test_hids])
        return ECE, MCE



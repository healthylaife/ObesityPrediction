#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import torch
import random
from sklearn import metrics
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./data/output"):
    os.makedirs("./data/output")

def fairness_evaluation(race,eth,gender,ob,payer,prob,labels):
    
    dic={0:350,1:351,2:352,3:353,4:354,5:355,6:356,7:357,8:358}
    #print(eth)
    #print(payer)
#     eth=[dic[x.item()] for x in eth]
#     race=[dic[x.item()] for x in race]
#     gender=[dic[x.item()] for x in gender]
    #print(race)
    with open("./data/5/demoVocab", 'rb') as fp:
            featVocab=pickle.load(fp)
    inv_featVocab = {v: k for k, v in featVocab.items()}
    eth=[inv_featVocab[int(key)] for key in eth]
    race=[inv_featVocab[int(key)] for key in race]
    gender=[inv_featVocab[int(key)] for key in gender]
    payer=[inv_featVocab[int(key)] for key in payer]
    #print(race)
    pred = [1 if p>0.5 else 0 for p in prob]
    d={'race':race,'eth':eth,'gender':gender,'obseity':ob,'payer':payer,'prob':prob,'pred':pred,'labels':labels}
    demo=pd.DataFrame(d)
    print(demo.obseity.unique())
    #output_dict["age_binned"] = output_dict.age.apply(lambda x:"{}-{}".format((x//10)*10,(x//10 + 1)*10))
    sensitive_columns = ["race","eth", "gender",'obseity',"payer"]


    def get_cm_parameters(gt, pred,prob):
        zipped_gt_pred = list(zip(gt,pred))
        n=len(zipped_gt_pred)
        tp = len([pair for pair in zipped_gt_pred if pair == (1,1)])
        tn = len([pair for pair in zipped_gt_pred if pair == (0,0)])
        fp = len([pair for pair in zipped_gt_pred if pair == (0,1)])
        fn = len([pair for pair in zipped_gt_pred if pair == (1,0)])

        try:
            tpr = tp/(tp + fn)
        except ZeroDivisionError:
            tpr = None
        try:
            tnr = tn/(tn + fp)
        except ZeroDivisionError:
            tnr = None
        try:
            fpr = fp/(fp + tn)
        except ZeroDivisionError:
            fpr = None
        try:
            fnr = fn/(fn + tp)
        except ZeroDivisionError:
            fnr = None
        try:
            pr = (tp + fp)/(len(zipped_gt_pred))
        except:
            pr = None
        try:
            nr = (tn + fn)/(len(zipped_gt_pred))
        except:
            nr = None
        try:
            acc = (tp+tn)/(len(zipped_gt_pred))
        except ZeroDivisionError:
            acc = None
        
        try:
            fpr1, tpr1, threshholds = metrics.roc_curve(gt, prob)
            auc = metrics.auc(fpr1, tpr1)
        except ZeroDivisionError:
            auc = None
            
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(gt, prob)
            apr = metrics.auc(recall, precision)
        except ZeroDivisionError:
            apr = None
            
        
        return n,tp, tn, fp, fn, tpr, tnr, fpr, fnr, pr, nr, acc,auc,apr

    report_list = []
    for sens_col in sensitive_columns:
        for group, aggregate in demo.groupby(sens_col):
            tmp_dct = {"sensitive_attribute": sens_col}
            n,tp, tn, fp, fn, tpr, tnr, fpr, fnr, pr, nr, acc,auc,apr = get_cm_parameters(list(aggregate.labels), list(aggregate.pred), list(aggregate.prob))
            tmp_dct.update(dict(
                group=group,n=n,tp=tp, tn=tn, fp=fp, fn=fn, tpr=tpr, tnr=tnr, fpr= fpr, fnr=fnr, pr=pr, nr=nr, accuracy=acc  ,auc=auc,apr=apr  
                )
            )
            report_list.append(tmp_dct)

    report = pd.DataFrame(report_list)
    report=report[report.group!='NI']
    report_groups = {c:i for i,c in enumerate(report.sensitive_attribute.unique())}


    def highlight(s):
        colors = [['background-color: yellow'], ['background-color: green'], ['background-color: red']]
        return colors[report_groups[s.sensitive_attribute]%len(colors)] * len(s)


    try:
        import jinja2
        display(report.style.apply(highlight, axis=1))
    except ImportError:
        display(report)

    #report.to_csv('./data/output/'+outputFile+'.csv',index=False)


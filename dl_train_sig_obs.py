import importlib
import warnings

import numpy as np
import pandas as pd
import torch

import mimic_model_sig_obs as model
import parameters
from parameters import *

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

importlib.reload(model)

importlib.reload(parameters)
from parameters import *


def run(obs):
    test_ids = pd.read_csv('./data/4/tt/test_id.csv', header=0)
    net = torch.load('saved_models/obsNew_4.tar')
    test_hids = list(test_ids['person_id'].unique())

    n_batches = int(len(test_hids) / (args.batch_size))

    for nbatch in range(n_batches):
        print("==================={0:2d} BATCH=====================".format(nbatch))
        test_hids = test_hids[nbatch * args.batch_size:(nbatch + 1) * args.batch_size]

        net.eval()

        enc_feat, enc_len, enc_age, enc_demo, enc_ob = encXY(test_hids)
        dec_feat, dec_labels, mask = decXY(test_hids)

        pred_mask = np.zeros((mask.shape[0], mask.shape[1]))

        if obs > 0:
            pred_mask[:, 0:obs] = mask[:, 0:obs]  # mask right
        pred_mask = torch.tensor(pred_mask)
        pred_mask = pred_mask.type(torch.DoubleTensor)
        # dec_feat

        dec_labels_pred = dec_labels * pred_mask
        pred_mask_feat = pred_mask.unsqueeze(2)
        pred_mask_feat = pred_mask_feat.repeat(1, 1, dec_feat.shape[2])
        pred_mask_feat = pred_mask_feat.type(torch.DoubleTensor)

        dec_feat_pred = dec_feat * pred_mask_feat
        if obs > 0:
            obs_idx = pred_mask[:, obs - 1]  # take last entry before prediction window
            obs_idx = torch.nonzero(obs_idx > 0)
            obs_idx = obs_idx.squeeze()
            dec_feat_pred = dec_feat_pred[obs_idx]
            dec_labels_pred = dec_labels_pred[obs_idx]
            pred_mask = pred_mask[obs_idx]
            mask = mask[obs_idx]
            dec_labels = dec_labels[obs_idx]
            enc_feat, enc_len, enc_age, enc_demo, enc_ob = enc_feat[obs_idx], enc_len[obs_idx], enc_age[
                obs_idx], enc_demo[obs_idx], enc_ob[obs_idx]

        output, prob, disc_input, logits = net(False, False, enc_feat, enc_len, enc_age, enc_demo,
                                               dec_feat_pred, dec_labels_pred, pred_mask)

        mask = np.asarray(mask)

        n_samples = np.zeros((args.time, args.labels))

        obs_mask = np.zeros((mask.shape[0], mask.shape[1]))
        obs_mask[:, obs:] = mask[:, obs:]  # mask left side

        for i in range(0, len(prob) - 1):  # time
            dec_labels_dec = torch.tensor(dec_labels[:, i + obs])
            m = obs_mask[:, i + obs]
            idx = list(np.where(m == 0)[0])
            logits_dec = logits[:, i, :]
            logits_dec = logits_dec.squeeze()
            for l in range(args.labels):  # class
                dec_labels_l = [1 if y == l else 0 for y in dec_labels_dec]

                prob_l = prob[i][:, l].data.cpu().numpy()

                logits_l = logits_dec.data.cpu().numpy()

                prob_l, dec_labels_l, logits_l = np.asarray(prob_l), np.asarray(dec_labels_l), np.asarray(
                    logits_l)
                prob_l = np.delete(prob_l, idx)

                n_samples[i, l] = prob_l.shape[0]


def encXY(ids):
    enc = pd.read_csv('./data/5/tt/enc_test.csv', header=0)
    enc_len = pd.read_csv('./data/5/tt/enc_len_test.csv', header=0)
    demo = pd.read_csv('./data/5/tt/demo_test.csv', header=0)
    enc_ob = pd.read_csv('./data/5/tt/bmi_2_test.csv', header=0)

    enc = enc[enc['person_id'].isin(ids)]
    demo = demo[demo['person_id'].isin(ids)]
    enc_len = enc_len[enc_len['index'].isin(ids)]
    enc_ob = enc_ob[enc_ob['person_id'].isin(ids)]

    enc_ob.loc[enc_ob.value == 'Normal', 'label'] = 0
    enc_ob.loc[enc_ob.value == 'Obesity', 'label'] = 2
    enc_ob.loc[enc_ob.value == 'Overweight', 'label'] = 1

    enc_feat = enc['feat_dict'].values
    enc_eth = demo['Eth_dict'].values
    enc_race = demo['Race_dict'].values
    enc_sex = demo['Sex_dict'].values
    enc_payer = demo['Payer_dict'].values
    enc_coi = demo['COI_dict'].values
    enc_len = enc_len['person_id'].values
    enc_ob = enc_ob['label'].values

    enc_age = enc['age_dict'].values

    # Reshape to 3-D
    enc_feat = torch.tensor(enc_feat)
    enc_feat = torch.reshape(enc_feat, (len(ids), -1))
    enc_feat = enc_feat.type(torch.LongTensor)

    enc_len = torch.tensor(enc_len)
    enc_len = enc_len.type(torch.LongTensor)

    enc_ob = torch.tensor(enc_ob)
    enc_ob = enc_ob.type(torch.LongTensor)

    enc_age = torch.tensor(enc_age)
    enc_age = torch.reshape(enc_age, (len(ids), -1))
    enc_age = enc_age.type(torch.LongTensor)

    enc_eth = torch.tensor(enc_eth)
    enc_eth = enc_eth.unsqueeze(1)
    enc_race = torch.tensor(enc_race)
    enc_race = enc_race.unsqueeze(1)
    enc_sex = torch.tensor(enc_sex)
    enc_sex = enc_sex.unsqueeze(1)
    enc_payer = torch.tensor(enc_payer)
    enc_payer = enc_payer.unsqueeze(1)
    enc_coi = torch.tensor(enc_coi)
    enc_coi = enc_coi.unsqueeze(1)

    enc_demo = torch.cat((enc_eth, enc_race), 1)
    enc_demo = torch.cat((enc_demo, enc_sex), 1)
    enc_demo = torch.cat((enc_demo, enc_payer), 1)
    enc_demo = torch.cat((enc_demo, enc_coi), 1)
    enc_demo = enc_demo.type(torch.LongTensor)

    return enc_feat, enc_len, enc_age, enc_demo, enc_ob


def decXY(ids):
    dec = pd.read_csv('./data/5/tt/dec_test.csv', header=0)
    labels = pd.read_csv('./data/4/tt/labels_test.csv', header=0)
    mask = pd.read_csv('./data/4/tt/mask_test.csv', header=0)

    dec = dec.fillna(0)
    dec = dec.apply(pd.to_numeric)
    del dec['age_dict']
    dec = dec[dec['person_id'].isin(ids)]
    labels = labels[labels['person_id'].isin(ids)]
    mask = mask[mask['person_id'].isin(ids)]

    dec_feat = dec.iloc[:, 2:].values

    dec_labels = labels['value'].values
    mask = mask['value'].values

    # Reshape to 3-D
    dec_feat = torch.tensor(dec_feat)
    dec_feat = torch.reshape(dec_feat, (len(ids), 8, dec_feat.shape[1]))

    dec_labels = torch.tensor(dec_labels)
    dec_labels = torch.reshape(dec_labels, (len(ids), -1))

    mask = torch.tensor(mask)
    mask = torch.reshape(mask, (len(ids), -1))

    return dec_feat, dec_labels, mask


if __name__ == '__main__':
    run(1)

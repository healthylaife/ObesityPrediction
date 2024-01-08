import importlib

import numpy as np
import pandas as pd
import torch

import mimic_model_sig_obs as model
import parameters

importlib.reload(model)

importlib.reload(parameters)


def inference(data, net, obs=1):
    net.eval()

    enc_feat, enc_len, enc_age, enc_demo = encXY(data)
    dec_feat, mask = decXY(data)

    pred_mask = np.zeros((mask.shape[0], mask.shape[1]))

    if obs > 0:
        pred_mask[:, 0:obs] = mask[:, 0:obs]  # mask right
    pred_mask = torch.tensor(pred_mask)
    pred_mask = pred_mask.type(torch.DoubleTensor)
    # dec_feat

    pred_mask_feat = pred_mask.unsqueeze(2)
    pred_mask_feat = pred_mask_feat.repeat(1, 1, dec_feat.shape[2])
    pred_mask_feat = pred_mask_feat.type(torch.DoubleTensor)

    dec_feat_pred = dec_feat * pred_mask_feat
    if obs > 0:
        obs_idx = pred_mask[:, obs - 1]  # take last entry before prediction window
        obs_idx = torch.nonzero(obs_idx > 0)
        obs_idx = obs_idx.squeeze()
        dec_feat_pred = dec_feat_pred[obs_idx]
        enc_feat, enc_len, enc_age, enc_demo = enc_feat[obs_idx], enc_len[obs_idx], enc_age[obs_idx], enc_demo[obs_idx]

    output, prob, disc_input, logits = net(False, False, enc_feat, enc_len, enc_age, enc_demo, dec_feat_pred)
    ########################################################################
    output_prob_list = []
    output_time_list = []
    for i in range(0, len(prob) - 1):  # time
        if output[i].data.cpu().numpy()[0] == 0:  ## 0 is for the first patient
            output_prob_list.append('No')
        else:
            output_prob_list.append('Yes')
        output_time_list.append(i + 1)
    ########################################################################
    # for i in range(0, len(prob) - 1):  # time
    #     print(prob[i][:, 1].data.cpu().numpy())[0]  # prob of class 1 (obesity) ---- [0] is for the first patient
    ########################################################################
    preds = pd.DataFrame({'Age (years)': output_time_list, 'Obesity': output_prob_list}).to_html(index=False)
    print(preds)
    return {'preds': preds}


def encXY(data):
    enc = data['enc']
    enc_len = pd.DataFrame(enc[enc['value'] != '0'].groupby('person_id').size().reset_index(name='counts'))
    demo = data['demo']

    enc_feat = enc['feat_dict'].values
    enc_eth = demo['Eth_dict'].values
    enc_race = demo['Race_dict'].values
    enc_sex = demo['Sex_dict'].values
    enc_payer = demo['Payer_dict'].values
    enc_coi = demo['COI_dict'].values
    enc_len = enc_len['counts'].values

    enc_age = enc['age_dict'].values

    ids_len = len(pd.unique(enc['person_id']))
    # Reshape to 3-D
    enc_feat = torch.tensor(enc_feat)
    enc_feat = torch.reshape(enc_feat, (ids_len, -1))
    enc_feat = enc_feat.type(torch.LongTensor)

    enc_len = torch.tensor(enc_len)
    enc_len = enc_len.type(torch.LongTensor)

    enc_age = torch.tensor(enc_age)
    enc_age = torch.reshape(enc_age, (ids_len, -1))
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

    return enc_feat, enc_len, enc_age, enc_demo


def decXY(data):
    dec = data['dec']

    dec = dec.fillna(0)
    dec = dec.apply(pd.to_numeric)
    del dec['age_dict']

    dec_feat = dec.iloc[:, 2:].values

    ids_len = len(pd.unique(dec['person_id']))

    mask = np.ones((ids_len * 8,))
    # mask = mask['value'].values

    # Reshape to 3-D
    dec_feat = torch.tensor(dec_feat)
    dec_feat = torch.reshape(dec_feat, (ids_len, 8, dec_feat.shape[1]))

    mask = torch.tensor(mask)
    mask = torch.reshape(mask, (ids_len, -1))

    return dec_feat, mask


if __name__ == '__main__':
    data = {}
    person_id = [820427166, 846275781, 848524638]
    enc_df = pd.read_csv('./data/5/tt/enc_test.csv', header=0)
    enc_df = enc_df[enc_df['person_id'].isin(person_id)]
    data['enc'] = enc_df

    demo_df = pd.read_csv('./data/5/tt/demo_test.csv', header=0)
    demo_df = demo_df[demo_df['person_id'].isin(person_id)]
    data['demo'] = demo_df

    dec_df = pd.read_csv('./data/5/tt/dec_test.csv', header=0)
    dec_df = dec_df[dec_df['person_id'].isin(person_id)]
    data['dec'] = dec_df

    net = torch.load('saved_models/obsNew_4.tar')

    inference(data, net, 1)

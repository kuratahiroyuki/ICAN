#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from model_param import model_param_list
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

import warnings
warnings.simplefilter('ignore')
import joblib

from subword_nmt.apply_bpe import BPE
import codecs
import selfies as sf

import os
import time
from Bio import SeqIO

from deep_network import CrossAttention
from measure import measure_evaluation, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC, cofusion_matrix
from prep_dp_dict import df_prep, pickle_load

metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def drug2emb_encoder(x, words2idx_d, max_d, model_params):
    if model_params['encode_method'] == 'smiles' :
        t1 = x
    elif model_params['encode_method'] == 'selfies' :
        t1 = sf.encoder(x).replace("][", "],[").split(',') 
    else : 
        t1 = dbpe.process_line(x).split()  

    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)
    
    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)

def protein2emb_encoder(x, words2idx_p, max_p, model_params):
    if model_params['encode_method'] == 'smiles' :
        t1 = x
    elif model_params['encode_method'] == 'selfies' :
        t1 = x
    else : 
        t1 = pbpe.process_line(x).split()

    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)


class BIN_Data_Encoder(data.Dataset):  
    def __init__(self, list_IDs, labels, df_dti, words2idx_d, max_d, words2idx_p, max_p, model_params):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.words2idx_d = words2idx_d
        self.max_d = max_d
        self.words2idx_p = words2idx_p
        self.max_p = max_p
        self.model_params = model_params
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):

        index = self.list_IDs[index]
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']
        
        d_v, input_mask_d = drug2emb_encoder(d, self.words2idx_d, self.max_d, self.model_params)
        p_v, input_mask_p = protein2emb_encoder(p, self.words2idx_p, self.max_p, self.model_params)       
        y = self.labels[index]

        return d_v, p_v, input_mask_d, input_mask_p, y.astype(np.float32)


def training_testing_process(config, options, model_params):
        print('training_process')

        df_train = options['dataset']['train']
        df_val = options['dataset']['val']
        df_test = options['dataset']['test']
        words2idx_d = options['words2idx_d']
        words2idx_p = options['words2idx_p']
        max_d = options['max_d']
        max_p = options['max_p']
     
        params = {'batch_size': config["training_batch_size"], 'shuffle': True, 'num_workers': 0,  'drop_last': True} 
        training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train, words2idx_d, max_d, words2idx_p, max_p, model_params) 
        training_generator = data.DataLoader(training_set, **params)

        params = {'batch_size': model_params["validation_batch_size"], 'shuffle': True, 'num_workers': 0,  'drop_last': True} 
        validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val, words2idx_d, max_d, words2idx_p, max_p, model_params) 
        validation_generator = data.DataLoader(validation_set, **params)
 
        net = CrossAttention(model_params, feature_size = model_params["feature_size"], n_heads = config["n_heads"], d_dim = config["d_dim"], feature = model_params["feature"], pooling_dropout = model_params["pooling_dropout"], linear_dropout = model_params["linear_dropout"])
        
        device = "cpu"
        if torch.cuda.is_available():
           device = "cuda"
           if torch.cuda.device_count() > 1:
               net = nn.DataParallel(net)
        net.to(device)
        
        opt = optim.Adam(params = net.parameters(), lr = config["lr"])
        criterion = torch.nn.BCELoss()

        if(model_params["stopping_met"] == "loss"):
            max_met = 100 # loss, min
        else:
            max_met = 0 #auc, max

        early_stop_count = 0
        with open(model_params['out_path'] + "/cv_result.txt", 'w') as f:
            print(model_params['out_path'], file = f, flush=True)    
            print("training...", file = f, flush=True)
            print("The number of training data:" + str(df_train.shape[0]), file = f, flush=True)
            print("The number of validation data:" + str(df_val.shape[0]), file = f, flush=True)
                        
            for epoch in range(model_params["max_epoch"]):
                training_losses, validation_losses, train_probs, val_probs, train_labels, val_labels = [], [], [], [], [], []
                
                print("epoch_" + str(epoch+1) + "=====================", file = f, flush=True)           
                net.train()
                for i, (drug_mat, protein_mat, attn_mask_drug, attn_mask_protein, labels) in enumerate(training_generator):
                    opt.zero_grad()
                    probs = net(drug_mat.long().to(device), protein_mat.long().to(device), attn_mask_drug.long().to(device), attn_mask_protein.long().to(device))
                    labels = Variable(torch.from_numpy(np.array(labels)).float()).to(device)
                    labels =labels.unsqueeze(1)            
                    loss1 = criterion(probs, labels)

                    loss1.backward()
                    opt.step()
                    training_losses.append(loss1)
                    train_probs.extend(probs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())
                    train_labels.extend(labels.cpu().clone().detach().squeeze(1).numpy().astype('int32').flatten().tolist())
                    
                loss_epoch = criterion(torch.tensor(train_probs).float(), torch.tensor(train_labels).float())

                print("=============================", file = f, flush = True)
                print("train_loss:: epoch: %d, value: %f,  time: %f" % (epoch+1, loss_epoch, time.time()-start), file = f, flush = True)
                print("train_loss:: epoch: %d, value: %f,  time: %f" % (epoch+1, loss_epoch, time.time()-start)  ) 
                for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](train_labels, train_probs, thresh = model_params["thresh"])
                    else:
                        metrics = metrics_dict[key](train_labels, train_probs)
                    print("train_" + key + ": " + str(metrics), file = f, flush=True) 
                
                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(train_labels, train_probs, thresh = model_params["thresh"])
                print("train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file = f, flush=True)
                print("train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file = f, flush=True)
                print("train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file = f, flush=True)
                print("train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file = f, flush=True)

                print("-----------------------------", file = f, flush = True)

                net.eval()
                val_probs, val_labels = [], []
                for i, (drug_mat, protein_mat, attn_mask_drug, attn_mask_protein, labels) in enumerate(validation_generator):
                    with torch.no_grad():    
                        probs = net(drug_mat.long().to(device), protein_mat.long().to(device), attn_mask_drug.long().to(device), attn_mask_protein.long().to(device))
                        labels = Variable(torch.from_numpy(np.array(labels)).float()).to(device)
                        labels =labels.unsqueeze(1)
                        
                        loss1 = criterion(probs, labels)

                        validation_losses.append(loss1)
                        val_probs.extend(probs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                        val_labels.extend(labels.cpu().detach().squeeze(1).numpy().astype('int32').flatten().tolist())
           
                loss_epoch = criterion(torch.tensor(val_probs).float(), torch.tensor(val_labels).float())
                
                print("validation loss:: "+ str(loss_epoch), file = f, flush = True)
                print("validation_loss:: epoch: %d, value: %f" % (epoch+1, loss_epoch)  ) 
                for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](val_labels, val_probs, thresh = model_params["thresh"])
                    else:
                        metrics = metrics_dict[key](val_labels, val_probs)
                    print("validation_" + key + ": " + str(metrics), file = f, flush=True)

                if(model_params["stopping_met"] == "loss"):
                    epoch_met = loss_epoch  #loss min
                else:
                    #epoch_met = metrics_dict[model_params["stopping_met"]](val_labels, val_probs) #auc, max
                    pass

                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(val_labels, val_probs, thresh = model_params["thresh"])
                print("validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file = f, flush=True)
                print("validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file = f, flush=True)
                print("validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file = f, flush=True)
                print("validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file = f, flush=True)
                
                if epoch_met < max_met:  # loss min
                    early_stop_count = 0
                    max_met = epoch_met

                    #torch.save(net.state_dict(), model_params['out_path'] +"/data_model/deep_model")
                    final_val_probs = val_probs
                    final_val_labels = val_labels
                    final_train_probs = train_probs
                    final_train_labels = train_labels                       
                else:
                    early_stop_count += 1
                    if early_stop_count >= config["early_stopping"]:
                        print('Traning can not improve from epoch {}\tBest {}: {}'.format(epoch + 1 - config["early_stopping"], model_params["stopping_met"], max_met), file = f, flush=True)
                        break

            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    train_metrics = metrics_dict[key](final_train_labels,final_train_probs,thresh = model_params["thresh"])
                    val_metrics = metrics_dict[key](final_val_labels,final_val_probs, thresh = model_params["thresh"])
                else:
                    train_metrics = metrics_dict[key](final_train_labels, final_train_probs)
                    val_metrics = metrics_dict[key](final_val_labels, final_val_probs)
                print("train_" + key + ": " + str(train_metrics), file = f, flush=True)
                print("valid_" + key + ": " + str(val_metrics), file = f, flush=True)

            threshold_1, threshold_2 = cutoff(final_val_labels, final_val_probs)
            #threshold_1 = model_params["thresh"]
            print("Best threshold (AUC) is " + str(threshold_1), file = f, flush=True)
            print("Best threshold (PRC) is " + str(threshold_2), file = f, flush=True)
            torch.save(net.state_dict(), model_params['out_path'] +"/data_model/deep_model")


        #testing_process
        params = {'batch_size':model_params['validation_batch_size'], 'shuffle': True, 'num_workers': 0,  'drop_last': True} 
        test_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test, words2idx_d, max_d, words2idx_p, max_p, model_params) 
        test_generator = data.DataLoader(test_set, **params)

        with open(model_params['out_path'] + "/test_result.txt", 'w') as f:  
            print(model_params['out_path'], file = f, flush=True)
            print("testing...", file = f, flush=True)
            print("The number of testing data:" + str(df_test.shape[0]), file = f, flush=True)
            
            net.eval()
            test_probs, test_labels = [], []

            for i, (drug_mat, protein_mat, attn_mask_drug, attn_mask_protein, labels) in enumerate(test_generator):
                with torch.no_grad():    
                    probs = net(drug_mat.long().to(device), protein_mat.long().to(device), attn_mask_drug.long().to(device), attn_mask_protein.long().to(device))
                    labels = Variable(torch.from_numpy(np.array(labels)).float()).to(device)
                    labels =labels.unsqueeze(1)
                    
                    test_probs.extend(probs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                    test_labels.extend(labels.cpu().detach().squeeze(1).numpy().astype('int32').flatten().tolist())

            print("test_threshold:: value: %f" % (threshold_1), file = f, flush=True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    test_metrics = metrics_dict[key](test_labels, test_probs, thresh = threshold_1)
                else:
                    test_metrics = metrics_dict[key](test_labels, test_probs)
                print("test_" + key + ": " + str(test_metrics), file = f, flush=True)
                
            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(test_labels, test_probs, thresh = threshold_1)
            print("test_true_negative:: value: %f" % (tn_t), file = f, flush=True)
            print("test_false_positive:: value: %f" % (fp_t), file = f, flush=True)
            print("test_false_negative:: value: %f" % (fn_t), file = f, flush=True)
            print("test_true_positive:: value: %f" % (tp_t), file = f, flush=True)

        return final_val_probs, final_val_labels, test_probs, test_labels              

###
if __name__ == '__main__':
    start=time.time()
    print("Setting parameters", flush = True)
    parser = ArgumentParser(description='Test') 
    
    parser.add_argument('--encode_method', help='fcs, smiles, or selfies')
    parser.add_argument('--algorithm', help='')
    parser.add_argument('--DB', choices=['DAVIS', 'BindingDB', 'BIOSNAP/full_data'], default='', type=str, metavar='DB', help='DB name')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run') 
    parser.add_argument('--iter_num', default=1, type=int, metavar='N', help='iteration number')
    args = parser.parse_args()
    model_params = model_param_list()
    model_params['encode_method'] = args.encode_method # fcs, smiles, selfies
    model_params['max_epoch'] = args.epochs
    model_params['out_path'] = './result' + '/result_%s' %args.DB + '/' + args.algorithm  #
    model_params['max_drug_seq'] = 50
    model_params['max_protein_seq'] = 545
    #sequence length

    emb_size=model_params['emb_size'] 
    max_d = model_params['max_drug_seq']
    max_p = model_params['max_protein_seq']
    dropout_rate=model_params['dropout_rate'] 

    config = {
        "training_batch_size": 128,   
        "n_heads" : 4,
        "lr": 0.001,
        "d_dim": 32,
        "early_stopping":20
    }
    iter_num = args.iter_num

    os.makedirs(model_params['out_path'] , exist_ok=True)
    os.makedirs( model_params["out_path"] + "/data_model", exist_ok=True)

    # words2idx_pの作成
    vocab_path = './ESPF/protein_codes_uniprot.txt'
    bpe_codes_protein = codecs.open(vocab_path)
    pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
    sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')
    idx2word_p = sub_csv['index'].values
    words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

    # words2idx_dの作成
    vocab_path = './ESPF/drug_codes_chembl.txt'
    bpe_codes_drug = codecs.open(vocab_path)
    #print('bpe_codes_drug \n{}'.format(bpe_codes_drug))
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    #print('dbpe \n{}'.format(dbpe)) #<subword_nmt.apply_bpe.BPE object at 0x7f4dfd0fa510>
    sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
    #print('sub_csv \n{}'.format(sub_csv))
    idx2word_d = sub_csv['index'].values
    #print('idx2word_d \n{}'.format(idx2word_d))
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
    #print('words2idx_d \n{}'.format(words2idx_d))
    
    #load smiles/selfies to index dictionaries 
    dict_path="./dictionary"
    comb_sel2idx_dict = dict_path + '/comb_sel2idx_dict.pkl'
    comb_sel_aa2idx_dict = dict_path + '/comb_sel_aa2idx_dict.pkl'
    comb_smi2idx_dict = dict_path + '/comb_smi2idx_dict.pkl'
    comb_smi_aa2idx_dict = dict_path + '/comb_smi_aa2idx_dict.pkl'

    comb_sel2idx_dict = pickle_load(comb_sel2idx_dict)
    comb_sel_aa2idx_dict = pickle_load(comb_sel_aa2idx_dict)
    comb_smi2idx_dict = pickle_load(comb_smi2idx_dict)
    comb_smi_aa2idx_dict = pickle_load(comb_smi_aa2idx_dict)

    #dataset input
    infile1 = "./dataset/%s/train.csv" %args.DB
    infile2 = "./dataset/%s/val.csv"  %args.DB
    infile3 = "./dataset/%s/test.csv" %args.DB

    """#The data tha use smiles and selfies is selected   
    df_train, df_val, df_test, num_data = df_prep(infile1, infile2, infile3)
    print(df_test)
    print('num_data {}'.format(num_data))
    """   
     
    df_train = pd.read_csv(infile1, sep=',') 
    df_val = pd.read_csv(infile2, sep=',') 
    df_test = pd.read_csv(infile3, sep=',') 
    
    if model_params['encode_method'] == 'selfies':
        options = {'dataset': {'train': df_train, 'val': df_val, 'test': df_test},'words2idx_d': comb_sel2idx_dict, 'words2idx_p': comb_sel_aa2idx_dict, 'max_d': max_d, 'max_p': max_p    }
        model_params['input_dim_drug'] = len(comb_sel2idx_dict) + len(comb_sel_aa2idx_dict)
        model_params['input_dim_target'] = len(comb_sel2idx_dict) + len(comb_sel_aa2idx_dict)
       
    elif model_params['encode_method'] == 'smiles' :
        options = {'dataset': {'train': df_train, 'val': df_val, 'test': df_test},'words2idx_d': comb_smi2idx_dict, 'words2idx_p': comb_smi_aa2idx_dict, 'max_d': max_d, 'max_p': max_p    }
        model_params['input_dim_drug'] = len(comb_smi2idx_dict) + len(comb_smi_aa2idx_dict)
        model_params['input_dim_target'] = len(comb_smi2idx_dict) + len(comb_smi_aa2idx_dict)

    else:
        options = {'dataset': {'train': df_train, 'val': df_val, 'test': df_test},'words2idx_d': words2idx_d, 'words2idx_p': words2idx_p, 'max_d': max_d, 'max_p': max_p    }

    # score dataframe
    index_value = range(1, iter_num +1)
    columns_measure= ['Sensitivity', 'Specificity', 'Accuracy', 'MCC', 'AUC', 'Precision', 'Recall', 'F1','AUPRC']
    score_val  = pd.DataFrame(data=[], index=index_value, columns=columns_measure)
    score_test = pd.DataFrame(data=[], index=index_value, columns=columns_measure)

    #training and testing
    for iter_cal in range(iter_num):
        val_prob, val_label, test_prob, test_label = training_testing_process(config, options, model_params)

        score_val = measure_evaluation(score_val, val_prob, val_label, iter_cal+1)          
        output = pd.DataFrame([val_prob, val_label], index = ["prob", "label"]).transpose()
        output.to_csv(model_params['out_path'] + '/val_roc_%s.csv' %index_value[iter_cal], header=True, index=True) 
        
        score_test = measure_evaluation(score_test, test_prob, test_label, iter_cal+1)        
        output = pd.DataFrame([test_prob, test_label], index = ["prob", "label"]).transpose()
        output.to_csv(model_params['out_path']  + '/test_roc_%s.csv' %index_value[iter_cal], header=True, index=True)
        
    mean_val=score_val.astype(float).mean(axis='index', numeric_only=True)
    std_val=score_val.astype(float).std(axis='index', numeric_only=True)
    mean_val=pd.DataFrame(np.array(mean_val).reshape(1,-1), index= ['mean'], columns=columns_measure)
    std_val=pd.DataFrame(np.array(std_val).reshape(1,-1), index= ['std'], columns=columns_measure)
    score_val=pd.concat([score_val, mean_val, std_val ])

    mean_test=score_test.astype(float).mean(axis='index', numeric_only=True)
    std_test=score_test.astype(float).std(axis='index', numeric_only=True)
    mean_test=pd.DataFrame(np.array(mean_test).reshape(1,-1), index= ['mean'], columns=columns_measure)
    std_test=pd.DataFrame(np.array(std_test).reshape(1,-1), index= ['std'], columns=columns_measure)
    score_test=pd.concat([score_test, mean_test, std_test ])

    print(score_val)
    print(score_test)  
    score_val.to_csv(model_params['out_path'] + '/val_measures.csv', header=True, index=True)
    score_test.to_csv(model_params['out_path'] + '/test_measures.csv', header=True, index=True)

    

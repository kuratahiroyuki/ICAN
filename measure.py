import numpy as np
import pandas as pd
import sklearn.metrics as metrics



def sensitivity(y_true, y_prob, thresh=0.5):
    #y_true = y_true.cpu().detach().numpy()
    #y_prob = (y_prob.cpu().detach().numpy() + 1 - thresh).astype(np.int16)
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tp / (tp + fn)

def specificity(y_true, y_prob, thresh=0.5):
    #y_true = y_true.cpu().detach().numpy()
    #y_prob = (y_prob.cpu().detach().numpy() + 1 - thresh).astype(np.int16)
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tn / (tn + fp)

def auc(y_true, y_prob):
    #y_true = y_true.cpu().detach().numpy()
    #y_prob = y_prob.cpu().detach().numpy()
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.roc_auc_score(y_true, y_prob)

def mcc(y_true, y_prob, thresh=0.5):
    #y_true = y_true.cpu().detach().numpy()
    #y_prob = (y_prob.cpu().detach().numpy() + 1 - thresh).astype(np.int16)
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.matthews_corrcoef(y_true, y_prob)

def accuracy(y_true, y_prob, thresh=0.5):
    #y_true = y_true.cpu().detach().numpy()
    #y_prob = (y_prob.cpu().detach().numpy() + 1 - thresh).astype(np.int16)
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.accuracy_score(y_true, y_prob)

def cutoff(y_true,y_prob):
    fpr, tpr, thresholds_1 = metrics.roc_curve(y_true, y_prob,drop_intermediate=False)
    precision, recall, thresholds_2 = metrics.precision_recall_curve(y_true, y_prob)
    #return thresholds[np.argmin(np.sqrt(((1-tpr)**2)+(fpr**2)))],fpr, tpr, thresholds
    return thresholds_1[np.argmax(np.array(tpr) - np.array(fpr))], thresholds_2[np.argmax((2 * precision * recall) / (precision + recall))]

def precision(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.precision_score(y_true,y_prob)

def recall(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.recall_score(y_true,y_prob)

def f1(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.f1_score(y_true,y_prob)

def AUPRC(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.average_precision_score(y_true, y_prob)

def cofusion_matrix(y_true,y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    #tn, fp, fn, tp = cm.flatten()

    return tn, fp, fn, tp

metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}
  

def measure_evaluation(score, prob, label, index_val):

    i = index_val - 1
    for key in metrics_dict.keys():
        if(key != "auc" and key != "AUPRC"):
                test_metrics = metrics_dict[key](label, prob, thresh = 0.5)
        else:
                test_metrics = metrics_dict[key](label, prob)
        #print("test_" + key + ": " + str(test_metrics),  flush=True)
        
        if key =='sensitivity'  :
           score.iloc[i,0]= metrics_dict[key](label, prob, thresh = 0.5)
        elif key =='specificity' :
           score.iloc[i,1]= metrics_dict[key](label, prob, thresh = 0.5)
        elif key =='accuracy' :
           score.iloc[i,2]= metrics_dict[key](label, prob, thresh = 0.5)
        elif key =='mcc' :
           score.iloc[i,3]= metrics_dict[key](label, prob, thresh = 0.5)
        elif key =='auc' :
           score.iloc[i,4]= metrics_dict[key](label, prob)   
        elif key =='precision' :                    
           score.iloc[i,5]= metrics_dict[key](label, prob, thresh = 0.5)  
        elif key =='recall' :
           score.iloc[i,6]= metrics_dict[key](label, prob, thresh = 0.5)                            
        elif key =='f1' :
           score.iloc[i,7]= metrics_dict[key](label, prob, thresh = 0.5)      
        elif key =='AUPRC' :
           score.iloc[i,8]= metrics_dict[key](label, prob)                            
        else:
           continue
        
    tn_t, fp_t, fn_t, tp_t = cofusion_matrix(label, prob, thresh = 0.5)
 
    return score


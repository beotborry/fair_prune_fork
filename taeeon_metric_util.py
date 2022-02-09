import numpy as np
import os
from scipy.io import savemat, loadmat
import torch
import sys
sys.path.append('..')
from sklearn.metrics import confusion_matrix


from scipy.special import softmax
from scipy.stats import entropy
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


def compute_ppv(confu_mat):
    fp = confu_mat[0][1]
    tp = confu_mat[1][1]
    ppv = tp / (fp + tp)
    return ppv


# TODO : group consider for binary task
def compute_acc(group_confus, num_groups):

#     total_confu = np.zeros_like(group_confus[str(0.0)])
    total_confu = np.zeros_like(group_confus[str(0)])
    for i in range(num_groups):
#         group_confu = group_confus[str(float(i))]
        group_confu = group_confus[str(int(i))]
        total_confu += group_confu
    
    num_classes = len(total_confu)
    if num_classes==2:
        acc = (total_confu[0,0] + total_confu[1,1]) / np.sum(total_confu)
        acc_per_group = np.zeros(num_groups)
        acc_per_class = np.zeros(num_classes)
        acc_p_cls_p_gro = np.zeros((num_groups,num_classes))
        for i in range(num_classes):
            acc_per_class[i] = total_confu[i,i]/np.sum(total_confu[i,:])
            for group in range(num_groups):
                group_confu = group_confus[str(int(group))]
#                 print('group confu ', group_confu)
                acc_p_cls_p_gro[group, i] += (group_confu[i,i] / np.sum(group_confu[i,:]))
                acc_per_group[group] += ((group_confu[i,i]) / np.sum(group_confu))
    #         acc_diff = np.abs(acc_per_group[0] - acc_per_group[1])
        return acc, acc_per_class, acc_per_group, acc_p_cls_p_gro
    else:
        acc_per_class = np.zeros(num_classes)
        acc_per_group = np.zeros(num_groups)
        acc_p_cls_p_gro = np.zeros((num_groups,num_classes))
        tp=np.zeros(num_classes)
        total_tp = 0
#         tp_0=0
#         tp_1=0
        for i in range(num_classes):
            for group in range(num_groups):
#                 group_confu = group_confus[str(float(group))]
                group_confu = group_confus[str(int(group))]
                tp[group] += group_confu[i,i]
                acc_p_cls_p_gro[group, i] += (group_confu[i,i] / np.sum(group_confu[i,:]))
                acc_per_group[group] += (group_confu[i,i] / np.sum(group_confu))
            
            total_tp += total_confu[i,i]    
            acc_per_class[i] = total_confu[i,i]/np.sum(total_confu[i,:])
        acc = total_tp/np.sum(total_confu)
#         acc_diff = np.abs((tp_0/np.sum(confu_mat0)) - (tp_1/np.sum(confu_mat1)))
#         return acc, acc_per_class, acc_diff, acc_pr_pg
        return acc, acc_per_class, acc_per_group, acc_p_cls_p_gro



def compute_ap(confu_mat, num_classes):
#     total_confu = confu_mat0 + confu_mat1

    total_count = np.sum(confu_mat)
    if num_classes == 2:
        ap = (confu_mat[0,1] + confu_mat[1,0])/total_count
    else:
        ap = np.zeros(num_classes)
        for i in range(num_classes):        
            pred_given_label = np.sum(confu_mat[i,:])
            label_given_pred = np.sum(confu_mat[:,i])
            fp = label_given_pred - confu_mat[i,i]
            fn = pred_given_label - confu_mat[i,i]
            ap[i] = (fp+fn) / total_count

    return ap

def compute_dp(confu_mat, num_classes):
#     print(confu_mat)
    total = np.sum(confu_mat)
    if num_classes == 2:
        dp = np.sum(confu_mat[:,1]) / total
    else:
#         dp = np.zeros(num_classes)
#         for i in range(num_classes):
#             p_hat = np.sum(confu_mat[:,i])
#             dp[i] = p_hat/total
        dp = np.sum(confu_mat, axis=0)/np.sum(confu_mat)

    return dp

def compute_eo_n_eopp(confu_mat, num_classes):
    
    if num_classes ==2:
        fpr = confu_mat[0,1] / (confu_mat[0,1] + confu_mat[0,0])
        tpr = confu_mat[1,1] / (confu_mat[1,1] + confu_mat[1,0])
        return fpr, tpr
    else:
        fpr_per_class = np.zeros(num_classes)
        tpr_per_class = np.zeros(num_classes)

        for i in range(num_classes):
            pred_given_label = np.sum(confu_mat[i,:])
            label_given_pred = np.sum(confu_mat[:,i])
            fp = label_given_pred - confu_mat[i,i]
            fn = pred_given_label - confu_mat[i,i]
            tp = confu_mat[i,i]
            tn = np.sum(confu_mat) - (pred_given_label+label_given_pred) + tp
            fpr_per_class[i] = fp / (fp+tn)
            tpr_per_class[i] = tp / (tp+fn)
    
        return fpr_per_class, tpr_per_class

def compute_fairness_metric(seed_confus, num_groups):
#     print(len(group_confus))
#     num_classes = len(group_confus[str(0.0)])
    num_classes = len(seed_confus[str(0)])
    dp_list = np.zeros((num_groups, num_classes))
    fpr_list = np.zeros((num_groups, num_classes))
    tpr_list = np.zeros((num_groups, num_classes))
    ap_list = np.zeros((num_groups, num_classes))
    
    for i in range(num_groups):
#         group_confu = group_confus[str(float(i))]
        group_confu = seed_confus[str(int(i))]
        dp_list[i] = compute_dp(group_confu, num_classes)
        fpr, tpr = compute_eo_n_eopp(group_confu, num_classes)
        fpr_list[i] = fpr
        tpr_list[i] = tpr
        ap_list[i] = compute_ap(group_confu, num_classes)

#     dp_1 = compute_dp(confu1,num_classes)
#     dp_2 = compute_dp(confu2,num_classes)
#     fpr1, tpr1 = compute_eo_n_eopp(confu1,num_classes)
#     fpr2, tpr2 = compute_eo_n_eopp(confu2,num_classes)
#     ap_1 = compute_ap(confu1, num_classes)
#     ap_2 = compute_ap(confu2, num_classes)

    return dp_list, fpr_list, tpr_list, ap_list

# TODO : num data average consider
def results_averaged_over_seeds(seeds_confus_path, num_groups, num_classes):
        
    dp_diff = {}
    eo_diff = {}
    eopp_diff = {}
    fpr_list = {}
    tpr_list = {}
    
    ap_diff = {}
    acc = {}
    
    dp_diff['max'] = 0
    dp_diff['avg'] = 0
    eo_diff['max'] = 0
    eo_diff['avg'] = 0
    eopp_diff['max'] = 0
    eopp_diff['avg'] = 0
    ap_diff['max'] = 0
    ap_diff['avg'] = 0
    
    dp_diff['diff_per_class'] = np.zeros(num_classes)
    eopp_diff['diff_per_class'] = np.zeros(num_classes)
    eo_diff['diff_per_class'] = np.zeros(num_classes)
    ap_diff['diff_per_class'] = np.zeros(num_classes)
    
    acc['total'] = 0
#     acc['per_class'] = 0 if num_classes==2 else np.zeros(num_classes)
    acc['per_class'] = np.zeros(num_classes)
    acc['per_group'] = np.zeros(num_groups)
#     acc['per_class_per_group'] = np.zeros(num_groups) if num_classes==2 else np.zeros((num_groups, num_classes))
    acc['per_class_per_group'] = np.zeros((num_groups, num_classes))
    acc['diff_max'] = 0
    acc['diff_avg'] = 0
    acc['diff_per_group'] = 0
    acc['diff_per_class'] = np.zeros(num_classes)
    
    acc['total_var'] = 0
    eopp_diff['avg_var'] = 0
    eopp_diff['max_var'] = 0
    

    
    for i, path in enumerate(seeds_confus_path):
        confu_mat = loadmat(path)
        dp_diff['data'], fpr_list['data'], tpr_list['data'], ap_diff['data'] = compute_fairness_metric(confu_mat, num_groups)
        
        dp_diff['per_class'] = np.max(dp_diff['data'], axis=0) - np.min(dp_diff['data'], axis=0)
        eopp_diff['per_class'] = np.max(tpr_list['data'], axis=0) - np.min(tpr_list['data'], axis=0)
        eo_diff['per_class'] = np.max(fpr_list['data'], axis=0) - np.min(fpr_list['data'], axis=0) + eopp_diff['per_class']
        ap_diff['per_class'] = np.max(ap_diff['data'], axis=0) - np.min(ap_diff['data'], axis=0)
        
        dp_diff['diff_per_class'] += dp_diff['per_class']
        eopp_diff['diff_per_class'] += eopp_diff['per_class']
        eo_diff['diff_per_class'] += eo_diff['per_class']
        ap_diff['diff_per_class'] += ap_diff['per_class']
            
        dp_diff['max'] += np.max(dp_diff['per_class'])
        dp_diff['avg'] += np.mean(dp_diff['per_class'])
        eo_diff['max'] += np.max(eo_diff['per_class'])
        eo_diff['avg'] += np.mean(eo_diff['per_class'])
        eopp_diff['max'] += np.max(eopp_diff['per_class'])
        eopp_diff['max_var'] += (np.max(eopp_diff['per_class'])*100)**2
        eopp_diff['avg'] += np.mean(eopp_diff['per_class'])
        eopp_diff['avg_var'] += (np.mean(eopp_diff['per_class'])*100)**2

        ap_diff['max'] += np.max(ap_diff['per_class'])
        ap_diff['avg'] += np.mean(ap_diff['per_class'])
        
        acc_results = compute_acc(confu_mat, num_groups)
        acc['total'] += acc_results[0]
        acc['total_var'] += (acc_results[0]*100)**2
        
        acc['per_class'] += acc_results[1]
        acc['per_group'] += acc_results[2]
        acc['per_class_per_group'] += acc_results[3]
        
        
        acc['diff_max'] += np.max(np.max(acc_results[3],axis=0) - np.min(acc_results[3], axis=0))
        acc['diff_avg'] += np.mean(np.max(acc_results[3],axis=0) - np.min(acc_results[3], axis=0))
        acc['diff_per_group'] += np.max(acc_results[2]) - np.min(acc_results[2])
        acc['diff_per_class'] += np.max(acc_results[3], axis=0) - np.min(acc_results[3], axis=0)
        
        
    dp_diff['diff_per_class'] /= len(seeds_confus_path)
    eopp_diff['diff_per_class'] /= len(seeds_confus_path)
    eo_diff['diff_per_class'] /= len(seeds_confus_path)
    ap_diff['diff_per_class'] /= len(seeds_confus_path)
    
    dp_diff['max'] /= len(seeds_confus_path)
    dp_diff['avg'] /= len(seeds_confus_path)
    eo_diff['max'] /= len(seeds_confus_path)
    eo_diff['avg'] /= len(seeds_confus_path)
    
    eopp_diff['max'] /= len(seeds_confus_path)
    eopp_diff['avg'] /= len(seeds_confus_path)
    
    eopp_diff['max_var'] /= len(seeds_confus_path)
    eopp_diff['max_var'] -= (eopp_diff['max']*100)**2
    
    eopp_diff['avg_var'] /= len(seeds_confus_path)
    eopp_diff['avg_var'] -= (eopp_diff['avg']*100)**2
    
    
    
    ap_diff['max'] /= len(seeds_confus_path)
    ap_diff['avg'] /= len(seeds_confus_path)
    
    acc['total'] /= len(seeds_confus_path)
    acc['total_var'] /= len(seeds_confus_path)
    acc['total_var'] -= (100*acc['total'])**2
    
    
    
    acc['per_class'] /= len(seeds_confus_path)
    acc['per_group'] /= len(seeds_confus_path)
    acc['per_class_per_group'] /= len(seeds_confus_path)
    
    acc['diff_max'] /= len(seeds_confus_path)
    acc['diff_avg'] /= len(seeds_confus_path)
    acc['diff_per_group'] /= len(seeds_confus_path)
    acc['diff_per_class'] /= len(seeds_confus_path)
    
    
#     dp_diff['per_class'] = np.max(dp_diff['data'],axis=0) - np.min(dp_diff['data'], axis=0)
#     eopp_diff['per_class'] = np.max(tpr_list['data'],axis=0) - np.min(tpr_list['data'], axis=0)
#     eo_diff['per_class'] = np.max(fpr_list['data'],axis=0) - np.min(fpr_list['data'], axis=0) + eopp_diff['per_class']
    
#     acc['diff_max'] = np.max(np.max(acc['per_class_per_group'],axis=0) - np.min(acc['per_class_per_group'], axis=0))
#     acc['diff_avg'] = np.mean(np.max(acc['per_class_per_group'],axis=0) - np.min(acc['per_class_per_group'], axis=0))
#     acc['diff_per_group'] = np.max(acc['per_group']) - np.min(acc['per_group'])
#     acc['diff_per_class'] = np.max(acc['per_class_per_group'],axis=0) - np.min(acc['per_class_per_group'],axis=0)
    
    return acc, dp_diff, eo_diff, eopp_diff, ap_diff
    
            
def compute_metric_diff(group_confus):
    
    num_classes = len(group_confus[0])
    num_groups = len(group_confus)
    dp_list = np.zeros(num_groups, num_classes)
    fpr_list = np.zeros(num_groups, num_classes)
    tpr_list = np.zeros(num_groups, num_classes)
    ap_list = np.zeros(num_groups, num_classes)
    
    for i, group_confu in enumerate(group_confus):
        dp_list[i] = compute_dp(group_confu, num_classes)
        fpr, tpr = compute_eo_n_epoo(group_confu, num_groups)
        fpr_list[i] = fpr
        tpr_list[i] = tpr
        ap_list[i] = compute_ap(group_confu, num_classes)
    
    
#     dp_1 = compute_dp(confu1,num_classes)
#     dp_2 = compute_dp(confu2,num_classes)
#     fpr1, tpr1 = compute_eo_n_eopp(confu1,num_classes)
#     fpr2, tpr2 = compute_eo_n_eopp(confu2,num_classes)
#     ap_1 = compute_ap(confu1, num_classes)
#     ap_2 = compute_ap(confu2, num_classes)
    
    dp_diff = {}
    eopp_diff = {}
    eo_diff = {}
    ap_diff = {}
    
    
    dp_diff = np.abs(dp_1 - dp_2)
    eopp_diff = np.abs(tpr1 - tpr2)
    eo_diff = np.abs(fpr1 - fpr2) + eopp_diff
    ap_diff = np.abs(ap_1 - ap_2)
#     return dp_diff
    return dp_diff, eo_diff, eopp_diff, ap_diff




def compute_acc_for_each_S(confu, num_groups):
    acc_for_each_S = []
    for i in range(num_groups):
        acc_tmp = np.trace(confu[str(i)])/np.sum(confu[str(i)])
        acc_for_each_S.append(acc_tmp)
    return acc_for_each_S

def max_acc_diff(acc_for_each_S):
    acc_list = np.array(acc_for_each_S)
    max_ = np.max(acc_list)
    min_ = np.min(acc_list)
    return max_ - min_

def avg_acc_diff(acc_for_each_S):
    pairwise_diff = [abs(x-y) for i,x in enumerate(acc_for_each_S) for j,y in enumerate(acc_for_each_S) if i != j]
    acc_diff_list = np.array(pairwise_diff)
    
    return np.mean(acc_diff_list)


def compute_accuarcy(confu, num_groups):
    true_positive = 0.0
    total_num = 0.0
    for i in range(num_groups):
        true_positive += np.trace(confu[str(i)])
        total_num += np.sum(confu[str(i)])
    acc = true_positive/total_num
    return acc


def avg_results_for_seed(path_list, num_groups):
    num_of_seed = len(path_list)
    
    confu_list = []
    for i in range(num_of_seed):
        confu_list.append(loadmat(path_list[i]))
        
    max_acc_diff_list = []
    for i in range(num_of_seed):
        max_acc_diff_list.append(max_acc_diff(compute_acc_for_each_S(confu_list[i], num_groups)))
    print('max_acc_diff_list',max_acc_diff_list)
    
    avg_acc_diff_list = []
    for i in range(num_of_seed):
        avg_acc_diff_list.append(avg_acc_diff(compute_acc_for_each_S(confu_list[i], num_groups)))
    print('avg_acc_diff_list',avg_acc_diff_list)
        
    acc_list = []
    for i in range(num_of_seed):
        acc_list.append(compute_accuarcy(confu_list[i], num_groups))
    print('acc_list',acc_list)
    print('acc_std',np.std(np.array(acc_list)))
    return np.average(np.array(acc_list)), np.average(np.array(max_acc_diff_list)), np.average(np.array(avg_acc_diff_list))

def max_DP(confusion_matrix, num_groups, num_classes):
    pred_prob_list = []
    for i in range(num_groups):
        pred_prob = np.sum(confusion_matrix[str(i)],axis=0)/np.sum(confusion_matrix[str(i)])
        pred_prob_list.append(pred_prob)
    max_DP = np.max(np.max(np.array(pred_prob_list),axis=0) - np.min(np.array(pred_prob_list),axis=0))
    return max_DP

def max_Eopp(confusion_matrix, num_groups, num_classes):
    pred_prob_for_each_Y_list = []
    for i in range(num_groups):
        pred_prob_for_each_Y = np.diagonal(confusion_matrix[str(i)])/np.sum(confusion_matrix[str(i)],axis=1)
        pred_prob_for_each_Y_list.append(pred_prob_for_each_Y)
    max_Eopp = np.max(np.max(np.array(pred_prob_for_each_Y_list),axis=0) - np.min(np.array(pred_prob_for_each_Y_list),axis=0))
    return max_Eopp

def mean_Eopp(confusion_matrix, num_groups, num_classes):
    pred_prob_for_each_Y_list = []
    for i in range(num_groups):
        pred_prob_for_each_Y = np.diagonal(confusion_matrix[str(i)])/np.sum(confusion_matrix[str(i)],axis=1)
        pred_prob_for_each_Y_list.append(pred_prob_for_each_Y)
    mean_Eopp = np.mean(np.max(np.array(pred_prob_for_each_Y_list),axis=0) - np.min(np.array(pred_prob_for_each_Y_list),axis=0))
    return mean_Eopp


def compute_eo_n_eopp(confu_for_S, num_classes):
    fpr_per_Y = np.zeros(num_classes)
    tpr_per_Y = np.zeros(num_classes)
    for i in range(num_classes):
        pred_given_label = np.sum(confu_for_S[i,:])
        label_given_pred = np.sum(confu_for_S[:,i])
        fp = label_given_pred - confu_for_S[i,i]
        fn = pred_given_label - confu_for_S[i,i]
        tp = confu_for_S[i,i]
        tn = np.sum(confu_for_S) - (pred_given_label+label_given_pred) + tp
        fpr_per_Y[i] = fp / (fp+tn)
        tpr_per_Y[i] = tp / (tp+fn)
    return fpr_per_Y, tpr_per_Y


def max_Eopp_binarized(confusion_matrix, num_groups, num_classes):    
    tpr_matrix = np.zeros((num_groups, num_classes))
    fpr_matrix = np.zeros((num_groups, num_classes))    
    for i in range(num_groups):
        confu = confusion_matrix[str(i)]
        fpr_per_Y, tpr_per_Y = compute_eo_n_eopp(confu, num_classes)
        fpr_matrix[i] = fpr_per_Y
        tpr_matrix[i] = tpr_per_Y
    
    max_Eopp = np.max(np.max(np.array(tpr_matrix),axis=0) - np.min(np.array(tpr_matrix),axis=0))
    
    return max_Eopp


def average_max_DP_list(path_list, num_groups, num_classes):
    num_of_seed = len(path_list)
    DP_list = []
    for i in range(num_of_seed):
        DP_list.append(max_DP(loadmat(path_list[i]), num_groups, num_classes))
    print('DP_seed :',np.array(DP_list))
    return np.average(np.array(DP_list))

def average_max_Eopp_list(path_list, num_groups, num_classes):
    num_of_seed = len(path_list)
    Eopp_list = []
    for i in range(num_of_seed):
        Eopp_list.append(max_Eopp(loadmat(path_list[i]), num_groups, num_classes))
    print('DEO_max_seed :',np.array(Eopp_list))
    print('DEO_max_std :',np.std(np.array(Eopp_list)))
    return np.average(np.array(Eopp_list))


def average_mean_Eopp_list(path_list, num_groups, num_classes):
    num_of_seed = len(path_list)
    Eopp_list = []
    for i in range(num_of_seed):
        Eopp_list.append(mean_Eopp(loadmat(path_list[i]), num_groups, num_classes))
    print('DEO_avg_seed :',np.array(Eopp_list))
    print('DEO_avg_std :',np.std(np.array(Eopp_list)))
    return np.average(np.array(Eopp_list))


def max_cal(confusion_matrix, num_groups, num_classes):
    pred_prob_for_each_Y_hat_list = []
    for i in range(num_groups):
        pred_prob_for_each_Y_hat = np.diagonal(confusion_matrix[str(i)])/np.sum(confusion_matrix[str(i)],axis=0)
        pred_prob_for_each_Y_hat_list.append(pred_prob_for_each_Y_hat)
    max_cal = np.max(np.max(np.array(pred_prob_for_each_Y_hat_list),axis=0) - np.min(np.array(pred_prob_for_each_Y_hat_list),axis=0))
    return max_cal

def average_max_cal_list(path_list, num_groups, num_classes):
    num_of_seed = len(path_list)
    max_cal_list = []
    for i in range(num_of_seed):
        max_cal_list.append(max_cal(loadmat(path_list[i]), num_groups, num_classes))
    return np.average(np.array(max_cal_list))

def mean_cal(confusion_matrix, num_groups, num_classes):
    pred_prob_for_each_Y_hat_list = []
    for i in range(num_groups):
        pred_prob_for_each_Y_hat = np.diagonal(confusion_matrix[str(i)])/np.sum(confusion_matrix[str(i)],axis=0)
        pred_prob_for_each_Y_hat_list.append(pred_prob_for_each_Y_hat)
    mean_cal = np.mean(np.max(np.array(pred_prob_for_each_Y_hat_list),axis=0) - np.min(np.array(pred_prob_for_each_Y_hat_list),axis=0))
    return mean_cal

def average_mean_cal_list(path_list, num_groups, num_classes):
    num_of_seed = len(path_list)
    mean_cal_list = []
    for i in range(num_of_seed):
        mean_cal_list.append(mean_cal(loadmat(path_list[i]), num_groups, num_classes))
    return np.average(np.array(mean_cal_list))

def groupwise_results(path_seed, num_groups):
    groupwise_acc_set = np.zeros(num_groups)
    groupwise_entropy_set = np.zeros(num_groups)
    groupwise_label_confidence_set = np.zeros(num_groups)
    
    for i in range(num_groups):
        groupwise_index = np.where(loadmat(path_seed)['sen_attr_set'][0]==i)
        groupwise_target = np.int_(loadmat(path_seed)['target_set'][0][groupwise_index])
        groupwise_pred_prob = softmax(loadmat(path_seed)['output_set'],axis=1)[groupwise_index]
        groupwise_pred = np.int_(np.argmax(groupwise_pred_prob, axis=1))
        groupwise_acc = np.sum(groupwise_target==groupwise_pred)/len(groupwise_index[0])
        groupwise_entropy = np.mean(entropy(groupwise_pred_prob, axis=1))
        groupwise_label_confidence = np.mean(groupwise_pred_prob[range(len(groupwise_pred_prob)),groupwise_target])
        
        groupwise_acc_set[i] = groupwise_acc
        groupwise_entropy_set[i] = groupwise_entropy
        groupwise_label_confidence_set[i] = groupwise_label_confidence
    return groupwise_acc_set, groupwise_entropy_set, groupwise_label_confidence_set


def grouplabelwise_results(path_seed, num_groups, num_classes):
    grouplabelwise_acc_set = np.zeros((num_groups, num_classes))
    grouplabelwise_entropy_set = np.zeros((num_groups, num_classes))
    grouplabelwise_label_confidence_set = np.zeros((num_groups, num_classes))

    for i in range(num_groups):
        for j in range(num_classes):
            grouplabelwise_index = np.where(((loadmat(path_seed)['sen_attr_set']==i) & (loadmat(path_seed)['target_set']==j))[0])
            grouplabelwise_target = np.int_(loadmat(path_seed)['target_set'][0][grouplabelwise_index])
            grouplabelwise_pred_prob = softmax(loadmat(path_seed)['output_set'],axis=1)[grouplabelwise_index]
            grouplabelwise_pred = np.int_(np.argmax(grouplabelwise_pred_prob, axis=1))
            grouplabelwise_acc = np.sum(grouplabelwise_target==grouplabelwise_pred)/len(grouplabelwise_index[0])
            grouplabelwise_entropy = np.mean(entropy(grouplabelwise_pred_prob, axis=1))
            grouplabelwise_label_confidence = np.mean(grouplabelwise_pred_prob[range(len(grouplabelwise_pred_prob)),grouplabelwise_target])
            
            grouplabelwise_acc_set[i,j] = grouplabelwise_acc
            grouplabelwise_entropy_set[i,j] = grouplabelwise_entropy
            grouplabelwise_label_confidence_set[i,j] = grouplabelwise_label_confidence
        
    return grouplabelwise_acc_set, grouplabelwise_entropy_set, grouplabelwise_label_confidence_set


def draw_groupwise_graph(path_seed, num_groups):
    groupwise_acc_set, groupwise_entropy_set, groupwise_label_confidence_set = groupwise_results(path_seed, num_groups)
    plt.subplot(1, 3, 1)
    plt.bar(range(num_groups), groupwise_acc_set, width=0.5)
    plt.xticks(np.arange(num_groups), ['Male', 'Female'], fontsize=15)
    plt.xlabel('Group', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim((0.7,1))
    plt.title('Group-wise Accuracy', fontsize=15)

    plt.subplot(1, 3, 2)
    plt.bar(range(num_groups), groupwise_entropy_set, width=0.5)
    plt.xticks(np.arange(num_groups), ['Male', 'Female'], fontsize=15)
    plt.xlabel('Group', fontsize=15)
    plt.ylabel('Entropy', fontsize=15)
    plt.ylim((0,1.1))
    plt.title('Group-wise Entropy', fontsize=15)


    plt.subplot(1, 3, 3)
    plt.bar(range(num_groups), groupwise_label_confidence_set, width=0.5)
    plt.xticks(np.arange(num_groups), ['Male', 'Female'], fontsize=15)
    plt.xlabel('Group', fontsize=15)
    plt.ylabel('Label confidence', fontsize=15)
    plt.ylim((0.6,1))
    plt.title('Group-wise Label confidence', fontsize=15)

    plt.subplots_adjust(left=0.0,
                        bottom=0.1, 
                        right=1.8, 
                        top=0.7, 
                        wspace=0.4, 
                        hspace=0.4)
    plt.show()
    
    
def draw_grouplabelwise_graph(path_seed, num_groups, num_classes, eval_set):
    grouplabelwise_acc_set, grouplabelwise_entropy_set, grouplabelwise_label_confidence_set = grouplabelwise_results(path_seed, num_groups, num_classes)
    color = iter(cm.rainbow(np.linspace(0.2, 2, 3)))
    width = 0.5
    plt.subplot(1, 3, 1)
    print(np.max(grouplabelwise_acc_set, axis=0) - np.min(grouplabelwise_acc_set, axis=0))
    for i in range(num_groups):
        plt.bar(np.arange(num_classes)-width/2+0.5*width*(2*i+1)/num_groups, grouplabelwise_acc_set[i], width=width/num_groups,  label='Group'+str(i))
    plt.xticks(np.arange(num_classes), fontsize=15)
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim((0.6,1))
    plt.title('Group & label-wise Accuracy : {}'.format(eval_set), fontsize=15)
    plt.legend()
    plt.subplot(1, 3, 2)
    for i in range(num_groups):
        plt.bar(np.arange(num_classes)-width/2+0.5*width*(2*i+1)/num_groups, grouplabelwise_entropy_set[i], width=width/num_groups,  label='Group'+str(i))
    plt.xticks(np.arange(num_classes), fontsize=15)
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('Entropy', fontsize=15)
    plt.ylim((0,1.1))
    plt.title('Group & label-wise Entropy : {}'.format(eval_set), fontsize=15)
    plt.legend()
    plt.subplot(1, 3, 3)
    for i in range(num_groups):
        plt.bar(np.arange(num_classes)-width/2+0.5*width*(2*i+1)/num_groups, grouplabelwise_label_confidence_set[i], width=width/num_groups,  label='Group'+str(i))
    plt.xticks(np.arange(num_classes), fontsize=15)
    plt.xticks(np.arange(num_classes), fontsize=15)
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('Label confidence', fontsize=15)
    plt.ylim((0.5,1))
    plt.title('Group & label-wise Label confidence'.format(eval_set), fontsize=15)
    plt.legend()
    plt.subplots_adjust(left=0.0,
                        bottom=0.1,
                        right=1.8,
                        top=0.7,
                        wspace=0.4,
                        hspace=0.4)
    plt.show()
    return np.max(grouplabelwise_acc_set, axis=0) - np.min(grouplabelwise_acc_set, axis=0)
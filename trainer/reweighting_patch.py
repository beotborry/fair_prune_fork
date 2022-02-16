from cmath import exp
import enum
import imp
from tokenize import Exponent, group


import torch

# update weight
def debias_weights(self, target_criterion, label, sen_attrs, extended_multipliers, num_groups, num_classes):  #
    if target_criterion == "eo" or target_criterion == "dp":
        weights = torch.zeros(len(label))
        w_matrix = torch.sigmoid(extended_multipliers) # g by c
        weights = w_matrix[sen_attrs, label]
    
    elif target_criterion == "eopp":
        exponents = torch.zeros(len(label))

        for i, m in enumerate[:, 0]:
            group_idxs = torch.where(sen_attrs == i)[0]
            exponents[group_idxs] -= m
        weights = torch.exp(exponents) / (torch.exp(exponents) + torch.exp(-exponents))
        weights = torch.where(label == 1, 1 - weights, weights)
    
    # for i in range(num_groups):
    #         group_idxs = torch.where(sen_attrs == i)[0]
    #         w_tilde = torch.exp(extended_multipliers[i])
    #         weights[group_idxs] += w_tilde[label[group_idxs]]
    #         weights[group_idxs] /= torch.sum(torch.exp(extended_multipliers), axis=0)[label[group_idxs]] #
                
    return weights

def get_error_and_violations_EOPP(self, y_pred, label, sen_attrs, num_groups=2, num_classes=2):
    acc = torch.mean((y_pred == label).float())
    total_num = len(y_pred)
    violations = torch.zeros((num_groups, num_classes))
    for g in range(num_groups):
        c = 1
        class_idxs = torch.where(label==c)[0]
        pred_class_idxs = torch.where(torch.logical_and(y_pred == c, label == c))[0]
        pivot = len(pred_class_idxs) / len(class_idxs) # P(y_hat = 1 | y = 1)
        group_class_idxs = torch.where(torch.logical_and(sen_attrs == g, label == c))[0]
        group_pred_class_idxs = torch.where(torch.logical_and(torch.logical_and(sen_attrs == g, y_pred == c), label == c))[0]
        violations[g, 0] = pivot - len(group_pred_class_idxs)/len(group_class_idxs)
        # violations[g, 0] = len(group_pred_class_idxs)/len(group_class_idxs) - pivot # P(y_hat = 1 | g = g, y = 1) - P(y_hat = 1 | y = 1)
        violations[g, 1] = violations[g, 0]
    print('violations', violations)
    return acc, violations
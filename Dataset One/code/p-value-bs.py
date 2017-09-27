# -*- coding: utf-8 -*-
"""
This script aims at validating the p-value significance from the T-test.
"""

import operator
import os
import time
import bisect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats



# the Dataset class encapsulate the information about a data set file and the related selected gene combination
class Dataset:
    def __init__(self, data, dataname, ft_idx, pos, neg, ft_rg, imb=0):
        self.data = data
        self.dataname = dataname
        self.pos = pos
        self.neg = neg
        self.imb = imb
        self.ft_idx = ft_idx
        self.ft_rg = ft_rg

# data set information and selected genes on 17 data sets
datasets = [
    Dataset(data="DLBCL", dataname="DLBCL", ft_idx="Genes", pos="DLBCL", neg="follicular_lymphoma",
            ft_rg=[12, 53, 38]),
    Dataset(data="Pros", dataname="Prostate", ft_idx="Genes", pos="Normal", neg="Tumour",
            ft_rg=[73, 0, 693, 14]),
    Dataset(data="Colon", dataname="Colon", ft_idx="Genes", pos="Normal tissue", neg="Tumor tissue",
            ft_rg=[224, 494, 14, 239, 175, 57]),
    Dataset(data="Leuk", dataname="Leukaemia", ft_idx="Golubtemplate", pos="ALL", neg="AML",
            ft_rg=[3, 6]),
    Dataset(data="Mye", dataname="Myeloma", ft_idx="Presence_of_bone_lesions", pos="TRUE", neg="FALSE",
            ft_rg=[403, 142, 14, 2, 377, 568, 82]),
    Dataset(data="ALL1", dataname="ALL1", ft_idx="BT", pos="Bcell", neg="Tcell",
            ft_rg=[0]),
    Dataset(data="ALL2", dataname="ALL2", ft_idx="relapse", pos="TRUE", neg="FALSE",
            ft_rg=[0, 736, 521, 77, 686, 79, 51, 759]),
    Dataset(data="ALL3", dataname="ALL3", ft_idx="mdr", pos="POS", neg="NEG",
            ft_rg=[769, 3, 487, 74, 714, 141, 51, 509]),
    Dataset(data="ALL4", dataname="ALL4", ft_idx="chr", pos="TRUE", neg="FALSE",
            ft_rg=[0, 5, 38, 753, 534, 281]),
    Dataset(data="CNS", dataname="CNS", ft_idx="gene", pos="POS", neg="NEG",
            ft_rg=[129, 130, 519, 8, 271, 272, 52]),
    Dataset(data="Lym", dataname="Lymphoma", ft_idx="gene", pos="ACL", neg="GCL",
            ft_rg=[3, 668, 4]),
    Dataset(data="Adeno", dataname="Adenoma", ft_idx="gene", pos="Normal", neg="Tumor",
            ft_rg=[467]),
    Dataset(data="Gas", dataname="Gastric", ft_idx="gene", pos="non-malignant", neg="tumor",
            ft_rg=[305, 76, 21]),
    Dataset(data="Gas1", dataname="Gastric1", ft_idx="gene", pos="N", neg="T",
            ft_rg=[131, 716, 247]),
    Dataset(data="Gas2", dataname="Gastric2", ft_idx="gene", pos="N", neg="T",
            ft_rg=[88, 37]),
    Dataset(data="T1D", dataname="T1D", ft_idx="gene", pos="uh", neg="td",
            ft_rg=[577, 679, 13, 558, 112, 977, 24]),
    Dataset(data="Stroke", dataname="Stroke", ft_idx="Probe_Set_ID", pos="CTRL", neg="IS",
            ft_rg=[0, 128, 275, 22])

]

# the record about time and settings about input and output
time_format = "-%m-%d-%H-%M-%S"
cur_time = time.strftime(time_format, time.localtime())
data_dir = "D:\\codes\\python\\MGRFE\\Dataset One\\data\\"
output_path = "D:\\codes\\python\\MGRFE\\Dataset One\\results\\AUC_RST\\"
output_stats = output_path + "all_aucs" + cur_time + ".txt"


# # guarantee the availability of the output path
# if not os.path.exists(output_path):
#     os.mkdir(output_path)
# out_file = open(output_stats, mode='w')

# for each data set
for i in range(len(datasets)):
    dataset = datasets[i]
    metrics = []
    file_path = data_dir + dataset.dataname + ".txt"

    # load the data and get the data set (X, y)
    df = pd.read_csv(file_path, sep="\t")
    df = df.set_index(dataset.ft_idx)
    classes = df.columns
    features = df.index
    pos = list(filter(lambda x: x.find(dataset.pos) != -1, classes))
    neg = list(filter(lambda x: x.find(dataset.neg) != -1, classes))
    t_vals, p_vals = stats.ttest_ind(df.ix[:, pos], df.ix[:, neg], axis=1)
    p_vals.sort()
    print(dataset.data)
    print(bisect.bisect(p_vals,0.05))


#     X = np.array(df.values).T
#     y = np.array([1] * pos_num + [0] * neg_num)
#
#     # data preprocessing
#     X = preprocessing.normalize(X)
#     X = X[:, dataset.ft_rg]
#
#
#     # ##########################################################
#     # Classification and ROC analysis
#
#     # Run classifier with cross-validation and plot ROC curves
#     print("\n# AUC calculation of %s"%(dataset.dataname))
#     out_file.write("\n# AUC calculation of %s\n"%(dataset.dataname))
#     cv = StratifiedKFold(n_splits=K, random_state=i, shuffle=True)
#
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#
#     i = 0
#     for train, test in cv.split(X, y):
#         classifier = GaussianNB()
#         probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
#         # Compute ROC curve and area the curve
#         fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#         plt.plot(fpr, tpr, lw=1, alpha=0.5,
#                  label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
#         i += 1
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#              label='Luck', alpha=.8)
#
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = np.mean(aucs)
#     std_auc = np.std(aucs)
#     print("AUC in K-fold: %r"%(aucs))
#     print("Mean: %0.2f, std: %0.2f"%(mean_auc,std_auc))
#     out_file.write("AUC in K-fold: %r\n"%(aucs))
#     out_file.write("Mean: %0.2f, std: %0.2f\n\n"%(mean_auc,std_auc))
#     all_aucs.append(mean_auc)
#     plt.plot(mean_fpr, mean_tpr, color='b',
#              label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#              lw=2, alpha=.8)
#
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4,
#                      label=r'$\pm$ 1 std. dev.')
#
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate',fontsize=16)
#     plt.ylabel('True Positive Rate',fontsize=16)
#     plt.title('ROC and AUC of ' + dataset.dataname,fontsize=18)
#     plt.legend(loc="lower right")
#     plt.savefig(output_path + dataset.dataname + cur_time + ".pdf")
#     plt.close()
#
#
# print("\n# AUC of all datasets:")
# print(str(all_aucs))
# out_file.write("\n# AUC of all datasets:\n")
# out_file.write(str(all_aucs)+"\n")
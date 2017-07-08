# -*- coding: utf-8 -*-
import operator
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB


class Dataset:
    def __init__(self, data, dataname, ft_idx, pos, neg, ft_rg, imb=0):
        self.data = data
        self.dataname = dataname
        self.pos = pos
        self.neg = neg
        self.imb = imb
        self.ft_idx = ft_idx
        self.ft_rg = ft_rg


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

ga_accs =  [1, 0.981, 0.985, 1, 0.937]
mc_accs =  [1,  0.95,  0.90, 1, 0.85]
ga_feats = [3, 3, 6, 2, 7]
mc_feats = [4, 3, 6, 2, 7]
data_names = ['DLBCL', 'Pros', 'Colon', 'Leuk', 'Mye']

ga_accs +=  [ 1, 0.91, 0.927, 0.99, 1, 1, 1]
mc_accs +=  [ 1, 0.75,  0.80, 0.88, 0.85, 1, 1]
ga_feats += [ 1, 8, 8, 6, 7, 3, 1]
mc_feats += [ 1, 2, 5, 2, 4, 4, 2]
data_names += [ 'ALL1', 'ALL2', 'ALL3', 'ALL4', 'CNS', 'Lym', 'Adeno']

ga_accs  += [    1, 0.98, 1, 0.91,    1]
mc_accs  += [ 0.97, 0.95, 1, 0.81, 0.85]
ga_feats += [ 3, 3, 2, 7, 4]
mc_feats += [ 3, 4, 2, 6, 1]
data_names += ['Gas', 'Gas1', 'Gas2', 'T1D', 'Stroke']


time_format = "-%m-%d-%H-%M-%S"
cur_time = time.strftime(time_format, time.localtime())
data_dir = "D:\\codes\\python\\MachineLearning\\McTwo\\"
output_path = "D:\\codes\\python\\MachineLearning\\Python_Lesson\\ftselect\\final_version\\cv\\"
output_stats = output_path + "whole_rst_cv" + cur_time + ".txt"
output_figure = output_path + "whole_rst_cv" + cur_time + ".pdf"
overall_metrics = []

time_format = "-%m-%d-%H-%M-%S"
cur_time = time.strftime(time_format, time.localtime())
data_dir = "D:\\codes\\python\\MachineLearning\\McTwo\\"
output_path = "D:\\codes\\python\\MachineLearning\\Python_Lesson\\ftselect\\final_version\\cv\\"
output_stats = output_path + "whole_rst_cv" + cur_time + ".txt"
output_figure = output_path + "whole_rst_cv" + cur_time + ".pdf"
overall_metrics = []

# K fold
K = 10
# T times k-fold
T = 10

if not os.path.exists(output_path):
    os.mkdir(output_path)
out_file = open(output_stats, mode='w')


# calculate acc
def calc_acc(con_matrix):
    TN = con_matrix[0][0]
    FP = con_matrix[0][1]
    FN = con_matrix[1][0]
    TP = con_matrix[1][1]
    P = TP + FN
    N = TN + FP
    Acc = (TP + TN) / (P + N)
    return Acc


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
    top_fts = sorted(list(zip(features, p_vals)), key=operator.itemgetter(1))
    features = [x[0] for x in top_fts]
    features_positions = dict(zip(features, range(len(features))))
    df = df.ix[:, pos + neg]
    df = df.ix[features, :]
    pos_num = len(pos)
    neg_num = len(neg)

    X = np.array(df.values).T
    y = np.array([1] * pos_num + [0] * neg_num)

    # data preprocessing
    X = preprocessing.normalize(X)

    # T  times cross-validation
    for j in range(T):
        if(j==0):
            print("\n# CV of " + dataset.data)
            out_file.write("\n# CV of " + dataset.data)
            print([x for x in dataset.ft_rg])
            print([features[x] for x in dataset.ft_rg])
            out_file.write(str([x for x in dataset.ft_rg]) + "\n")
            out_file.write(str([features[x] for x in dataset.ft_rg]) + "\n")
        mt_clf = 0
        X_cur = X[:, dataset.ft_rg]
        y_cur = y

        out_file.write(str(X_cur.shape) + "\n")
        skf = StratifiedKFold(n_splits=K, random_state=j, shuffle=True)
        clf = []
        for train_index, test_index in skf.split(X_cur, y_cur):

            clf = GaussianNB()
            X_train = X_cur[train_index, :]
            X_test = X_cur[test_index, :]
            y_train = y_cur[train_index]
            y_test = y_cur[test_index]
            clf.fit(X_train, y_train)
            predicts = clf.predict(X_test)
            result = confusion_matrix(y_test, predicts, labels=[0, 1])
            out_file.write(str(result) + "\n")
            mt_clf = mt_clf + calc_acc(result)
        metrics.append(mt_clf / K)
    print(metrics)
    out_file.write(str(metrics) + "\n")
    overall_metrics.append(metrics)

print("\n# Mean acc of all datasets:")
print([sum(x) / T for x in overall_metrics])
out_file.write("\n# Mean acc of all datasets:")
out_file.write(str([sum(x) / T for x in overall_metrics]))

plt.figure(figsize=(15, 7), dpi=80)
plt.boxplot(overall_metrics, labels=data_names)
plt.yticks([0.1 * i for i in range(5, 11)])
plt.ylim(0.5, 1.1)
plt.title(str(T) + " times " +str(K) + "-fold cross validation", fontsize=16)
plt.ylabel('Acc', fontsize=15)

plt.savefig(output_figure)
plt.show()
plt.close()
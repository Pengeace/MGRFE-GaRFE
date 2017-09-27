# -*- coding: utf-8 -*-
"""
This script implements the multiple k-fold cross validation
to validate the classification performance of the selected gene combination on each data set.

By default, 10 times 10-fold cross validation will be processed
and the parameters can be adjusted if needed.
"""

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

# =====================================================
# the Dataset class contains the file information and
# selected gene combination of a data set
# =====================================================
class Dataset:
    def __init__(self, data, dataname, cls, names, ft_idx, ft_rg):
        self.data = data
        self.dataname = dataname
        self.classes = cls
        self.cls_names = names
        self.ft_idx = ft_idx
        self.ft_rg = ft_rg

datasets = [
    Dataset(data="SRBCT", dataname="SRBCT", cls=4, names=["EWS", "BL", "NB", "RMS"], ft_idx="Image_Id.",
            ft_rg=[2049, 1954, 713, 1707, 1433] ),
    Dataset(data="ALL_AML", dataname="ALL_AML", cls=2, names=["ALL","AML"], ft_idx="gene",
            ft_rg=[3, 6]),
    Dataset(data="MLL", dataname="MLL", cls=3, names=["ALL", "AML", "MLL"], ft_idx="Name",
            ft_rg=[8936, 11296, 3276])
]


# time record and settings about the input and output path
time_format = "-%m-%d-%H-%M-%S"
cur_time = time.strftime(time_format, time.localtime())
file_path = "D:\\codes\\python\\MGRFE\\Dataset Two\\data\\"
output_path = "D:\\codes\\python\\MGRFE\\Dataset Two\\results\\CV_RST\\"
output_stats = output_path + "whole_rst_cv" + cur_time + ".txt"
output_figure = output_path + "whole_rst_cv" + cur_time + ".pdf"
overall_metrics = []

# K-fold
K = 10
# T times k-fold
T = 10

# guarantee the availability of the output path
if not os.path.exists(output_path):
    os.mkdir(output_path)
out_file = open(output_stats, mode='w')


# calculate acc from confusion matrix
def calc_acc(con_matrix):
    length = len(con_matrix)
    right = 0
    all = 0
    for i in range(length):
        all = all + sum(con_matrix[i])
        right = right + con_matrix[i][i]
    Acc = right / all
    return Acc

# deal with each dataset
for i in range(len(datasets)):
    dataset = datasets[i]
    metrics = []
    data_path = file_path + dataset.dataname + ".txt"

    # load the data and get the data set (X, y)
    df = pd.read_csv(data_path, sep="\t")
    df = df.set_index(dataset.ft_idx)
    classes = df.columns
    features = df.index
    cls = [0] * dataset.classes
    for i in range(dataset.classes):
        cls[i] = list(filter(lambda x: x.find(dataset.cls_names[i]) != -1, classes))

    whole_sample = []
    for i in range(dataset.classes):
        whole_sample = whole_sample + cls[i]

    df = df.ix[:, whole_sample]

    if(dataset.classes==2):
        t_vals, p_vals = stats.ttest_ind(df.ix[:, cls[0]], df.ix[:, cls[1]], axis=1)
        top_fts = sorted(list(zip(features, p_vals)), key=operator.itemgetter(1))
        features = [x[0] for x in top_fts]
        df = df.ix[features, :]

    X = np.array(df.values).T
    y = []
    for i in range(dataset.classes):
        y = y + [i] * len(cls[i])
    y = np.array(y)

    if(dataset.dataname!="SRBCT"):
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

        # print(X_cur.shape)
        # out_file.write(str(X_cur.shape) + "\n")
        skf = StratifiedKFold(n_splits=K, random_state=j, shuffle=True)
        clf = []
        for train_index, test_index in skf.split(X_cur, y_cur):
            # print("train_index:" + str(train_index))
            # print("test_index:" + str(test_index))
            # out_file.write("train_index:" + str(train_index) + "\n")
            # out_file.write("test_index:" + str(test_index) + "\n")
            clf = GaussianNB()
            X_train = X_cur[train_index, :]
            X_test = X_cur[test_index, :]
            y_train = y_cur[train_index]
            y_test = y_cur[test_index]
            clf.fit(X_train, y_train)
            predicts = clf.predict(X_test)
            result = confusion_matrix(y_test, predicts, list(range(dataset.classes)))
            # print(result)
            # out_file.write(str(result) + "\n")
            mt_clf = mt_clf + calc_acc(result)
        metrics.append(mt_clf / K)
    print(metrics)
    out_file.write(str(metrics) + "\n")
    overall_metrics.append(metrics)

print("\n# Mean acc of all datasets:")
print([sum(x) / T for x in overall_metrics])
out_file.write("\n# Mean acc of all datasets:")
out_file.write(str([sum(x) / T for x in overall_metrics]))

plt.figure(figsize=(8, 5), dpi=80)
plt.boxplot(overall_metrics, labels=[x.data for x in datasets])
plt.yticks([0.1 * i for i in range(5, 11)])
plt.ylim(0.5, 1.1)
ax = plt.gca()
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(12)
plt.title(str(T) + ' times ' + str(K) +"-fold cross validation", fontsize=16)
plt.ylabel('$Acc$', fontsize=14)

plt.savefig(output_figure)
plt.show()
plt.close()
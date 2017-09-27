# -*- coding: utf-8 -*-
"""
This script implements the micro average and macro average calculation by k-fold cross validation
to validate the classification performance of the selected gene combination on each data set.

By default, 5-fold cross validation will be processed
and the parameter can be adjusted if needed.
"""

import operator
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
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
output_path = "D:\\codes\\python\\MGRFE\\Dataset Two\\results\\macro_micro_RST\\"
output_stats = output_path + "mirco_macro" + cur_time + ".txt"
overall_metrics = []

# K-fold CV
K = 5

# guarantee the availability of the output path
if not os.path.exists(output_path):
    os.mkdir(output_path)
out_file = open(output_stats, mode='w')


# calculate Acc, in fact, here Acc is micro average f1 measure
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
    micro_average=np.array([0.0]*3)
    macro_average=np.array([0.0]*3)
    acc=0.0
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
        X = preprocessing.normalize(X)
    X = X[:, dataset.ft_rg]

    # Run classifier with cross-validation
    print("\n# Micro/macro calculation of %s" % (dataset.dataname))
    out_file.write("\n# Micro/macro calculation of %s\n" % (dataset.dataname))
    cv = StratifiedKFold(n_splits=K, random_state=i, shuffle=True)

    for train, test in cv.split(X, y):
        clf = GaussianNB()
        X_train = X[train, :]
        X_test = X[test, :]
        y_train = y[train]
        y_test = y[test]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        micro_average += precision_recall_fscore_support(y_test, y_pred, average="micro")[0:3]
        macro_average += precision_recall_fscore_support(y_test, y_pred, average="macro")[0:3]
        result = confusion_matrix(y_test, y_pred, list(range(dataset.classes)))
        acc += calc_acc(result)

    micro_average /= K
    macro_average /= K
    acc /= K
    print("Micro average precision recall f1: %r" % (micro_average))
    print("Macro average precision recall f1: %r" % (macro_average))
    print("Acc: %r" % (acc))
    out_file.write("Micro average precision recall f1: %r\n" % (micro_average))
    out_file.write("Macro average precision recall f1: %r\n" % (macro_average))
    out_file.write("Acc: %r\n" % (acc))
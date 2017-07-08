# -*- coding: utf-8 -*-
import math
import operator
import os
import random
import time
import sys
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minepy import MINE
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

sys.path.append("D:\\codes\\python\\MachineLearning\\further_exp\\fur_test\\codes\\")

from individual import Individual
from population import Population

# calculate the metrics
def calc_metrics(con_matrix):
    length = len(con_matrix[0])
    T = [0] * length
    NUM = [0] * length
    for i in range(length):
        T[i] = con_matrix[i][i]
        NUM[i] = sum(con_matrix[i])

        # avoid error of division by zero
        if(NUM[i]==0):
            T[i] = NUM[i] = 1

    Acc = (sum(T) / sum(NUM))
    Avc = 0.0
    for i in range(length):
        Avc = Avc + int(T[i]) / int(NUM[i])
    Avc /= length
    return [Acc, Avc]


# the parameter search not used, pure K-fold
def calc_fitness(pop):
    rsts = list()
    for gene_rg in pop:
        indiv = Individual()
        indiv.ft_rg = gene_rg
        indiv.gene_len = len(gene_rg)
        # train test
        mt_clf_ta = np.array([0.0] * len(metrics_names))
        mt_clf_te = np.array([0.0] * len(metrics_names))

        X_cur = X[:, gene_rg]
        skf = StratifiedKFold(n_splits=K, random_state=random.randint(0, 100), shuffle=True)
        clf = []
        for train_index, test_index in skf.split(X_cur, y):
            clf = GaussianNB()
            X_train = X_cur[train_index, :]
            X_test = X_cur[test_index, :]
            y_train = y[train_index]
            y_test = y[test_index]
            clf.fit(X_train, y_train)
            # print(clf.best_params_)
            predicts = clf.predict(X_test)
            result = confusion_matrix(y_test, predicts, labels=list(range(cls_size)))
            mt_clf_ta += np.array(calc_metrics(result))

            predicts = clf.predict(X_train)
            result = confusion_matrix(y_train, predicts, labels=list(range(cls_size)))
            mt_clf_te += np.array(calc_metrics(result))
        indiv.metrics_ta = list(map(lambda x: x / K, mt_clf_ta))
        indiv.metrics_te = list(map(lambda x: x / K, mt_clf_te))
        indiv.clf = clf
        indiv.acc = 0.5 * indiv.metrics_ta[0] + 0.5 * indiv.metrics_te[0]
        indiv.index = indiv.acc
        rsts.append(indiv)
    return rsts


class APop(Population):
    def __init__(self, gene_len, potential_gene_rg, pop_size=100,
                 generation_size=10, pc=0.6, pm=0.15, best_size=5):
        self.pop_size = pop_size
        self.gene_len = gene_len
        self.potential_gene_rg = potential_gene_rg
        self.generation_size = generation_size  # varies according to the gene_len
        self.pc = pc
        self.pm = pm
        self.best_size = best_size
        self.mean_fit = []
        self.generation_cur = 0


        if(self.gene_len>50):
            self.generation_size = 1
        elif(self.gene_len>35):
            self.generation_size = 2
        else:
            self.generation_size = 3

    def calc_fitness(self, pop):
        return calc_fitness(pop)

class RFE():
    K = 5
    # time format
    time_format = "-%m-%d-%H-%M-%S"

    def __init__(self, dataset, initial_ft_rg, cur_layer, rfe_num, file_path, max_gene_len,
                 cls_size, cls_names, index_name, pop_best_size=5, pop_size=90, best_size=30):

        self.dataset_name = dataset
        self.initial_ft_rg = initial_ft_rg
        self.cur_layer = cur_layer
        self.rfe_num = rfe_num

        self.cur_version = dataset_name + "_v2.0"
        self.cur_time = time.strftime(self.time_format, time.localtime())
        self.file_path = file_path
        self.output_path = "D:\\codes\\python\\MachineLearning\\Python_Lesson\\ftselect\\populations_f\\" + dataset_name + "\\"
        self.plot_name = dataset + "-layer-" + str(cur_layer) + "-rfe-" + str(rfe_num) + self.cur_time + ".pdf"
        self.stats_name = dataset + "-layer-" + str(cur_layer) + "-rfe-" + str(rfe_num) + self.cur_time + ".txt"
        self.cls_size = cls_size
        self.cls_names = cls_names
        self.ft_index_name = index_name
        self.max_gene_len = max_gene_len

        self.rfe_info = "# RFE-" + str(self.rfe_num) + "\n" + \
                        "# layer:" + str(self.cur_layer) + "\n" + \
                        "# GaussianNB() \n" + \
                        "# Shuffle processed in " + str(self.K) + "-Fold \n"

        # settings about the populations and evolution
        self.pop_size = pop_size  # population size for GA
        self.best_size = best_size  # RFE best size
        self.pop_best_size = pop_best_size  # population best size

    def prepare(self):
        print("\n\n\n**************\n")
        print(self.initial_ft_rg)
        print(len(self.initial_ft_rg))
        print(self.max_gene_len)
        self.first_pop_gene = [random.sample(self.initial_ft_rg, self.max_gene_len) for i in range(self.pop_size)]

    # RFE and output
    def down(self):
        self.prepare()
        whole_pop = []
        best_ones = []
        whole_metrics = []

        whole_pop_size = 0
        cur_gene_len = self.max_gene_len

        # evolution and feature elimination process
        while cur_gene_len >= 1:

            new_pop = APop(gene_len=cur_gene_len, potential_gene_rg=self.initial_ft_rg,
                           pop_size=self.pop_size,
                           best_size=self.best_size)

            if whole_pop_size == 0:
                new_pop.set_pop(self.first_pop_gene)
            else:
                new_pop.set_pop(whole_pop[whole_pop_size - 1].pop)
            new_pop.evolve()
            best_ones = sorted(best_ones + new_pop.best_cur, reverse=True)[0:self.best_size]

            whole_metrics.append([cur_gene_len, new_pop.mean_fit[-1], new_pop.best_cur[0].index])
            whole_pop.append(new_pop)
            whole_pop_size = whole_pop_size + 1
            print("whole_pop_size:", whole_pop_size)
            print("cur_gene_len : ", cur_gene_len)
            print("best_cur : \n", new_pop.best_cur[0])
            print("mean_fit & best_fit", whole_metrics[-1])
            if cur_gene_len > 50:
                cur_gene_len = cur_gene_len - 3
            elif cur_gene_len > 30:
                cur_gene_len = cur_gene_len - 2
            else:
                cur_gene_len = cur_gene_len - 1

        ###############################

        # return best individuals to this layer
        global best_ones_layer
        if (type(best_ones_layer[self.cur_layer]) is int):
            best_ones_layer[self.cur_layer] = best_ones
        else:
            best_ones_layer[self.cur_layer] += best_ones

        ################################

        # output and save the results
        whole_metrics = sorted(whole_metrics, key=operator.itemgetter(0))
        out_file = open(output_path + self.stats_name, mode='w')
        print("/***Generated from " + cur_version + "***/\n")
        out_file.write("/***Generated from " + cur_version + "***/\n")
        print("\n" + self.rfe_info + "\n")
        out_file.write("\n" + self.rfe_info + "\n")
        print("\n------- Overall statistics ---------\n")
        out_file.write("\n------------ Overall statistics -------------\n")
        print("\ngene_len\tmean_fit\tbest_fit\n")
        out_file.write("\ngene_len\t\tmean_fit\t\tbest_fit\n")
        for x in whole_metrics:
            out_file.write(str(x[0]) + "\t\t" + str(round(x[1], 3)) + "\t\t" + str(round(x[2], 3)) + "\n")
            print(x)
        print("\n\n----------- Overall best individuals -----------\n\n")
        out_file.write("\n\n----------- Overall best individuals -----------\n\n")
        for i in range(self.best_size):
            print(best_ones[i].shorter_str())
            out_file.write(best_ones[i].shorter_str() + "\n")
        print("\n\n----------- Overall best features -----------\n")
        out_file.write("\n\n----------- Overall best features -----------\n")

        best_ft_range = set(reduce(lambda x, y: x + y, [indiv.ft_rg for indiv in best_ones]))
        indiv_ft_rgs = [dict(zip(x.ft_rg, [1] * len(x.ft_rg))) for x in best_ones]
        whole_best_fts = list(zip(best_ft_range,
                                  [sum([x[f] for x in indiv_ft_rgs if f in x.keys()]) for f in
                                   best_ft_range]))
        whole_best_fts = sorted(whole_best_fts, key=operator.itemgetter(1), reverse=True)
        print("\nvotes\t\tcur_pos\t\tfeature_name\n")
        out_file.write("\nvotes\t\tcur_pos\t\tfeature_name\n")
        for x in whole_best_fts:
            print(x[1], "\t\t", x[0], "\t\t", features[x[0]])
            out_file.write(str(x[1]) + "\t\t" + str(x[0]) + "\t\t" + str(features[x[0]]) + "\n")
        print("\nThe features voted more than or equal to 5 times are:\n")
        print([features[x[0]] for x in whole_best_fts if x[1] >= 5])
        out_file.write(str([features[x[0]] for x in whole_best_fts if x[1] >= 5]) + "\n")
        print("\n\n------------- The end---------------\n")
        out_file.write("\n\n------------- The end---------------\n")
        out_file.flush()
        out_file.close()

        # draw a plot to illustrate the results
        whole_metrics = sorted(whole_metrics, key=operator.itemgetter(0))
        best_ones = sorted(best_ones, key=operator.attrgetter('gene_len'))

        plot_names = ["Mean fitness", "Best one's fitness", "Global best ones"]
        plt.figure(figsize=(9, 5), dpi=80)
        plt.subplot(1, 1, 1)
        plt.plot([x[0] for x in whole_metrics], [x[1] for x in whole_metrics], linewidth=2, label=plot_names[0],
                 color='gray')
        plt.plot([x[0] for x in whole_metrics], [x[2] for x in whole_metrics], linewidth=2, label=plot_names[1],
                 color='c')
        plt.scatter([x.gene_len for x in best_ones], [x.index for x in best_ones], marker='d', s=130,
                    label=plot_names[2], color='deepskyblue')
        ax = plt.gca()
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(11)
        plt.xlabel('Feature number', fontsize=12)
        plt.ylabel('ACC', fontsize=12)
        plt.title("GaRfe-" + str(self.dataset_name) + "-layer" + str(self.cur_layer), fontsize=16)
        plt.legend(loc='lower right')
        plt.savefig(output_path + self.plot_name)
        plt.close()

        ###############################


############################################

# =========================================
# Parameters setting part
# =========================================
dataset_name = "MLL"
cur_version = dataset_name + "_v2.0"

time_format = "-%m-%d-%H-%M-%S"
cur_time = time.strftime(time_format, time.localtime())
file_path = "D:\\codes\\python\\MachineLearning\\further_exp\\data\\MLL.txt"
output_path = "D:\\codes\\python\\MachineLearning\\further_exp\\populations_f\\" + dataset_name + "\\"
cls_size=3
cls_names=["ALL", "AML", "MLL"]

ft_index_name = "Name"
metrics_names = ["Acc", "Avc"]

Total_layer = 2
K = 5
global_best_size = 150
layer_best_size = 150

# all positions after MIC calculation
# the length of the feature range for the initialization of the first population
initial_ft_num = 1000

# RFE numbers in every layer
rfe_numbers = [4,2]
# feature numbers in every layer
feature_numbers_layer = [1000,80]
rfe_feature_number_layer = [70,40]

# =========================================
# load the data, get the data set (X, y), set feature range
# =========================================

df = pd.read_csv(file_path, sep="\t")
df = df.set_index(ft_index_name)
classes = df.columns
features = df.index
cls = [0] * cls_size
for i in range(cls_size):
    cls[i] = list(filter(lambda x: x.find(cls_names[i]) != -1, classes))

whole_sample = []
for i in range(cls_size):
    whole_sample = whole_sample + cls[i]

df = df.ix[:, whole_sample]

# X, y
X = np.array(df.values).T
y = []
for i in range(cls_size):
    y = y + [i] * len(cls[i])
y = np.array(y)


# # data preprocessing
# X = preprocessing.normalize(X)

if not os.path.exists(output_path):
    os.mkdir(output_path)


# MIC calculation
mine = MINE()
mic_scores = []
for i in range(len(X[0])):
    mine.compute_score(X[:, i], y)
    mic_scores.append(mine.mic())


top_fts_mic = sorted(list(zip(range(len(X[0])), mic_scores)), key=operator.itemgetter(1), reverse=True)
top_mic_pos = [x[0] for x in top_fts_mic[0:initial_ft_num]]

# =========================================
# Go down!
# =========================================

ft_rg_layer = [0] * Total_layer # feature range for every layer

best_ones_layer = [0] * Total_layer # best ones in each layer
global_best = []                   # best ones in all layers

for l in range(Total_layer):
    # feature range for each layer
    if(l==0):
        ft_rg_layer[l] = top_mic_pos  # the first layer
    else:
        best_ones_layer[l-1] = sorted(best_ones_layer[l-1], reverse=True)[0:layer_best_size]
        global_best += best_ones_layer[l-1]
        global_best = sorted(global_best, reverse=True)[0:global_best_size]
        ft_rg_layer[l] = []
        tmp = 0
        while(len(ft_rg_layer[l]) < feature_numbers_layer[l] and tmp<len(global_best)):
            ft_rg_layer[l] = list(set(ft_rg_layer[l] + global_best[tmp].ft_rg))
            tmp = tmp+1

    # create a layer
    for rfe_num in range(rfe_numbers[l]):
        rfe = RFE(dataset=dataset_name, initial_ft_rg=ft_rg_layer[l],
                  cur_layer=l,rfe_num=rfe_num,file_path=file_path,max_gene_len=min(rfe_feature_number_layer[l],len(ft_rg_layer[l])-1),
                 cls_size=cls_size,cls_names=cls_names,index_name=ft_index_name,
                  pop_best_size=5,pop_size=100,best_size=25)
        rfe.down()


# output
best_ones_layer[l] = sorted(best_ones_layer[l - 1], reverse=True)[0:layer_best_size]
global_best += best_ones_layer[l]
global_best = sorted(global_best, reverse=True)[0:global_best_size]

for l in range(Total_layer):
    file_name = dataset_name + "-Layer-" + str(l) + time.strftime(time_format, time.localtime()) + ".txt"
    out_file = open(output_path + file_name, mode='w')
    print("\n/***Generated from " + "layer-" + str(l) + "***/\n")
    out_file.write("/***Generated from " + "layer-" + str(l) + "***/\n")
    print("\n\n----------- Overall best individuals -----------\n\n")
    out_file.write("\n\n----------- Overall best individuals -----------\n\n")
    for i in range(len(best_ones_layer[l])):
        print(best_ones_layer[l][i].shorter_str())
        out_file.write(best_ones_layer[l][i].shorter_str() + "\n")
    print("\n\n------------- The end---------------\n")
    out_file.write("\n\n------------- The end---------------\n")
    out_file.flush()
    out_file.close()

file_name = dataset_name + "-AllLayers-" + time.strftime(time_format, time.localtime()) + ".txt"
out_file = open(output_path + file_name, mode='w')
print("\n/***Generated from " + " All Layers " + "***/\n")
out_file.write("/***Generated from " + "layer-" + str(l) + "***/\n")
print("\n\n----------- Overall best individuals -----------\n\n")
out_file.write("\n\n----------- Overall best individuals -----------\n\n")
for i in range(len(global_best)):
    print(global_best[i].shorter_str())
    out_file.write(global_best[i].shorter_str() + "\n")
print("\n\n------------- The end---------------\n")
out_file.write("\n\n------------- The end---------------\n")
out_file.flush()
out_file.close()
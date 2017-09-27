# -*- coding: utf-8 -*-
"""
This script is for dealing with the dataset SRBCT.

1. The Individual class is the individual in genetic algorithm (GA).
2. The Population class and its subclass APop implements GA,
with two functions of calc_fitness and calc_metrics helping calculate the fitness and classification metrcis of each GA individual.
3. The RFE class implements GaRFE process.
4. The left codes are the main part of MGRFE.
"""

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

# import two classes of Individual and Population from local path
sys.path.append("D:\\codes\\python\\MGRFE\\Dataset Two\\code")
from individual import Individual
from population import Population

# the function for calculating the 2 metrics Acc and Avc from the confusion matrix
def calc_metrics(con_matrix):
    length = len(con_matrix[0])
    T = [0] * length
    NUM = [0] * length
    for i in range(length):
        T[i] = con_matrix[i][i]
        NUM[i] = sum(con_matrix[i])
        # avoid the error of division by zero
        if(NUM[i]==0):
            T[i] = NUM[i] = 1
    Acc = sum(T) / sum(NUM)
    Avc = 0.0
    for i in range(length):
        Avc = Avc + T[i] / NUM[i]
    Avc /= length
    return [Acc, Avc]


# fitness calculation part of GA
# parameter search not used, pure K-fold
def calc_fitness(pop):          # the pop parameter is a list of gene combinations
    rsts = list()               # rsts will contain the generated individuals from all gene combinations
    if len(pop[0]) > 20:        # smaller k in k-fold CV for lower time cost when the length of gene combination is above 15
        k = K - 2
    else:
        k = K
    for gene_rg in pop:
        indiv = Individual()    # create a GA Individual for each gene combination in the pop parameter
        indiv.ft_rg = gene_rg
        indiv.gene_len = len(gene_rg)
        mt_clf_ta = np.array([0.0] * len(metrics_names))
        mt_clf_te = np.array([0.0] * len(metrics_names))
        X_cur = X[:, gene_rg]
        # k fold cross validation
        skf = StratifiedKFold(n_splits=k, random_state=random.randint(0, 100), shuffle=True)
        clf = []
        for train_index, test_index in skf.split(X_cur, y):
            clf = GaussianNB()
            X_train = X_cur[train_index, :]
            X_test = X_cur[test_index, :]
            y_train = y[train_index]
            y_test = y[test_index]
            clf.fit(X_train, y_train)
            predicts = clf.predict(X_test)
            result = confusion_matrix(y_test, predicts)
            mt_clf_ta += calc_metrics(result)

            predicts = clf.predict(X_train)
            result = confusion_matrix(y_train, predicts)
            mt_clf_te += calc_metrics(result)
        indiv.metrics_ta = list(mt_clf_ta / k)
        indiv.metrics_te = list(mt_clf_te / k)
        indiv.clf = clf
        indiv.acc = 0.4 * indiv.metrics_ta[0] + 0.6 * indiv.metrics_te[0]
        indiv.index = indiv.acc
        rsts.append(indiv)
    return rsts



# =========================================================
# the APop class extends the Population class to perform GA
# =========================================================
class APop(Population):
    def __init__(self, gene_len, potential_gene_rg, pop_size=100,
                 generation_size=3, pc=0.5, pm=0.2, best_size=5):
        self.pop_size = pop_size                # the population size which is the number of total individuals in it
        self.gene_len = gene_len                # the gene length in this population
        self.potential_gene_rg = potential_gene_rg  # the candidate gene set for a GA run
        self.generation_size = generation_size  # varies according to the gene_len
        self.pc = pc                            # crossover probability used in cross function
        self.pm = pm                            # mutation probability used in mutation function
        self.best_size = best_size              # the size of best individuals
        self.mean_fit = []
        self.generation_cur = 0
        # generation_size is the GA iteration times ,
        #  which is currently set form 1 to 3, and larger iteration times for GA population with smaller gene length to do more efficient search
        if(self.gene_len>20):
            self.generation_size = 1
        elif(self.gene_len>10):
            self.generation_size = 2
        else:
            self.generation_size = 3

    # calculate the fitness of all individuals
    def calc_fitness(self, pop):
        return calc_fitness(pop)


# =============================================================
# the RFE (recursive feature elimination) class performs the core process of GaRFE,
# which is illustrated as an inverted triangle and functions as the search unit in each layer of MGRFE
# =============================================================
class RFE():
    # K fold CV
    K = 5
    # time format for output
    time_format = "-%m-%d-%H-%M-%S"
    def __init__(self, dataset, initial_ft_rg, cur_layer, rfe_num, file_path, output_path, max_gene_len,
                 pop_best_size=5, pop_size=85, best_size=30):

        self.dataset_name = dataset
        self.initial_ft_rg = initial_ft_rg
        self.cur_layer = cur_layer
        self.rfe_num = rfe_num

        self.cur_version = dataset_name + "_v2.0"
        self.cur_time = time.strftime(self.time_format, time.localtime())
        self.file_path = file_path
        self.output_path = output_path
        self.plot_name = dataset + "-layer-" + str(cur_layer) + "-rfe-" + str(rfe_num) + self.cur_time + ".pdf"
        self.stats_name = dataset + "-layer-" + str(cur_layer) + "-rfe-" + str(rfe_num) + self.cur_time + ".txt"
        self.max_gene_len = max_gene_len

        self.rfe_info = "# RFE-" + str(self.rfe_num) + "\n" + \
                        "# layer:" + str(self.cur_layer) + "\n" + \
                        "# GaussianNB() \n" + \
                        "# Shuffle processed in " + str(self.K) + "-Fold \n"

        # settings about the populations and evolution
        self.pop_size = pop_size            # population size for GA
        self.best_size = best_size          # RFE best size
        self.pop_best_size = pop_best_size  # population best size
        self.start_time=time.time()         # program time

    # randomly generate the candidate gene set for the first GA population
    def prepare(self):
        self.first_pop_gene = [random.sample(self.initial_ft_rg, self.max_gene_len) for i in range(self.pop_size)]

    # RFE and output
    def down(self):
        self.prepare()
        whole_pop = []
        best_ones = []
        whole_metrics = []
        whole_pop_size = 0
        cur_gene_len = self.max_gene_len

        # =========================================
        # evolution and feature elimination process
        while cur_gene_len >= 1:
            # every new_pop means a GA run
            new_pop = APop(gene_len=cur_gene_len, potential_gene_rg=self.initial_ft_rg,
                           pop_size=self.pop_size,
                           best_size=self.pop_best_size)

            if whole_pop_size == 0:
                new_pop.set_pop(self.first_pop_gene)
            else:
                new_pop.set_pop(whole_pop[whole_pop_size - 1].pop)
            new_pop.evolve()
            best_ones = np.append(best_ones, new_pop.best_cur)       # update the best individuals in this RFE
            best_ones.sort()
            best_ones = best_ones[:self.best_size]
            whole_metrics.append([cur_gene_len, new_pop.mean_fit[-1], new_pop.best_cur[0].index])
            whole_pop.append(new_pop)
            whole_pop_size = whole_pop_size + 1
            print("whole_pop_size:", whole_pop_size)
            print("cur_gene_len : ", cur_gene_len)
            print("best_cur : \n", new_pop.best_cur[0])
            print("mean_fit & best_fit", whole_metrics[-1])
            if cur_gene_len > 30:
                cur_gene_len = cur_gene_len - 3
            elif cur_gene_len > 20:
                cur_gene_len = cur_gene_len - 2
            else:
                cur_gene_len = cur_gene_len - 1

        # ==============================================
        # return best individuals to this layer of MGRFR
        global best_ones_layer
        if (type(best_ones_layer[self.cur_layer]) is int):
            best_ones_layer[self.cur_layer] = list(best_ones)
        else:
            best_ones_layer[self.cur_layer] += list(best_ones)

        # ================================================================
        # output and save the results of this GaRFE process to a text file
        whole_metrics = sorted(whole_metrics, key=operator.itemgetter(0))
        out_file = open(output_path + self.stats_name, mode='w')
        print("/***Generated from " + cur_version + "***/\n")
        out_file.write("/***Generated from " + cur_version + "***/\n")
        print("\n" + self.rfe_info)
        out_file.write("\n" + self.rfe_info + "\n")
        print("Time cost of this GaRFE: %s seconds\n"%((time.time() - self.start_time)))
        out_file.write("Time cost of this GaRFE: %s seconds\n" % ((time.time() - self.start_time)))
        print("\n------- Overall statistics ---------\n")
        out_file.write("\n------------ Overall statistics -------------\n")
        print("\ngene_len\tmean_fit\tbest_fit\n")
        out_file.write("\ngene_len\t\tmean_fit\t\tbest_fit\n")
        for x in whole_metrics:
            out_file.write(str(x[0]) + "\t\t" + str(round(x[1], 3)) + "\t\t" + str(round(x[2], 3)) + "\n")
            print(x)
        print("\n\n----------- Overall best individuals -----------\n\n")
        out_file.write("\n\n----------- Overall best individuals -----------\n\n")
        for best in best_ones:
            print(str(best))
            out_file.write(str(best) + "\n")
        print("\n\n----------- Overall best features -----------\n")
        out_file.write("\n\n----------- Overall best features -----------\n")

        best_ft_range = set(reduce(lambda x, y: x + y, [indiv.ft_rg for indiv in best_ones]))
        indiv_ft_rgs = [dict(zip(x.ft_rg, [1] * len(x.ft_rg))) for x in best_ones]
        whole_best_fts = list(zip(best_ft_range,
                                  [sum([x[f] for x in indiv_ft_rgs if f in x.keys()]) for f in
                                   best_ft_range]))
        whole_best_fts = sorted(whole_best_fts, key=operator.itemgetter(1),reverse=True)
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

        # ===========================================================
        # draw a plot to illustrate the results of this GeRFE process
        whole_metrics = sorted(whole_metrics, key=operator.itemgetter(0))
        best_ones = sorted(best_ones, key=operator.attrgetter('gene_len'))

        plot_names = ["Mean fitness", "Best one's fitness", "Global best ones"]
        plt.figure(figsize=(6, 4.7), dpi=120)
        plt.subplot(1, 1, 1)
        plt.plot([x[0] for x in whole_metrics], [x[1] for x in whole_metrics], linewidth=2, label=plot_names[0],
                 color='gray')
        plt.plot([x[0] for x in whole_metrics], [x[2] for x in whole_metrics], linewidth=2, label=plot_names[1],
                 color='c')
        plt.scatter([x.gene_len for x in best_ones], [x.index for x in best_ones], marker='d', s=100,
                    label=plot_names[2], color='deepskyblue')
        ax = plt.gca()
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(11)
        plt.xlabel('Feature number', fontsize=13)
        plt.ylabel('$Acc$', fontsize=13)
        plt.title("GaRFE-" + str(self.dataset_name) + "-layer" + str(self.cur_layer), fontsize=16)
        plt.legend(loc='lower right')
        plt.savefig(output_path + self.plot_name)
        plt.close()


############################################
#       Here is the MGRFE main part.
#
# =========================================
# parameters setting
# =========================================
dataset_name = "SRBCT"
cur_version = dataset_name + "_v2.0"
time_format = "-%m-%d-%H-%M-%S"
cur_time = time.strftime(time_format, time.localtime()) # record program run time
start_time = time.time()
file_path = "D:\\codes\\python\\MGRFE\\Dataset Two\\data\\SRBCT.txt"
output_path = "D:\\codes\\python\\MGRFE\\Dataset Two\\results\\"+dataset_name+"\\"+dataset_name+cur_time+"\\"
cls_size=4
cls_names=["EWS", "BL", "NB", "RMS"]
ft_index_name = "Image_Id."
metrics_names = ["Acc", "Avc"]

Total_layer = 2             # total iterative layer number of MGRFE
K = 5                       # K-fold CV in calculating the fitness and classification metrics of GA individuals
global_best_size = 150      # the number of optimal gene combinations or individuals stored in MGRFE
layer_best_size = 150       # the number of optimal gene combinations or individuals stored in each layer of MGRFE
# the size of candidate gene set after MIC calculation,
# the length of the feature range for the initialization of the first GA population
initial_ft_num = 700
# GaRFE numbers in every layer of MGRFE
rfe_numbers = [2,2,1]
# feature numbers in every layer, the size of candidate gene set
feature_numbers_layer = [700,100,40]
rfe_feature_number_layer = [70,35,15]

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


# data preprocessing
X = preprocessing.normalize(X)

# check availability of the output path
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

# preprocessing end
preprocess_time=time.time()

# =========================================
# Go down! (multilayer of GaRFE)
# =========================================
ft_rg_layer = [0] * Total_layer         # feature range for every layer
best_ones_layer = [0] * Total_layer     # list for best individuals in each layer
global_best = []                        # list for best individuals among all layers

for l in range(Total_layer):
    # feature range for each layer
    if(l==0):
        ft_rg_layer[l] = top_mic_pos  # the first layer
    else:
        best_ones_layer[l-1] = sorted(best_ones_layer[l-1])[0:layer_best_size]
        global_best += best_ones_layer[l-1]
        global_best = sorted(global_best)[0:global_best_size]
        ft_rg_layer[l] = []
        tmp = 0
        while(len(ft_rg_layer[l]) < feature_numbers_layer[l] and tmp<len(global_best)):
            ft_rg_layer[l] = list(set(ft_rg_layer[l] + global_best[tmp].ft_rg))
            tmp = tmp+1

    # create a new layer
    for rfe_num in range(rfe_numbers[l]):
        rfe = RFE(dataset=dataset_name, initial_ft_rg=ft_rg_layer[l],
                  cur_layer=l,rfe_num=rfe_num,file_path=file_path,output_path=output_path,max_gene_len=min(rfe_feature_number_layer[l],len(ft_rg_layer[l])-1),
                  pop_best_size=10,pop_size=100,best_size=25)
        rfe.down()
best_ones_layer[l] = sorted(best_ones_layer[l - 1])[0:layer_best_size]
global_best += best_ones_layer[l]
global_best = sorted(global_best)[0:global_best_size]

# =========================================
# Output  part
# =========================================
# output the best individuals in each layer
for l in range(Total_layer):
    file_name = dataset_name + "-Layer-" + str(l) + time.strftime(time_format, time.localtime()) + ".txt"
    out_file = open(output_path + file_name, mode='w')
    print("\n/***Generated from " + "layer-" + str(l) + "***/\n")
    out_file.write("/***Generated from " + "layer-" + str(l) + "***/\n")
    print("\n\n----------- Overall best individuals -----------\n\n")
    out_file.write("\n\n----------- Overall best individuals -----------\n\n")
    for best in best_ones_layer[l]:
        print(str(best))
        out_file.write(str(best) + "\n")
    print("\n\n------------- The end---------------\n")
    out_file.write("\n\n------------- The end---------------\n")
    out_file.flush()
    out_file.close()

# output the best individuals among all layers
file_name = dataset_name + "-AllLayers-" + time.strftime(time_format, time.localtime()) + ".txt"
out_file = open(output_path + file_name, mode='w')
print("\n/***Generated from " + " All Layers " + "***/\n")
out_file.write("\n/***Generated from " + " All Layers " + "***/\n")
print("Total time cost of this MGRFE: %s seconds\n" % ((time.time() - start_time)))
out_file.write("Total time cost of this MGRFE: %s seconds\n" % ((time.time() - start_time)))
print("Time cost in preprocessing: %s seconds\n" % ((preprocess_time - start_time)))
out_file.write("Time cost in preprocessing: %s seconds\n" % ((preprocess_time - start_time)))
print("\n\n----------- Overall best individuals -----------\n\n")
out_file.write("\n\n----------- Overall best individuals -----------\n\n")
for best in global_best:
    print(str(best))
    out_file.write(str(best)+"\n")
print("\n\n------------- The end---------------\n")
out_file.write("\n\n------------- The end---------------\n")
out_file.flush()
out_file.close()
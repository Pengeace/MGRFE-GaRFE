# -*- coding: utf-8 -*-
"""
This script of the Individual class implements the individual in the Genetic Algorithm (GA) used in MGRFE.
Each individual represents a gene combination and the related classification metrics and information of it.

Two points need pay attention to:
1. The property index_gap is used to adjust the rankings in sort for two individuals having similar fitness values 
(absolute difference value less than the index_gap) but one has smaller gene size than another.
2. The formula of the index property and elements in the metrics property can be defined and adjusted at where the Individual class is used.
"""

import math

# =========================================================
# the Individual class represents the individual in GA
# =========================================================
class Individual(object):
    index_gap = 0.005             # adjust rankings for individuals having similar fitness and gene combination size values

    def __init__(self,index=0):

        self.index = index        # the fitness value, used for selection
        self.acc = 0              # accuracy in classification
        self.gene_len = 1000      # the size of the gene combination
        self.ft_rg = []           # a list of integer gene numbers to represent a gene combination
        self.clf = []             # the classifier used in classification 
        self.metrics_ta = []      # a list of the classification metrics from the train data, the elements in it can be arbitrarily specified
        self.metrics_te = []      # a list of the classification metrics from the test data

    # offer the comparision manner between individuals so that sort can be processed
    def __gt__(self, other):
        if(self.gene_len == other.gene_len):
            if(self.index < other.index):
                return True
        else:
            if (self.index < other.index - self.index_gap) \
                or (math.fabs(self.index - other.index) < self.index_gap and self.gene_len > other.gene_len):
                return True
        return False

    # define how an individual can be convert to string for ease of output and print 
    def __str__(self):
        return "["+'\n'.join([
            "acc : " + str(self.acc),
            "index : " + str(self.index),
            "gene_len : " + str(self.gene_len),
            "ft_rg : " + str(self.ft_rg),
            "clf : " + str(self.clf),
            "metrics_train : " + str(self.metrics_ta),
            "metrics_test : " + str(self.metrics_te)])+"]"
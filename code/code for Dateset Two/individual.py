#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math


class Individual(object):
    index_gap = 0.009

    def __init__(self,index=0):

        self.index = index        # used for selection
        self.acc = 0              # accuracy
        self.gene_len = 1000
        self.ft_rg = []
        self.clf = []
        self.metrics_train = []
        self.metrics_test = []

    def __lt__(self, other):
        flag = False
        if(self.gene_len == other.gene_len):
            if(self.index < other.index):
                flag = True
        else:
            if (self.index < other.index - self.index_gap) \
                or (math.fabs(self.index - other.index) < self.index_gap and self.gene_len > other.gene_len):
                flag = True
        return flag


    def __str__(self):
        return "["+'\n'.join([
            "acc : " + str(self.acc),
            "index : " + str(self.index),
            "gene_len : " + str(self.gene_len),
            "ft_rg : " + str(self.ft_rg),
            "clf : " + str(self.clf),
            "metrics_train : " + str(self.metrics_ta),
            "metrics_test : " + str(self.metrics_te)])+"]"

    def shorter_str(self):
        mts_ta = [round(x,3) for x in self.metrics_ta]
        mts_te = [round(x,3) for x in self.metrics_te]
        return "[" + '\n'.join([
            "acc : " + str(round(self.acc, 3)),
            "index : " + str(round(self.index, 3)),
            "gene_len : " + str(self.gene_len),
            "ft_rg : " + str(self.ft_rg),
            "clf : " + str(self.clf),
            "metrics_train : " + str(mts_ta),
            "metrics_test : " + str(mts_te)]) + "]"
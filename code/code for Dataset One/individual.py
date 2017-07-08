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
        self.metrics = []

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
            "metrics : " + str(self.metrics)])+"]"

    def shorter_str(self):
        mts = [round(x,3) for x in self.metrics]
        return "[" + '\n'.join([
            "acc : " + str(round(self.acc, 3)),
            "index : " + str(round(self.index, 3)),
            "gene_len : " + str(self.gene_len),
            "ft_rg : " + str(self.ft_rg),
            "clf : " + str(self.clf),
            "metrics : " + str(mts)]) + "]"
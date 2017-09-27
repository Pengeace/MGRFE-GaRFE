# -*- coding: utf-8 -*-
"""
This script of the Population class implements the Genetic Algorithm in MGRFE.

Three points need pay attention to:
1. Variable length integer encoding method for GA chromosome is used, so each individual represents a variable length gene combination.
2. The selection process uses linear ranking selection rather than the roulette wheel selection for the former works much better than the latter in MGRFE according to experiments.
3. Remember to avoid duplicated genes when generating the children individuals in the operators of mutation and crossover.
"""

import random
import numpy as np
# =========================================================
# the Population class implements the Genetic Algorithm (GA)
# =========================================================
class Population(object):
    def __init__(self, gene_len, potential_gene_rg, pop_size=90,
                 generation_size=10, pc=0.6, pm=0.2, best_size=5):
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
        if(self.gene_len>50):
            self.generation_size = 1
        elif(self.gene_len>25):
            self.generation_size = 2
        else:
            self.generation_size = 3


    # randomly generate the initial individuals in the population
    def set_pop(self, pre_pop):
        if len(pre_pop) != 0:
            self.pop = list(pre_pop)
            self.pop = [random.sample(pre_pop[i], self.gene_len) for i in range(self.pop_size)]
        else:
            self.pop = [random.sample(self.potential_gene_rg, self.gene_len) for i in range(self.pop_size)]

        self.best_cur = []  # best individuals
        self.is_first_ger = 1


    # calculate the fitness of all individuals, implemented by subclass
    def calc_fitness(pop):
        pass


    # selection operator, just linear ranking selection
    # which is simpler than roulette wheel selection but more efficient for the task here
    def select_linear(self):
        # fitness calculation
        if self.is_first_ger:
            results = np.array(self.calc_fitness(self.subpop+self.pop))
            self.is_first_ger = 0
        else:
            results = np.append(self.pop_indivs,self.calc_fitness(self.subpop))

        # linear selection and update records
        fitness = np.array([x.index for x in results])
        sum_fit = sum(fitness)
        cur_pop_size = (len(self.pop) + len(self.subpop))
        self.mean_fit.append(sum_fit / cur_pop_size)
        if len(self.best_cur) != 0:
            results = np.append(results,self.best_cur)
        results.sort()

        self.pop_indivs = np.array(results[0:self.pop_size])
        self.pop = [ x.ft_rg for x in self.pop_indivs ]
        self.best_cur = results[0:self.best_size]


    # crossover operator, just single point crossover
    def cross(self):
        self.subpop = []        # the list of children individuals
        for i in range(self.pop_size):
            if random.random() < self.pc:
                another = random.randint(0, self.pop_size - 1)
                pos = random.randint(0, self.gene_len)      # crossover position
                # attention , remember to avoid the duplication of genes which will lead to the decrease of indeed existed genes
                tmp1 = set(random.sample(self.pop[i], pos) + random.sample(self.pop[another], self.gene_len - pos))
                tmp2 = set(random.sample(self.pop[another], pos) + random.sample(self.pop[i], self.gene_len - pos))
                while len(tmp1) < self.gene_len:
                    tmp1.add (self.pop[i][random.randint(0, self.gene_len - 1)])
                while len(tmp2) < self.gene_len:
                    tmp2.add (self.pop[another][random.randint(0, self.gene_len - 1)])
                self.subpop += [list(tmp1)]
                self.subpop += [list(tmp2)]


    # mutate operator, just replace some gene features with others
    def mutate(self):
        positions = int((0.05 * self.gene_len)) + 2 # the position number to do mutation for a selected individual
        for i in range(self.pop_size):
            if random.random() < self.pm:
                temp = self.pop[i]
                for j in range(positions):
                    pos = random.randint(0, self.gene_len - 1)
                    r = self.pop[random.randint(0, self.pop_size - 1)][random.randint(0, self.gene_len - 1)]
                    # attention, remember to avoid duplicated genes
                    if r not in temp:
                        temp[pos] = r
                self.subpop += [list(temp)]

    # evolution, every loop means an GA iteration
    def evolve(self):
        print('\n------ Current gene length = ' + str(self.gene_len) + '-----------')
        while self.generation_cur < self.generation_size:
            self.cross()
            print("cross end")
            self.mutate()
            print("mutate end")
            self.select_linear()
            print("select end")
            self.generation_cur += 1
            print('Current generation:' + str(self.generation_cur))
            print('Current best one:\n' + str(self.best_cur[0]))
        self.pop_indivs = []
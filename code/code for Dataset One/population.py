# -*- coding: utf-8 -*-

import random
import numpy as np

# population performs the execution of Genetic Algorithm
class Population(object):
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
            self.generation_size = 3
        elif(self.gene_len>25):
            self.generation_size = 3
        else:
            self.generation_size = 4

    def set_pop(self, pre_pop):
        if len(pre_pop) != 0:
            self.pop = list(pre_pop)
            self.pop = [random.sample(pre_pop[i], self.gene_len) for i in range(self.pop_size)]
        else:
            self.pop = [random.sample(self.potential_gene_rg, self.gene_len) for i in range(self.pop_size)]

        self.best_cur = []  # best individuals

    # realized by subclass
    def calc_fitness(pop):
        pass

    def select_linear(self):

        results = self.calc_fitness(self.subpop+self.pop)


        fitness = [x.index for x in results]
        sum_fit = sum(fitness)
        cur_pop_size = (len(self.pop) + len(self.subpop))
        self.mean_fit.append(sum_fit / cur_pop_size)

        if len(self.best_cur) != 0:
            results = results + self.best_cur
        results.sort(reverse=True)
        new_pop = [ x.ft_rg for x in results[0:self.pop_size] ]


        self.best_cur = results[0:self.best_size]

        self.pop = new_pop


    def cross(self):
        self.subpop = []
        for i in range(self.pop_size):
            if random.random() < self.pc:
                another = random.randint(0, self.pop_size - 1)
                pos = random.randint(0, self.gene_len)

                tmp1 = set(random.sample(self.pop[i], pos) + random.sample(self.pop[another], self.gene_len - pos))
                tmp2 = set(random.sample(self.pop[another], pos) + random.sample(self.pop[i], self.gene_len - pos))

                while len(tmp1) < self.gene_len:
                    tmp1.add(self.pop[i][random.randint(0, self.gene_len - 1)])

                while len(tmp2) < self.gene_len:
                    tmp2.add(self.pop[another][random.randint(0, self.gene_len - 1)])

                self.subpop += [list(tmp1)]
                self.subpop += [list(tmp2)]

    # replace a feature with another feature in other individuals
    def mutate(self):
        for i in range(self.pop_size):
            if random.random() < self.pm:
                pos = random.randint(0, self.gene_len - 1)
                r = self.pop[random.randint(0, self.pop_size - 1)][random.randint(0, self.gene_len - 1)]
                times = 5
                while r in self.pop[i] and times > 0:
                    times -= 1
                    r = self.pop[random.randint(0, self.pop_size - 1)][random.randint(0, self.gene_len - 1)]
                if times>0:
                    temp = self.pop[i]
                    temp[pos] = r
                    self.subpop +=[list(set(temp))]


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
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 03:46:05 2014

@author: mit
"""

"""
This script implement gibbs sampling with information in the form of alpha parameter
used for the dirichlet prior of the cathegorical distribution. Therefore, we can 
derive the full conditional probability distribution and sample from there.
"""

from numpy.random import randint
from sampling import sample
import math
import matplotlib.pyplot as plt

# Read data from file
filename = 'a.seq'
f = open(filename, 'r')

K = int(f.readline())
N = int(f.readline())
w = int(f.readline())
alphabet = list(f.readline()[:-1])
alpha_b = map(float, f.readline()[:-1].split(',') )
alpha_w = map(float, f.readline()[:-1].split(',') )

sequences = []
for i in xrange(K):
    seq = f.readline()[:-1].split(',')
    sequences += [seq]

position = map(int, f.readline()[:-1].split(',') )
f.close()

def full_conditional(sequences, pos, seqth):
    
    # Count the background
    q = {x:[1]*w for x in alphabet}
    p = {x: 1 for x in alphabet}
    
    for i in xrange(len(sequences)):
        if i == seqth:
            continue
        for j in xrange(len(sequences[i])):
            if j < pos[i] or j > pos[i]+w:
                c = sequences[i][j]
                p[c] = p[c]+1
    
    for i in xrange(len(sequences)):
        if i== seqth:
            continue
        for j in xrange(w):
            start_pos = pos[i]
            c = sequences[i][start_pos+j]
            q[c][j] = q[c][j]+1
    
    A = [0]*(N-w)
    
    for i in xrange(N-w):
        pback = math.gamma(sum(alpha_b)) / math.gamma(K*(N-w) + sum(alpha_b))
        
        extra = {'A':0, 'C':0, 'T':0, 'G':0}
        for j in xrange(N):
            if j < i or j > i+w:
                c = sequences[seqth][j]
                extra[c] += 1
        for j in xrange(len(alphabet)):
            a = alphabet[j]
            pback *= math.gamma(p[a]+extra[a] + alpha_b[j]) / math.gamma(alpha_b[j])
            
        pcol = 1
        
        for j in xrange(w):
            pm = math.gamma(sum(alpha_w)) / math.gamma(K + sum(alpha_w))
            
            for k in xrange(len(alphabet)):
                a = alphabet[k]
                extra = 0
                if sequences[seqth][i+j] == a:
                    extra = 1
                pm *= math.gamma(q[a][j]+extra) / math.gamma(alpha_w[k])
            
            pcol = pcol*pm
        
        A[i] = pback * pcol
    
    return A
        

# First, initialize the start position randomly
pos = [randint(0, N-w+1) for x in xrange(K)]
orig_pos = pos[:]

MAX_ITER = 100
p = [0]*(N-w)
b = [0]*MAX_ITER
for it in xrange(MAX_ITER):
    
    for i in xrange(K):
        p = full_conditional(sequences, pos, i)
        
        total = sum(p)
        p = map(lambda x: x/total, p)
        # Sample new position
        #print 'p', p
        pos[i] = sample(range(N-w), p)
    b[it] = max(p)


# Happy printing
print 'start pos', orig_pos
print 'last pos', pos
print 'true pos', position
plt.plot(b)
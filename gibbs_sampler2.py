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
from sequence_generator import sample
import math

# Related information
sequences = []
K = len(sequences)
N = 15
w = 6
alphabet = ['A', 'C', 'T', 'G']
M = len(alphabet)
alpha_b = [1,1,1,1]
alpha_w = [10,2,8,3]

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
            a = alphabet(j)
            pback *= math.gamma(p[a]+extra[a] + alpha_b[j]) / math.gamma(alpha_b[j])
            
        pcol = 1
        
        for j in xrange(w):
            pm = math.gamma(sum(alpha_w)) / math.gamma(K + sum(alpha_w))
            
            for k in xrange(len(alphabet)):
                a = alphabet(k)
                extra = 0
                if sequences[seqth][i+j] == a:
                    extra = 1
                pm *= math.gamma(q[a][j]+extra) / math.gamma(alpha_w[k])
            
            pcol = pcol*pm
        
        A[i] = pback * pcol
    
    return A
        

# First, initialize the start position randomly
pos = [randint(0, N-w+1) for x in xrange(K)]

MAX_ITER = 10
for it in xrange(MAX_ITER):
    
    for i in xrange(K):
        p = [0]*(N-w)
        p = full_conditional(sequences, pos, i)
        
        # Sample new position
        pos[i] = sample(range(N-w), p)


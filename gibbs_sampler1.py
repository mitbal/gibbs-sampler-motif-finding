# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:17:08 2014

@author: mit
"""

"""
This script implements gibbs sampling to find motif in DNA sequence. The sampler
works with no information about probability distribution of character in the sequence, 
"""

from numpy.random import randint
from sequence_generator import sample

# The information should be provided in the beginning
sequences = []
K = len(sequences)
N = 15
w = 6
alphabet = ['A', 'C', 'T', 'G']
M = len(alphabet)


def compute_model(sequences, pos, alphabet, w):
    """
    This method compute the probability model of background and word based on data in 
    the sequences.
    """
    q = {x:[1]*w for x in alphabet}
    p = {x: 1 for x in alphabet}
    
    # Count the number of character occurrence in the particular position of word
    for i in xrange(len(sequences)):
        start_pos = pos[i]        
        for j in xrange(w):
            c = sequences[i][start_pos+j]
            q[c][j] += 1
    # Normalize the count
    for c in alphabet:
        for j in xrange(w):
            q[c][j] = q[c][j] / float( K+len(alphabet) )
    
    # Count the number of character occurrence in background position
    # which mean everywhere except in the word position
    for i in xrange(len(sequences)):
        for j in xrange(len(sequences[i])):
            if j < pos[i] or j > pos[i]+w:
                c = sequences[i][j]
                p[c] += 1
    # Normalize the count
    total = sum(p.values())
    for c in alphabet:
        p[c] = p[c] / float(total)
    
    return q, p
            

# First, initialize the state (in this case position) randomly
pos = [randint(0,N-w+1)]

# Loop until converge (the burn-in phase)
MAX_ITER = 10
for it in xrange(MAX_ITER):
    # We pick the sequence, well, in sequence starting from index 0
    for i in xrange(K):
        # We sample the next position of magic word in this sequence
        # Therefore, we exclude this sequence from model calculation
        seq_minus = sequences[:]; del seq_minus[i]
        pos_minus = pos[:]; del pos_minus[i]
        q, p = compute_model(seq_minus, pos_minus)
        
        # We try for every possible position of magic word in sequence i and
        # calculate the probability of it being as background or magic word
        # The probability for magic word is calculated by multiplying the probability
        # for each character in each position
        qx = [1]*(N-w)
        px = [1]*(N-w)
        for j in xrange(N-w):
            for k in xrange(w):
                c = sequences[i][j+k]
                qx[j] = qx[j] * q[c][k]
                px[j] = px[j] * p[c]
        
        # Compute the ratio between word and background, the pythonic way
        Aj = [x/y for (x,y) in zip(qx, px)]
        norm_c = sum(Aj)
        Aj = map(lambda x: x/norm_c, Aj)
        
        # Sampling new position with regards to probability distribution Aj
        pos[i] = sample(range(N-w), Aj)

# Happy printing
print 'new pos', pos

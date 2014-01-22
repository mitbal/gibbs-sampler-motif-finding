# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:52:08 2014

@author: mit
"""

"""
# This script generate sequence with hidden word with certain generative model
# Description taken from the assignment:

The following generative model generates K sequences of length N: s1,...,sk where
si = si1,...,siN. All sequences are over the alphabet [M]. Each of these sequences
has a magic word of length w hidden in it and the rest of the sequence is called 
background.

First, for each i, a start position is sampled uniformly from [N-w+1]. The the 
jth positions in the magic word are sampled from qj(x), which is Cat(x|thetaj) 
where thetaj has has a Dir(thetaj|alpha) prior. All other positions in the 
sequences are sampled from the background distribution q(x), which is 
Cat(x|theta) where theta has a Dir(theta|alphaprime) prior.
"""

import numpy as np
from numpy.random import rand
from numpy.random import dirichlet
from numpy.random import randint


### The properties for the generated sequence

K = 10   # The number of sequence
N = 15  # The length for each sequence
w = 6   # The length of the magic word

alphabet = ['A', 'C', 'T', 'G']         # The alphabet used in the sequence
M = len(alphabet)                       # The number of alphabet used

alpha_b = [1]*M             # The alpha parameter for dirichlet prior of background letter
alpha_w = [10,2,8,3]         # The alpha parameter for dirichlet prior of hidden word


### Start the generator part

def sample(alphabet, dist):
    """ This method produce a new discrete sample list by alphabet with probability
    distribution given by dist.
    The length of alphabet and dist must be same."""
    sampl = None
    cum_dist = np.cumsum(dist)
    r = rand()
    for i in xrange(len(dist)):
        if r < cum_dist[i]:
            sampl = alphabet[i]
            break
    
    return sampl

# First, generate the starting position of the magic word for all sequences uniformly
position = [0]*K
for i in xrange(K):
    position[i] = randint(0, N-w+1)
    
# Generate the background letters for all sequences
cat_b = dirichlet(alpha_b)
sequences = []
for i in xrange(K):
    seq = []
    for j in xrange(N):
        seq += [sample(alphabet, cat_b)]
    sequences += [seq]

# Generate the magic words
theta = dirichlet(alpha_w, w)
for i in xrange(K):
    start_pos = position[i]
    for j in xrange(w):
        sequences[i][start_pos+j] = sample(alphabet, theta[j])


# Happy printing
print 'theta', theta
print 'word position', position
print 'sequences'
for seq in sequences:
    print seq
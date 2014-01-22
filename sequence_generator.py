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

from numpy.random import dirichlet
from numpy.random import randint
from sampling import sample

WRITE_TO_FILE = True

### The properties for the generated sequence

K = 10   # The number of sequence
N = 15  # The length for each sequence
w = 6   # The length of the magic word

alphabet = ['A', 'C', 'T', 'G']         # The alphabet used in the sequence
M = len(alphabet)                       # The number of alphabet used

alpha_b = [1]*M             # The alpha parameter for dirichlet prior of background letter
alpha_w = [10,2,8,3]         # The alpha parameter for dirichlet prior of hidden word


### Start the generator part

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


if WRITE_TO_FILE:
    ### Store the generated sequence to file to be read later
    filename = 'a.seq'
    
    f = open(filename, 'w')
    
    f.write(str(K)+'\n')
    f.write(str(N)+'\n')
    f.write(str(w)+'\n')
    f.write(''.join(alphabet)+'\n')
    
    f.write(','.join(map(str, alpha_b)) +'\n')
    f.write(','.join(map(str, alpha_w)) +'\n')
    
    for s in sequences:
        f.write(','.join(map(str, s)) +'\n')
    
    f.write(','.join(map(str, position)) +'\n')
    
    f.close()
    
else:
    print 'K', K
    print 'N', N
    print 'w', w
    print 'alphabet', alphabet
    print 'alpha_b', alpha_b
    print 'alpha_w', alpha_w
    
    for s in sequences:
        print s
    print 'position', position

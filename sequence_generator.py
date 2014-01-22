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

K = 3   # The number of sequence
N = 12  # The length for each sequence
w = 5   # The length of the magic word

alphabet = ['A', 'C', 'T', 'G']         # The alphabet used in the sequence
M = len(alphabet)                       # The number of alphabet used

alpha_b = [1]*M             # The alpha parameter for dirichlet prior of background letter
alpha_w = [2,5,3,4]         # The alpha parameter for dirichlet prior of hidden word




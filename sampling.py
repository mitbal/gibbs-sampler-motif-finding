# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:56:25 2014

@author: mit
"""

import numpy as np
from numpy.random import rand

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
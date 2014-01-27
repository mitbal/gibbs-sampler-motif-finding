# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 03:36:52 2014

@author: mit
"""

"""One example on how to use gibbs sampling to find hidden pattern or motif
inside a sequence given a whole number of sequences as training data"""

import gibbs


# Test the method sampling from module gibbs
seqs = ['thequickdog', 'browndog', 'dogwood']
k = 3

new_pos = gibbs.sampling(seqs, k)

words = [seqs[i][new_pos[i]:new_pos[i]+k] for i in xrange(len(seqs))]
print words

# In order to enhance the accuracy, run the sampling a couple of times (multiple chains)
result = {}
for i in xrange(20):
    new_pos = gibbs.sampling(seqs, k)
    #print new_pos
    tnp = tuple(new_pos)    
    if tnp in result:
        result[tnp] += 1
    else:
        result[tnp] = 1

max_vote = 0
max_pos = None
for key in result:
    #print key, result[key]
    if result[key] > max_vote:
        max_pos = list(key)
        max_vote = result[key]

words = [seqs[i][max_pos[i]:max_pos[i]+k] for i in xrange(len(seqs))]
print words
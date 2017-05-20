#!/usr/bin/env python
"""
An object to pad and generate batches of data for the VRAE model

__author__ = "Valentin Lievin, DTU, Denmark"
__copyright__ = "Copyright 2017, Valentin Lievin"
__credits__ = ["Valentin Lievin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Valentin Lievin"
__email__ = "valentin.lievin@gmail.com"
__status__ = "Development"
"""

import numpy as np
from operator import itemgetter 

class Generator:
    def __init__(self, x, y, batch_size):
        """
        initialize the class with inputs 'x' and labels 'y'
        Args:
            x (list of Objects): list of inputs
            y (list of Objects): list of labels
            batch_size (Natural Interger): number of elements to be produced at each iteration
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.index = list(xrange(len(x)))
        self.n_steps = len(x) // batch_size
        self.step = 0
        assert len(self.x) == len(self.y)
        assert self.n_steps > 0
        
    def iterations_per_epoch(self):
        """
        Return the number of iteration per epoch
        """
        return len(self.x) // self.batch_size
        
    def shuffle(self):
        """
        shuffle index and re-initialize step
        """
        np.random.shuffle(self.index)
        self.step = 0
    
    def epochCompleted(self):
        """
        Says if a whole epoch has been processed
        Returns:
            True if completed else False
        """
        return (self.step+1) * self.batch_size > len(self.x)
    
    def raw_batch(self):
        """
        return the next batch without padding
        Returns:
            a tuple batch_xs, batch_ys
            batch_xs: a list of batch_len objects
            batch_ys: the list of corresponding labels
        """
        assert (not self.epochCompleted())
        indexes = self.index[self.step * self.batch_size : (self.step+1) * self.batch_size]
        self.step += 1
        return itemgetter(*indexes)(self.x), itemgetter(*indexes)(self.y)
    
    def pad(self, l, n):
        """
        padd a sequence l to a sequence of length n. We assume that the symbol representing padding is 0.
        Args:
            l: sequence to be padded
            n : target number of elmements
        Returns:
            the input sequence padded with 0s
        """
        return l[:n] + [0]*(n-len(l))

    def next_batch(self):
        """
        return a padded batch. We assume that the symbol representing padding is 0
        Returns:
            a tuple batch_xs, batch_ys; batch_weights, max_length
                batch_xs: a padded list of batch_len objects
                batch_ys: the list of corresponding labels
                batch_lengths: the list of sequence lengths
                batch_weights: a list of weights corresponding to 0 if it's a padded element, 0 otherwise
                max_length: the maximum length in the current batch
                batch_ys: labels
        """
        batch_xs, batch_ys = self.raw_batch()
        batch_lengths = [len(x) for x in batch_xs]
        max_length = max(batch_lengths)
        padded_batch_xs = [ self.pad(d, max_length) for d in batch_xs ]
        batch_weights = [ [ 1 if dd>0 else 0 for dd in d] for d in padded_batch_xs]
        batch_ys
        return padded_batch_xs, batch_ys, batch_lengths, batch_weights, max_length
        
        
    
    
    
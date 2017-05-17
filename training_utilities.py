#!/usr/bin/env python
"""
function used to control training: deterministic warm-up and learning rate decay

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
import scipy.interpolate as si
from scipy import interpolate


def BetaGenerator(epoches, beta_decay_period, beta_decay_offset):
    """
    Return a generator which gives the value of Beta for a given epoche for the Deterministic Warmup
    The deterministic warm up trick has been described in this paper: 
    
    Args:
        epoches: number of epoches
        beta_decay_period: duration of the variation of beta (beta is increased from 0 to 1 during this period)
        beta_decay_offset: duration after which beta begins to be increased
    
    Return: a function which return Beta value with an epoch value as input
    """
    points = [[0,0], [0, beta_decay_offset],[0, beta_decay_offset + 0.33 * beta_decay_period], [1, beta_decay_offset + 0.66*beta_decay_period],[1, beta_decay_offset + beta_decay_period], [1, epoches] ];
    points = np.array(points)
    x = points[:,0]
    y = points[:,1]
    t = range(len(points))
    ipl_t = np.linspace(0.0, len(points) - 1, 100)
    x_tup = si.splrep(t, x, k=3)
    y_tup = si.splrep(t, y, k=3)
    x_list = list(x_tup)
    xl = x.tolist()
    x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]
    y_list = list(y_tup)
    yl = y.tolist()
    y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]
    x_i = si.splev(ipl_t, x_list)
    y_i = si.splev(ipl_t, y_list)
    return interpolate.interp1d(y_i, x_i)


class LearningRateControler:
    def __init__(self, initial_value, change_rate, decay_factor, minimum_value= 1e-6):
        """
        initialize the class
        """
        self.learning_rate = initial_value
        self.last_losses = []
        self.change_rate = change_rate
        self.measure = change_rate // 2
        self.decay_factor = decay_factor
        self.runs_since_last_learning_rate_change = 0
        self.minimum = minimum_value
        
    def update(self,loss):
        self.last_losses.append(loss)
        self.runs_since_last_learning_rate_change += 1
        if self.runs_since_last_learning_rate_change > self.change_rate:
            if np.mean(self.last_losses[- 2*self.measure : - self.measure ]) <= np.mean(self.last_losses[- self.measure : ]):
                self.learning_rate *= self.decay_factor
                self.runs_since_last_learning_rate_change = 0
                if self.learning_rate < self.minimum:
                    self.learning_rate = self.minimum 
    def reset(self):
        self.runs_since_last_learning_rate_change = 0
        
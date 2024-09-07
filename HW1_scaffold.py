import random
random.seed(112358)

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# TensorFlow and tf.keras
import tensorflow as tf

%matplotlib inline
# 1.1
# your code here 

# 1.1.2
# your code here

# 1.2
# build your NN 
# your code here


# compile it and run it
# your code here 


# plot train and val acc as  a function of epochs
# your code here


# primer to print: 
# print("NN_model_train_auc:", roc_auc_score(y_train, y_hat))
# your code here 


# your code here

# 1.3
# Fit the logistic regression model
# your code here

# 1.4
# 1.4.1
# your code here

# 1.4.1要写解释
# 1.4.2
# your code here

# 1.4.3
# your code here

# 1.4.4
# your code here

# 1.4.4要写解释

# 1.5
def progressbar(n_step, n_total):
    """Prints self-updating progress bar to stdout to track for-loop progress
    
    There are entire 3rd-party libraries dedicated to custom progress-bars.
    A simple function like this is often more than enough to get the job done.
    
    :param n_total: total number of expected for-loop iterations
    :type n_total: int
    :param n_step: current iteration number, starting at 0
    :type n_step: int

    .. example::
    
        for i in range(n_iterations):
            progressbar(i, n_iterations)
            
    .. source:
    
        This function is a simplified version of code found here:
        https://stackoverflow.com/questions/3160699/python-progress-bar/15860757#15860757
    """
    n_step = n_step + 1
    barlen = 50
    progress = n_step / n_total
    block = int(round(barlen * progress))
    status = ""
    if n_step == n_total:
        status = "Done...\r\n\n"
    text = "\r [{0}] {1}/{2} {3}".format(
        "=" * block + "-" * (barlen - block),
        n_step,
        n_total,
        status,
    )
    sys.stdout.write(text)
    sys.stdout.flush()

%%time
# Bootstrap and train your networks and get predictions on fixed X test
# your code here


# generate your plot
# your code here

# 1.5此处要写解释
# 1.6
# your code here

# 1.6此处要写解释

# 2.1
# your code here 

# 2.2
# your code here

# 2.2此处要写解释

# 2.3.1
# your code here

# 2.3.2
# your code here

# 2.3.3
# your code here

# 2.3.4
# your code here

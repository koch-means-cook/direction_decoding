#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:30:09 2019

@author: koch
"""

# Import libraries

# Misc
from pathlib import Path
from os import listdir
from os import system
from os import remove
from copy import deepcopy
from sys import argv

# Math
import numpy as np
import scipy.io
import scipy.stats
from random import sample
from collections import Counter

# Image processing
import nibabel
from nilearn import signal
from nilearn import image
import pandas as pd

# Machine learning
import sklearn.model_selection
import sklearn.svm

# Plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec


''' Decoding for subject depending on input'''


# Set up paths
# BIDS directory (containing betas and raw data)
bids_dir = str(Path.home()) + '/direction_decoding_BIDS'
# Path of repository
repo_dir = str(Path.home() + '/direction_decoding')
results_dir = (repo_dir + '/decoding/backwards_decoding/train_beta_test_raw/results')


# Make list of all files in 'derivatives'
derivatives_list = np.array(listdir(bids_dir + '/derivatives'))
# Get indices of all files containing string 'sub-'
sub_index = [i for i, v in enumerate(derivatives_list) if 'sub-' in v]
# Make new list only containing subjects and sort it
sub_list = derivatives_list[sub_index]
sub_list = sorted(sub_list)
index_old = [i for i, v in enumerate(sub_list) if 'old' in v]
index_young = [i for i, v in enumerate(sub_list) if 'young' in v]

#sub_list = sub_list[0:5]

# Give message to user
print('Collecting files...')

#---
# Classification accuracies
# Get path for each accuracy file in directory
path_accuracies = [results_dir + '/' + x + '_clf_accuracy.tsv'
                   for x in sub_list]
# Load accuracies for each subject into list of data frames
files_accuracies = [pd.read_csv(path_accuracies[x],
                                sep='\t') for x in np.arange(len(sub_list))]
# Concatenate list of data frames to one DF
files_accuracies = pd.concat(files_accuracies)

#---
# Classification accuracies permutation
path_accuracies_permut = [results_dir + '/' + x + '_clf_accuracy_permut.tsv'
                          for x in sub_list]
files_accuracies_permut = [pd.read_csv(path_accuracies_permut[x],
                                 sep='\t') for x in np.arange(len(sub_list))]
files_accuracies_permut = pd.concat(files_accuracies_permut)

#---
# Predictions
path_predictions = [results_dir + '/' + x + '_clf_prediction.tsv'
                    for x in sub_list]
files_predictions = [pd.read_csv(path_predictions[x],
                                 sep='\t') for x in np.arange(len(sub_list))]
files_predictions = pd.concat(files_predictions)

#---
# Predictions permutation
path_predictions_permut = [results_dir + '/' + x + '_clf_prediction_permut.tsv'
                           for x in sub_list]
files_predictions_permut = [pd.read_csv(path_predictions_permut[x],
                                        sep='\t') for x in np.arange(
                                                len(sub_list))]
files_predictions_permut = pd.concat(files_predictions_permut)



#------------------------------------------------------------------------------
# Save concatenated data frames to files

# Give message to user
print('Saving concatenated files...')

#---
# Classification accuracies
path = results_dir + '/clf_accuracy.tsv'
files_accuracies.to_csv(path, sep='\t', index=False)

#---
# Classification accuracies permutation
path = results_dir + '/clf_accuracy_permut.tsv'
files_accuracies_permut.to_csv(path, sep='\t', index=False)

#---
# Predictions
path = results_dir + '/clf_prediction.tsv'
files_predictions.to_csv(path, sep='\t', index=False)

#---
# Predictions permutation
path = results_dir + '/clf_prediction_permut.tsv'
files_predictions_permut.to_csv(path, sep='\t', index=False)

#------------------------------------------------------------------------------
# Delete source files

# Give message to user
print('Deleting source files...')

for x in np.arange(len(sub_list)):
    remove(path_accuracies[x])
    remove(path_accuracies_permut[x])
    remove(path_predictions[x])
    remove(path_predictions_permut[x])

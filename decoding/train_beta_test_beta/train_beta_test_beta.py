r#!/usr/bin/env python3
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


''' # Define functions used in this script '''
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 1. Create_beta_matrix
# Function to create input matrix for classifier training on betas
# - Define path to betas
# - Load preprocessed betas
# - Transform into input matrix (1D voxel array)
# - Return input matrix for both buffers in list of lists
#       - 1st list: buffer 1, buffer 2
#       - 2nd lst: Mask 1, ..., Mask n
#       - Example: output[0][4] = beta_mat of Buffer1, 5th Mask
#       - Example: output[1][0] = beta_mat of Buffer2, 1st Mask

# Parameters:
# 1. bids_dir: Path to BIDS structured directory
# 2. sub_id: participant code
# 3. masks: 3D array of mask values in 1s and 0s, 4D array in case of
# multiple masks
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def Create_beta_matrix(bids_dir, sub_id, masks):

    # Pose input of masks in appropriate format in case only one mask is
    # given
    if len(masks.shape) == 3:
        # Add 4th dimension and fill first 3 with provided mask
        pattern = np.zeros([masks.shape[0], masks.shape[1], masks.shape[2], 1])
        pattern[:,:,:,0] = masks
        # Make new 4D array to masks
        masks = pattern

    # Define path which contains all corrected, smoothed, and z-scored  betas:
    corr_dir_b1 = (bids_dir + '/derivatives/' + sub_id +
                   '/glm/buffer1' + '/estimate/corr')
    corr_dir_b2 = (bids_dir + '/derivatives/' + sub_id +
                   '/glm/buffer2' + '/estimate/corr')

    # Give number of betas to load (skipping motion and mean related betas)
    req_betas = np.concatenate([np.arange(1,13),
                                np.arange(19,31)])

    # Allocate 4D array to store betas, dim 1-3: Voxel values, dim 4:
    # different betas
    beta_mat_b1 = np.zeros([masks.shape[0],
                            masks.shape[1],
                            masks.shape[2],
                            len(req_betas)])
    beta_mat_b2 = deepcopy(beta_mat_b1)

    # Load unmasked betas from both buffers
    for beta_count, beta_id in enumerate(req_betas):

        # Path to load specific beta
        path_b1 = corr_dir_b1 + '/zscbeta_' + str(beta_id).zfill(4) + '.nii'
        path_b2 = corr_dir_b2 + '/zscbeta_' + str(beta_id).zfill(4) + '.nii'

        # Load beta .nii with nibabel function
        nii_beta_b1 = nibabel.load(path_b1)
        nii_beta_b2 = nibabel.load(path_b2)

        # Transform nii file into matrix:
        volume_b1 = nii_beta_b1.get_data()
        volume_b2 = nii_beta_b2.get_data()

        # Concatenate beta into 4D array
        beta_mat_b1[:,:,:,beta_count] = volume_b1
        beta_mat_b2[:,:,:,beta_count] = volume_b2

    # Mask betas and pose into list with one entry for each mask
    # Create list of beta matrices with length of number of provided masks
    masked_beta_mat_b1 = [0] * masks.shape[3]
    masked_beta_mat_b2 = deepcopy(masked_beta_mat_b1)

    for mask_count in np.arange(masks.shape[3]):

        # Get current mask
        current_mask = masks[:,:,:,mask_count]

        # Allocate 2D array: 1st D holding all non-zero masked voxels, 2nd D
        # holding different betas
        current_beta_mat_b1 = np.zeros([np.flatnonzero(current_mask).size,
                                        len(req_betas)])
        current_beta_mat_b2 = deepcopy(current_beta_mat_b1)

        # Fill current beta mat with masked voxels for each beta
        for beta_count, beta_id in enumerate(req_betas):

            # Use mask to eliminate all values outside of the mask and
            # bring values into 1-D array for buffer 1 and 2
            masked_beta_b1 = beta_mat_b1[:,:,:,beta_count]
            masked_beta_b1 = masked_beta_b1[np.nonzero(current_mask)]

            masked_beta_b2 = beta_mat_b2[:,:,:,beta_count]
            masked_beta_b2 = masked_beta_b2[np.nonzero(current_mask)]

            # Fill current beta_mat
            current_beta_mat_b1[:,beta_count] = masked_beta_b1
            current_beta_mat_b2[:,beta_count] = masked_beta_b2

        # Transpose to make different betas the rows of the matrix
        current_beta_mat_b1 = np.transpose(current_beta_mat_b1, (1,0))
        current_beta_mat_b2 = np.transpose(current_beta_mat_b2, (1,0))

        # Fill list of different finished beta_mats for each mask
        masked_beta_mat_b1[mask_count] = current_beta_mat_b1
        masked_beta_mat_b2[mask_count] = current_beta_mat_b2

    # Return complete list of masked beta matrices in list form
    return(masked_beta_mat_b1, masked_beta_mat_b2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Function to create 4D matrix of masks
# - Define path to mask directory
# - Load masks as nii files
# - Transform into 3D arrays
# - Append 3D arrays of masks to 4D array
# - Return 4D matrix of masks


# Parameters:
# 1. bids_dir: Path to BIDS structured directory
# 2. sub_id: participant code
# 3. mask_names: Array of strings with mask names
# 4. buffering: Logical indicating if to return one mask for each buffer or
# across buffer masks (shared mask)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def Load_masks(bids_dir, sub_id, mask_names, buffering):

    # Allocate list to safe mask matrixes in
    all_masks_b1 = [0] * len(mask_names)
    all_masks_b2 = deepcopy(all_masks_b1)

    # Loop over each mask
    for mask_count, mask_id in enumerate(mask_names):

        # Get path to mask
        mask_dir = bids_dir + '/derivatives/' + sub_id + '/preproc/seg'

        # Create mask path for both buffers
        path_mask_b1 = (mask_dir + '/' + sub_id + '_mask_' + mask_id +
                        '_buffer1.nii')
        path_mask_b2 = (mask_dir + '/' + sub_id + '_mask_' + mask_id +
                        '_buffer2.nii')

        # Load mask to .nii object
        mask_b1 = nibabel.load(path_mask_b1)
        mask_b2 = nibabel.load(path_mask_b2)

        # Extract matrix from .nii
        mask_b1 = mask_b1.get_data()
        mask_b2 = mask_b2.get_data()

        # Post matrices into lists
        all_masks_b1[mask_count] = mask_b1
        all_masks_b2[mask_count] = mask_b2

    # Post list into array format with 4th D being different masks
    # Create empty 4D array with correct dimensions (given by first mask)
    all_masks_mat_b1 = np.zeros([all_masks_b1[0].shape[0],
                                 all_masks_b1[0].shape[1],
                                 all_masks_b1[0].shape[2],
                                 len(mask_names)])
    all_masks_mat_b2 = deepcopy(all_masks_mat_b1)

    # Fill 4D array with masks from lists
    for mask_count, mask_id in enumerate(mask_names):
        all_masks_mat_b1[:,:,:,mask_count] = all_masks_b1[mask_count]
        all_masks_mat_b2[:,:,:,mask_count] = all_masks_b2[mask_count]

    # Return buffer specific masks if indicated by 'buffering' parameter
    if bool(buffering):
        return(all_masks_mat_b1, all_masks_mat_b2)

    # In case combined mask is required create combined mask
    else:
        # Allocate empty 4d array holding combined masks
        all_masks_mat = np.zeros([all_masks_b1[0].shape[0],
                                 all_masks_b1[0].shape[1],
                                 all_masks_b1[0].shape[2],
                                 len(mask_names)])

        # Fill empty 4D array with combined masks
        for mask_count, mask_id in enumerate(mask_names):
            # Allocate empty mask array same shape as other masks
            mask = np.zeros([all_masks_b1[0].shape[0],
                                 all_masks_b1[0].shape[1],
                                 all_masks_b1[0].shape[2]])

            # Fill empty mask with 1 at every coordinate where both bufferd
            # masks are 1 (shared mask)
            shared_voxels = np.where(all_masks_mat_b1[:,:,:,mask_count] +
                                     all_masks_mat_b2[:,:,:,mask_count] == 2)
            mask[shared_voxels] = 1

            # Fill empty array holding all shared masks with current mask
            all_masks_mat[:,:,:,mask_count] = mask

        # Return shared masks in 4D array
        return(all_masks_mat)


''' # Start decoding script '''


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Function to decode and predict direction
# - Load all data required (motion, raw, betas, behavioral)
# - Train and test classifier for each mask and save accuracy and predictions
# - Train classifier on permuted labels and save accuracy and predictions

# Parameters:
# 1. bids_dir: BIDS structured directory
# 2. sub_id: Participant code (as string)
# 3. mask_names: List of stings with mask codes
# 4. n_permutation: Number of permutations
# 5. out_dir: Output directory to save results
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def main(bids_dir, sub_id, mask_names, n_permutation, out_dir):

    # Df to store predictions in (1st C: Subjects, 2nd C: Mask, ...) for real
    # classification and permutation
    clf_prediction = pd.DataFrame()
    clf_prediction_permut = pd.DataFrame()
    # Df to store acuracy in
    clf_accuracy = pd.DataFrame()
    clf_accuracy_permut = pd.DataFrame()



    # Give message to user which subject is investigated
    print('------------------------------------------------------------------')
    print('Working on subject:\t\t\t--|', sub_id, '|--')
    print('------------------------------------------------------------------')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Load required data
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Give message to user
    print('\tLoading required data...')

    # Unbuffered (shared mask array)
    mask_mat = Load_masks(bids_dir, sub_id, mask_names, buffering=0)
    # Buffered (two seperate mask arrays)
    mask_mat_b1, mask_mat_b2 = Load_masks(bids_dir, sub_id, mask_names,
                                          buffering=1)

    # Load betas for each buffer into 2D matrix (rows: betas, cols: voxels)
    beta_mat_b1, beta_mat_b2 = Create_beta_matrix(bids_dir, sub_id, mask_mat)

    # Relate betas to directions
    labels_beta = pd.DataFrame()
    labels_beta['angle_bin'] = np.tile(np.arange(1,7), 4)
    labels_beta['fold'] = np.repeat(np.arange(1,5), 6)
    labels_beta['prediction'] = float('nan')


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Mask specific analysis
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Loop over masks
    for mask_count, mask_id in enumerate(mask_names):

        # Give message to user: Processed mask
        print('\t', mask_id, '\t:', end='')

        # Get current mask (unbuffered and buffered)
        c_mask = mask_mat[:,:,:,mask_count]
        c_mask_b1 = mask_mat_b1[:,:,:,mask_count]
        c_mask_b2 = mask_mat_b2[:,:,:,mask_count]

        # Get current beta_mat for both buffers
        c_beta_mat_b1 = beta_mat_b1[mask_count]
        c_beta_mat_b2 = beta_mat_b2[mask_count]

        # z-score beta mat for both buffers
        c_beta_mat_b1 = scipy.stats.zscore(c_beta_mat_b1, axis=1)
        c_beta_mat_b2 = scipy.stats.zscore(c_beta_mat_b2, axis=1)

        # Allocate df holding accuracy of classification for each fold
        results_accuracy = pd.DataFrame()
        results_accuracy['buffer'] = np.repeat([1,2], 4)
        results_accuracy['fold'] = np.tile(np.array([1,2,3,4]), 2)
        results_accuracy['accuracy'] = float('nan')
        results_accuracy['correlation'] = float('nan')

        # Allocate df holding prediction for each fold
        results_prediction = pd.DataFrame()
        results_prediction['buffer'] = np.repeat([1,2], 4*6)
        results_prediction['fold'] = np.tile(
                np.repeat(np.array([1,2,3,4]), 6),
                2)
        results_prediction['angle_bin'] = np.tile([1,2,3,4,5,6], 4*2)
        results_prediction['prediction'] = float('nan')


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Start classification
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Give message to user: Classification starts
        print('\tClassification...', end='')

        # Loop over buffer
        for buffer_count, buffer_id in enumerate([1,2]):

            # Select beta matrix for specific buffer
            if buffer_id == 1:
                c_beta_mat = c_beta_mat_b1
            if buffer_id == 2:
                c_beta_mat = c_beta_mat_b2

            # Copy labels to create array holding prediction, correct labels,
            # fold, and buffer
            predict_labels = deepcopy(labels_beta)
            predict_labels['buffer'] = buffer_id

            # Loop over cross validation folds (iterated fold is testing fold)
            for fold_count, fold_id in enumerate(
                    np.unique(labels_beta.loc[:,'fold'])
                    ):

                # Get training set and training labels for fold and buffer
                train_set = c_beta_mat[labels_beta.loc[:,'fold'] != fold_id,:]

                train_labels = labels_beta.loc[
                        labels_beta.loc[:,'fold'] != fold_id,'angle_bin']


                # Get testing set and testing labels
                test_set = c_beta_mat[labels_beta.loc[:,'fold'] == fold_id,:]
                # Labels and bins for mode
                test_labels = labels_beta.loc[
                        labels_beta.loc[:,'fold'] == fold_id,'angle_bin']


                # Create classifier object
                clf = sklearn.svm.LinearSVC(C=1,
                                            max_iter=10000)

                # Train classifier
                clf.fit(train_set, train_labels)

                # Predict angle_bins
                prediction = clf.predict(test_set)

                # Put prediction into label vector
                index = labels_beta.loc[:,'fold'] == fold_id
                predict_labels.loc[index,'prediction'] = prediction


                # Save accuracy for mean and mode by comparing number of correct
                # predictions of bin to overall number of predictions
                results_accuracy.loc[fold_count + 4*buffer_count,
                                     'accuracy'] = (
                                     np.sum(np.equal(prediction,
                                                     test_labels)) /
                                     len(prediction)
                                     )

                # Save correlation between predicted bin and continuous label
                results_accuracy.loc[fold_count + 4*buffer_count,
                                     'correlation'] = (
                                     scipy.stats.pearsonr(prediction,
                                                          test_labels)[0]
                                     )

            # Save label vector with predictions into data frame
            results_prediction.loc[
                    results_prediction.loc[: , 'buffer'] == buffer_id,
                    'prediction'] = (
                    predict_labels.loc[:,'prediction'].values
                    )

        # Add subject id and mask
        results_prediction['sub_id'] = sub_id
        results_prediction['mask'] = mask_id

        # Add subject id and mask to data frame of accuracy
        results_accuracy['sub_id']= sub_id
        results_accuracy['mask']= mask_id


        # Append to results data frame
        clf_prediction = clf_prediction.append(results_prediction)
        clf_accuracy = clf_accuracy.append(results_accuracy)

        # Give message to user: Classification done
        print('Done | ', end='')


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Start permutation
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Give message to user: Permutation starts
        print('Permutation...', end='')

        # Allocate df holding accuracy of classification for each fold
        results_accuracy = pd.DataFrame()
        results_accuracy['buffer'] = np.repeat([1,2], 4)
        results_accuracy['fold'] = np.tile(np.array([1,2,3,4]), 2)
        results_accuracy['accuracy'] = float('nan')
        results_accuracy['correlation'] = float('nan')

        # Allocate df holding prediction for each fold
        results_prediction = pd.DataFrame()
        results_prediction['buffer'] = np.repeat([1,2], 4*6)
        results_prediction['fold'] = np.tile(
                np.repeat(np.array([1,2,3,4]), 6),
                2)
        results_prediction['angle_bin'] = np.tile([1,2,3,4,5,6], 4*2)
        results_prediction['prediction'] = float('nan')

        # Same classification process with permuted labels
        for perm_count in np.arange(n_permutation):

            # Loop over buffers
            for buffer_count, buffer_id in enumerate([1,2]):

                # Copy labels to create array holding prediction, correct labels,
                # fold, and buffer
                predict_labels = deepcopy(labels_beta)
                predict_labels['buffer'] = buffer_id

                # Select beta matrix for specific buffer
                if buffer_id == 1:
                    c_beta_mat = c_beta_mat_b1
                if buffer_id == 2:
                    c_beta_mat = c_beta_mat_b2

                # Loop over cross validation folds (iterated fold is testing fold)
                for fold_count, fold_id in enumerate(
                        np.unique(labels_beta.loc[:,'fold'])
                        ):

                    # Get training set and training labels for fold and buffer
                    train_set = c_beta_mat[labels_beta.loc[:,'fold'] !=
                                           fold_id,:]

                    # Shuffle labels inside each fold per buffer
                    # Buffer 1
                    train_labels = labels_beta.loc[
                            labels_beta.loc[:,'fold'] != fold_id,
                            ['angle_bin', 'fold']
                            ]
                    for x in np.unique(train_labels.loc[:,'fold']):
                        train_labels.loc[train_labels.loc[:,'fold'] == x,
                                            'angle_bin'] = sample(
                                train_labels.loc[
                                        train_labels.loc[:,'fold'] == x,
                                        'angle_bin'].tolist(),
                                        len(train_labels.loc[
                                                train_labels.loc[:,'fold'] == x,
                                                'angle_bin'].values)
                                )

                    # Set index for TRs in current buffer and fold
                    index = np.logical_and(
                                predict_labels.loc[:,'fold'] == fold_id,
                                predict_labels.loc[:,'buffer'] == buffer_id)

                    # Get testing set and testing labels
                    test_set = c_beta_mat[labels_beta.loc[:,'fold'] == fold_id,:]
                    # Labels and bins for mode
                    test_labels = labels_beta.loc[
                            labels_beta.loc[:,'fold'] == fold_id,'angle_bin']

                    # Create classifier object
                    clf = sklearn.svm.LinearSVC(C=1,
                                                max_iter=10000)

                    # Train classifier
                    clf.fit(train_set, train_labels.iloc[:,0].values)

                    # Predict angle_bins
                    prediction = clf.predict(test_set)

                    # Put prediction into label vector
                    index = labels_beta.loc[:,'fold'] == fold_id
                    predict_labels.loc[index,'prediction'] = prediction


                    # Save accuracy for mean and mode by comparing number of correct
                    # predictions of bin to overall number of predictions
                    results_accuracy.loc[fold_count + 4*buffer_count,
                                         'accuracy'] = (
                                         np.sum(np.equal(prediction,
                                                         test_labels)) /
                                         len(prediction)
                                         )

                    # Save correlation between predicted bin and continuous label
                    results_accuracy.loc[fold_count + 4*buffer_count,
                                         'correlation'] = (
                                         scipy.stats.pearsonr(prediction,
                                                              test_labels)[0]
                                         )

                # Save label vector with predictions into data frame
                results_prediction.loc[
                        results_prediction.loc[: , 'buffer'] == buffer_id,
                        'prediction'] = (
                        predict_labels.loc[:,'prediction'].values
                        )

            # Add subject id and mask
            results_prediction['sub_id'] = sub_id
            results_prediction['mask'] = mask_id
            results_prediction['permutation'] = perm_count


            # Add subject id and mask to data frame of accuracy
            results_accuracy['sub_id']= sub_id
            results_accuracy['mask']= mask_id
            results_accuracy['permutation']= perm_count

            # Append to results data frame
            clf_prediction_permut = clf_prediction_permut.append(
                    results_prediction)
            clf_accuracy_permut = clf_accuracy_permut.append(results_accuracy)

        # Give message to user: Permutation done
        print('Done')


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Save results for subject
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Sort results DFs and save to .tsv
    # Prediction
    clf_prediction = clf_prediction[['sub_id', 'mask', 'buffer', 'fold',
                                     'angle_bin','prediction']]
    path = out_dir + '/' + sub_id + '_clf_prediction.tsv'
    clf_prediction.to_csv(path, sep='\t', index=False)
    # Accuracy
    clf_accuracy = clf_accuracy[['sub_id', 'mask', 'buffer', 'fold',
                                 'accuracy','correlation']]
    path = out_dir + '/' + sub_id + '_clf_accuracy.tsv'
    clf_accuracy.to_csv(path, sep='\t', index=False)


    # Permutation files
    # Prediction
    clf_prediction_permut = clf_prediction_permut[['sub_id', 'mask', 'buffer',
                                                   'fold', 'angle_bin',
                                                   'prediction',
                                                   'permutation']]
    path = out_dir + '/' + sub_id + '_clf_prediction_permut.tsv'
    clf_prediction_permut.to_csv(path, sep='\t', index=False)
    # Accuracy
    clf_accuracy_permut = clf_accuracy_permut[['sub_id', 'mask', 'buffer',
                                               'fold', 'accuracy',
                                               'correlation',
                                               'permutation']]
    path = out_dir + '/' + sub_id + '_clf_accuracy_permut.tsv'
    clf_accuracy_permut.to_csv(path, sep='\t', index=False)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

''' Decoding for subject depending on input'''

# Set up paths
# BIDS directory (containing betas and raw data)
bids_dir = str(Path.home()) + '/direction_decoding_BIDS'
# Path of repository
repo_dir = str(Path.home() + '/direction_decoding')
# Path to save results to (relative to repo)
save_dir = (repo_dir + '/decoding/backwards_decoding/train_beta_test_raw' +
            '/results')

# Define key variables based on command line input
bids_dir = argv[1]
sub_id = argv[2]
mask_names = argv[3]
# Convert input to list
mask_names = mask_names.strip('[]').split(',')
n_permutation = int(argv[4])
out_dir = argv[5]


# Call decoding function with user inputs
main(bids_dir, sub_id, mask_names, n_permutation, out_dir)

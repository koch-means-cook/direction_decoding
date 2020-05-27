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
# 1. Create_raw_matrix
# Function to create matrix for classifier test on raw data (each TR one row)
# - Define path to raw data
# - Load smoothed or unsmoothed raw_data
# - Transform into test matrix (2D array: 1st D voxel array, 2nd D TR)
# - Return test matrix of raw data for each run and mask in list of lists
#       - 1st list: Run 1, Run 2
#       - 2nd lst: Mask 1, ..., Mask n
#       - Example: output[0][4] = raw_mat of Run 1, 5th Mask
#       - Example: output[1][0] = raw_mat of Run 2, 1st Mask

# Parameters:
# 1. bids_dir: Path to BIDS structured directory
# 2. sub_id: participant code
# 3. masks: 3D array of mask values in 1s and 0s, 4D array in case of
# multiple masks
# 4. preproc: Logical to pick preprocessed or unprocessedraw data
#       - preproc already corrected, smoothed, and z-scored the data and
#       eliminated TRs which are not in the behavioral data
#       - 0: Use unprocessed data
#       - 1: Use preprocessed data
# 5. smooth: In case of unprocessed data. Boolean to determine if data should
# be smoothed
#       - 0: Data will not be smoothed
#       - 1: Data will be smoothed
# 6. smooth_kernel: Number to determine kernel size of smoothing (in case
# data will be smoothed)
# 7. concat: Boolean to determine if separated runs should be returned
# concatenated or as separated runs (in list format)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def Create_raw_matrix(bids_dir, sub_id, masks, preproc, smooth, smooth_kernel,
                      concat):

    # Pose input of masks in appropriate format in case only one mask is
    # given
    if len(masks.shape) == 3:
        # Add 4th dimension and fill first 3 with provided mask
        pattern = np.zeros([masks.shape[0], masks.shape[1], masks.shape[2], 1])
        pattern[:,:,:,0] = masks
        # Make new 4D array to masks
        masks = pattern

    # Provide path to raw data dependent on taking preprocessed or unprocessed
    # data
    if bool(preproc):
        raw_dir_r1 = (bids_dir + '/derivatives/' + sub_id +
                      '/preproc/classification_raw/zscsar' + sub_id +
                      '_task-feedback1_bold.nii')
        raw_dir_r2 = (bids_dir + '/derivatives/' + sub_id +
                      '/preproc/classification_raw/zscsar' + sub_id +
                      '_task-feedback2_bold.nii')
    else:
        raw_dir_r1 = (bids_dir + '/derivatives/' + sub_id +
                      '/preproc/smooth/sar' + sub_id +
                      '_task-feedback1_bold.nii')
        raw_dir_r2 = (bids_dir + '/derivatives/' + sub_id +
                      '/preproc/smooth/sar' + sub_id +
                      '_task-feedback2_bold.nii')

    # Load selected raw data into matrix
    nii_raw_r1 = nibabel.load(raw_dir_r1)
    nii_raw_r2 = nibabel.load(raw_dir_r2)

    # Smooth data with provided kernel size in case unprocessed data is loaded
    if bool(smooth):
        nii_raw_r1 = image.smooth_img(nii_raw_r1, smooth_kernel)
        nii_raw_r2 = image.smooth_img(nii_raw_r2, smooth_kernel)

    # Cast nii data into matrix
    volume_raw_r1 = nii_raw_r1.get_data()
    volume_raw_r2 = nii_raw_r2.get_data()

    # Ger number of TRs in each run
    n_tr_r1 = volume_raw_r1.shape[3]
    n_tr_r2 = volume_raw_r2.shape[3]

    # Allocate list to store raw matrices for different masks
    masked_raw_mat_r1 = [0] * masks.shape[3]
    masked_raw_mat_r2 = deepcopy(masked_raw_mat_r1)
    # Allocate list for concatenated runs
    masked_raw_mat_concat = deepcopy(masked_raw_mat_r1)

    # Create different raw matrix for each mask
    for mask_count in np.arange(masks.shape[3]):

        # Get current mask
        current_mask = masks[:,:,:,mask_count]

        # Allocate 2D array: 1st D holding all non-zero masked voxels, 2nd D
        # holding different TRs
        current_raw_mat_r1 = np.zeros([np.flatnonzero(current_mask).size,
                                       n_tr_r1])
        current_raw_mat_r2 = np.zeros([np.flatnonzero(current_mask).size,
                                       n_tr_r2])

        # Fill allocated raw matrices with masked TRs (Separated for runs
        # since number of TRs differs)
        # Run 1
        for tr_count in np.arange(volume_raw_r1.shape[3]):

            # Isolate current TR
            current_tr = volume_raw_r1[:,:,:,tr_count]

            # Fill raw matrix with masked values for current TR
            current_raw_mat_r1[:,tr_count] = (
                    current_tr[np.nonzero(current_mask)]
                    )

        # Run 2
        for tr_count in np.arange(volume_raw_r2.shape[3]):
            current_tr = volume_raw_r2[:,:,:,tr_count]
            current_raw_mat_r2[:,tr_count] = (
                    current_tr[np.nonzero(current_mask)]
                    )

        # Transpose to make different TRs the rows of the matrix
        current_raw_mat_r1 = np.transpose(current_raw_mat_r1, (1,0))
        current_raw_mat_r2 = np.transpose(current_raw_mat_r2, (1,0))

        # Fill list of raw matrices with raw matrix for current mask
        masked_raw_mat_r1[mask_count] = current_raw_mat_r1
        masked_raw_mat_r2[mask_count] = current_raw_mat_r2

        # Create concatenated runs to return
        masked_raw_mat_concat[mask_count] = np.concatenate(
                [masked_raw_mat_r1[mask_count],
                masked_raw_mat_r2[mask_count]])

    # Check for concatenate variable
    # Do not concatenate runs
    if bool(not concat):
        # Return complete list of masked raw matrices in list form
        return(masked_raw_mat_r1, masked_raw_mat_r2)
    elif bool(concat):
        # Return concatenated runs
        return(masked_raw_mat_concat)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# 2. Create_beta_matrix
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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Function to load behavioral data about runs
# - Load behavioral data about runs
# - Option to relate run1 to run2 (e.g. if last TR run 1 = 100, First TR of
# run 2 = 101)
# - Option to concatenate behavioral data
# - Add overall event counter


# Parameters:
# 1. bids_dir: Path to BIDS structured directory
# 2. sub_id: participant code
# 3. relate: Boolean if both runs should be ralted to each other (e.g. )
# 4. max_tr_r1: Maximum of TRs of first run (used to relate TRs between runs)
# 5. concat: Boolean to determine if separated runs should be returned
# concatenated or as separated runs (in list format)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def Load_behavioral(bids_dir, sub_id, relate, max_tr_r1, concat):

    # Paths to both behavioral runs
    path_r1 = (bids_dir + '/derivatives/' + sub_id + '/behavior/' + sub_id +
               '_3d1.tsv')
    path_r2 = (bids_dir + '/derivatives/' + sub_id + '/behavior/' + sub_id +
               '_3d2.tsv')

    # Load behavioral .tsv files
    behav_r1 = pd.read_csv(path_r1, sep='\t')
    behav_r2 = pd.read_csv(path_r2, sep='\t')

    # Create joined event counter column
    behav_r1['overall event'] = 0
    behav_r2['overall event'] = 0

    # Fill overall counter with increasing events from folds
    # Fold 1
    behav_r1.loc[behav_r1.loc[:,'fold_id'] == 1 ,'overall event']  = (
            behav_r1.loc[behav_r1.loc[:,'fold_id'] ==1 ,'event in fold'])
    # Fold 2
    behav_r1.loc[behav_r1.loc[:,'fold_id'] == 2 ,'overall event']  = (
            behav_r1.loc[behav_r1.loc[:,'fold_id'] == 2 ,'event in fold'] +
            np.max(behav_r1.loc[behav_r1.loc[:,'fold_id'] == 1 ,
                                'overall event']))
    # Fold 3
    behav_r2.loc[behav_r2.loc[:,'fold_id'] == 3 ,'overall event']  = (
            behav_r2.loc[behav_r2.loc[:,'fold_id'] == 3 ,'event in fold'] +
            np.max(behav_r1.loc[behav_r1.loc[:,'fold_id'] == 2 ,
                                'overall event']))
    # Fold 4
    behav_r2.loc[behav_r2.loc[:,'fold_id'] == 4 ,'overall event']  = (
            behav_r2.loc[behav_r2.loc[:,'fold_id'] == 4 ,'event in fold'] +
            np.max(behav_r2.loc[behav_r2.loc[:,'fold_id'] == 3 ,
                                'overall event']))

    # If both runs should be related
    if bool(relate):

        # Make timing relative to previous timing
        behav_r2.loc[:, 'Time'] += behav_r1['Time'].iloc[-1]

        # Make TRs relative to first run
        behav_r2.loc[:, 'TR number corrected'] += (
                max_tr_r1)
        behav_r2.loc[:, 'TR number uncorrected'] += (
                max_tr_r1)

    # If both runs should be concatenated
    if bool(concat):

        # Concatenate both runs
        frames = [behav_r1, behav_r2]
        behav = pd.concat(frames)

        # Return concatenated runs
        return(behav)

    # If runs should be returned separately
    elif bool(not concat):
        return(behav_r1, behav_r2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Function to load motion data of subject
# - Load motion data about runs stemming from realignment
# - Add overall event counter

# Parameters:
# 1. bids_dir: Path to BIDS structured directory
# 2. sub_id: participant code
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def Load_motion(bids_dir, sub_id):

    # Paths to both behavioral runs
    path_r1 = (bids_dir + '/derivatives/' + sub_id + '/preproc/realign/' +
               sub_id + '_mov_reg1.txt')
    path_r2 = (bids_dir + '/derivatives/' + sub_id + '/preproc/realign/' +
               sub_id + '_mov_reg2.txt')

    # Load behavioral .tsv files
    motion_r1 = pd.read_csv(path_r1, delimiter=r"\s+", header=None)
    motion_r2 = pd.read_csv(path_r2, delimiter=r"\s+", header=None)

    return(motion_r1, motion_r2)



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

    #- - - - - - - - - - - - - - - - -
    # To select certain subjects:
    #sub_list = sub_list[0:1]
    #- - - - - - - - - - - - - - - - -

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Construct loop to let the classification run for separate masks
    #mask_names = ['beta',
    #              'entorhinal',
    #              'hc',
    #              'm1',
    #              'rsc',
    #              'subiculum',
    #              'thal',
    #              'v1',
    #              'ent_hc']
    # Alternative mask_names for only specific masks:
    #mask_names = ['v1', 'rsc', 'm1']

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

    # Load raw data into same format as beta matrix, masked by shared mask
    raw_mat_r1, raw_mat_r2 = Create_raw_matrix(bids_dir, sub_id, mask_mat,
                                               preproc=0,
                                               smooth=1, smooth_kernel=3,
                                               concat=0)
    raw_mat = Create_raw_matrix(bids_dir, sub_id, mask_mat,
                                           preproc=0,
                                           smooth=1, smooth_kernel=3,
                                           concat=1)

    # Load subject motion from realignment for signal.claen
    motion_r1, motion_r2 = Load_motion(bids_dir, sub_id)

    # Signal clean raw mat runs for each mask
    raw_mat_r1 = list(
            signal.clean(x,
                         confounds=motion_r1.values,
                         high_pass = 1/128,
                         t_r = 2.4)
            for x in raw_mat_r1)
    raw_mat_r2 = list(
            signal.clean(x,
                         confounds=motion_r2.values,
                         high_pass = 1/128,
                         t_r = 2.4)
            for x in raw_mat_r2)

    # Voxel-internal z-scoring
    raw_mat_r1 = list(scipy.stats.zscore(x, axis=0) for x in raw_mat_r1)
    raw_mat_r2 = list(scipy.stats.zscore(x, axis=0) for x in raw_mat_r2)

    # TR wise z-scoring
    raw_mat_r1 = list(scipy.stats.zscore(x, axis=1) for x in raw_mat_r1)
    raw_mat_r2 = list(scipy.stats.zscore(x, axis=1) for x in raw_mat_r2)


    # Get maximum of TRs in first run to relate TRs between behavioral runs
    max_tr_r1 = raw_mat_r1[0].shape[0]
    # Load behavioral runs as concatenated file
    behav = Load_behavioral(bids_dir,
                            sub_id,
                            relate=1,
                            max_tr_r1=max_tr_r1,
                            concat=1)


    # Relate TRs to behavioral
    # Data frame holding TRs and associated direction
    labels_raw = pd.DataFrame()
    # trs of functional data (+1 since TRs in behav file are counted
    # from starting 0)
    labels_raw['tr'] = np.arange(raw_mat[0].shape[0]) + 1
    # Add column holding directional values for each TR
    labels_raw['angle_mode'] = float('nan')
    labels_raw['angle_mean'] = float('nan')
    labels_raw['fold'] = float('nan')
    labels_raw['buffer'] = float('nan')



    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Get directional value for each TR
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Loop over each TR with directional event (mode and mean)
    for tr_count, tr_id in enumerate(
            np.unique(behav['TR number corrected'])
            ):
        # Index of behavioral data for a certain TR
        index = behav.loc[:,'TR number corrected'] == tr_id
        # Get distribution of raw angles during TR
        angle_dist = np.round(
                behav.loc[index,'direction angle by YAW'].values,
                2)
        # Get mode and mean of angles during one TR
        mode_angle = scipy.stats.mode(angle_dist).mode[0]
        mean_angle = np.mean(angle_dist)
        # Get fold the TR is in (can also be in multiple folds, so take mode)
        fold_dist = behav.loc[index, 'fold_id'].values
        mode_fold = scipy.stats.mode(fold_dist).mode[0]
        # Get buffer TR is in
        buffer_dist = behav.loc[index, 'buffer']
        mode_buffer = scipy.stats.mode(buffer_dist).mode[0]

        # Fill label array with directional events and fold
        # Eliminate cases in which corrected TR is beyond the collected TRs
        # (since corrected TRs is +2 TRs that might not have been measured)
        if tr_id <= raw_mat[0].shape[0]:
            index = labels_raw.loc[:,'tr'] == tr_id
            labels_raw.loc[index,'angle_mode'] = mode_angle
            labels_raw.loc[index,'angle_mean'] = np.round(mean_angle, 2)
            labels_raw.loc[index,'fold'] = mode_fold
            labels_raw.loc[index, 'buffer'] = mode_buffer

    # Get label vector that only contains events (eliminate NaN entries)
    index = np.where(~np.isnan(labels_raw.loc[:,'fold']))[0]
    predict_labels = labels_raw.loc[index,:]

    # Add bin and prediction column to label vector
    predict_labels['bin_mode'] = float('nan')
    predict_labels['bin_mean'] = float('nan')
    predict_labels['prediction'] = float('nan')

    # Fill bin columns with correct bin
    predict_labels['bin_mode'] = np.floor(
            predict_labels['angle_mode'] / 60
            ) + 1
    predict_labels['bin_mean'] = np.floor(
            predict_labels['angle_mean'] / 60
            ) + 1



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

        # z-score beta mat (whole single beta) for both buffers
        c_beta_mat_b1 = scipy.stats.zscore(c_beta_mat_b1, axis=1)
        c_beta_mat_b2 = scipy.stats.zscore(c_beta_mat_b2, axis=1)

        # Get current concatenated raw_mat (from shared mask)
        c_raw_mat_r1 = raw_mat_r1[mask_count]
        c_raw_mat_r2 = raw_mat_r2[mask_count]

        # Concatenate cleaned runs
        c_raw_mat = np.concatenate([c_raw_mat_r1, c_raw_mat_r2])

        # Eliminate volumes from raw_mat which can't be predicted (since TRs
        # which where in no events)
        index = predict_labels.loc[:,'tr'] - 1
        predict_raw_mat = c_raw_mat[index,:]

        # Allocate df holding accuracy of classification for each fold
        results_accuracy = pd.DataFrame()
        results_accuracy['buffer'] = np.repeat([1,2], 4)
        results_accuracy['fold'] = np.tile(np.array([1,2,3,4]), 2)
        results_accuracy['accuracy_mean'] = float('nan')
        results_accuracy['accuracy_mode'] = float('nan')
        results_accuracy['correlation_mean'] = float('nan')
        results_accuracy['correlation_mode'] = float('nan')



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

            # Loop over cross validation folds (iterated fold is testing fold)
            for fold_count, fold_id in enumerate(
                    np.unique(predict_labels.loc[:,'fold'])
                    ):

                # Get training set and training labels for fold and buffer
                train_set = c_beta_mat[labels_beta.loc[:,'fold'] != fold_id,:]

                train_labels = labels_beta.loc[
                        labels_beta.loc[:,'fold'] != fold_id,'angle_bin']

                # Set index for TRs in current buffer and fold
                index = np.logical_and(
                            predict_labels.loc[:,'fold'] == fold_id,
                            predict_labels.loc[:,'buffer'] == buffer_id)

                # Get testing set and testing labels (over concatenated
                # raw_data)
                test_set = predict_raw_mat[index,:]
                # Labels and bins for mode
                test_labels_mode = predict_labels.loc[index,
                                                      'angle_mode'].values
                test_bins_mode = predict_labels.loc[index,'bin_mode'].values
                # Labels and bins for mean
                test_labels_mean = predict_labels.loc[index,
                                                      'angle_mean'].values
                test_bins_mean = predict_labels.loc[index,'bin_mean'].values

                # Create classifier object
                clf = sklearn.svm.LinearSVC(C=1,
                                            max_iter=10000)

                # Train classifier
                clf.fit(train_set, train_labels)

                # Predict angle_bins
                prediction = clf.predict(test_set)

                # Put prediction into label vector
                predict_labels.loc[index,'prediction'] = prediction


                # Save accuracy for mean and mode by comparing number of correct
                # predictions of bin to overall number of predictions
                # Mean
                results_accuracy.loc[fold_count + 4*buffer_count,
                                     'accuracy_mean'] = (
                                     np.sum(np.equal(prediction,
                                                     test_bins_mean)) /
                                     len(prediction)
                                     )
                # Mode
                results_accuracy.loc[fold_count + 4*buffer_count,
                                     'accuracy_mode'] = (
                                     np.sum(np.equal(prediction,
                                                     test_bins_mode)) /
                                     len(prediction)
                                     )

                # Save correlation between predicted bin and continuous label
                # Mean
                results_accuracy.loc[fold_count + 4*buffer_count,
                                     'correlation_mean'] = (
                                     scipy.stats.pearsonr(prediction,
                                                          test_labels_mean)[0]
                                     )
                # Mode
                results_accuracy.loc[fold_count + 4*buffer_count,
                                     'correlation_mode'] = (
                                     scipy.stats.pearsonr(prediction,
                                                          test_labels_mode)[0]
                                     )

        # Save label vector with predictions into data frame
        results_prediction = deepcopy(predict_labels)
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
        results_accuracy['accuracy_mean'] = float('nan')
        results_accuracy['accuracy_mode'] = float('nan')
        results_accuracy['correlation_mean'] = float('nan')
        results_accuracy['correlation_mode'] = float('nan')

        # Same classification process with permuted labels
        for perm_count in np.arange(n_permutation):

            # Loop over buffers
            for buffer_count, buffer_id in enumerate([1,2]):

                # Select beta matrix for specific buffer
                if buffer_id == 1:
                    c_beta_mat = c_beta_mat_b1
                if buffer_id == 2:
                    c_beta_mat = c_beta_mat_b2

                # Loop over cross validation folds (iterated fold is testing fold)
                for fold_count, fold_id in enumerate(
                        np.unique(predict_labels.loc[:,'fold'])
                        ):

                    # Get training set and training labels for fold (over
                    # concatenated buffers of betas)
                    train_set = c_beta_mat[
                            labels_beta.loc[:,'fold'] != fold_id,
                            :]

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

                    # Get testing set and testing labels (over concatenated
                    # raw_data)
                    test_set = predict_raw_mat[index,:]
                    # Labels and bins for mode
                    test_labels_mode = predict_labels.loc[index,
                                                          'angle_mode'].values
                    test_bins_mode = predict_labels.loc[index,'bin_mode'].values
                    # Labels and bins for mean
                    test_labels_mean = predict_labels.loc[index,
                                                          'angle_mean'].values
                    test_bins_mean = predict_labels.loc[index,'bin_mean'].values

                    # Create classifier object
                    clf = sklearn.svm.LinearSVC(C=1,
                                                max_iter=10000)

                    # Train classifier
                    clf.fit(train_set, train_labels.iloc[:,0].values)

                    # Predict angle_bins
                    prediction = clf.predict(test_set)

                    # Put prediction into label vector
                    predict_labels.loc[index,'prediction'] = prediction


                    # Save accuracy for mean and mode by comparing number of
                    # correct predictions of bin to overall number of predictions
                    # Mean
                    results_accuracy.loc[fold_count + 4*buffer_count,'accuracy_mean'] = (
                            np.sum(np.equal(prediction, test_bins_mean)) /
                            len(prediction)
                            )
                    # Mode
                    results_accuracy.loc[fold_count + 4*buffer_count,'accuracy_mode'] = (
                            np.sum(np.equal(prediction, test_bins_mode)) /
                            len(prediction)
                            )

                    # Save correlation between predicted bin and continuous label
                    # Mean
                    results_accuracy.loc[fold_count + 4*buffer_count,'correlation_mean'] = (
                            scipy.stats.pearsonr(prediction, test_labels_mean)[0]
                            )
                    # Mode
                    results_accuracy.loc[fold_count + 4*buffer_count,'correlation_mode'] = (
                            scipy.stats.pearsonr(prediction, test_labels_mode)[0]
                            )

            # Save label vector with predictions into data frame
            results_prediction = deepcopy(predict_labels)
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
                                     'tr',
                                     'angle_mode', 'angle_mean',
                                     'bin_mode', 'bin_mean',
                                     'prediction']]
    path = out_dir + '/' + sub_id + '_clf_prediction.tsv'
    clf_prediction.to_csv(path, sep='\t', index=False)
    # Accuracy
    clf_accuracy = clf_accuracy[['sub_id', 'mask', 'buffer', 'fold',
                                 'accuracy_mode', 'accuracy_mean',
                                 'correlation_mode', 'correlation_mean']]
    path = out_dir + '/' + sub_id + '_clf_accuracy.tsv'
    clf_accuracy.to_csv(path, sep='\t', index=False)


    # Permutation files
    # Prediction
    clf_prediction_permut = clf_prediction_permut[['sub_id', 'mask',
                                                   'buffer',
                                                   'fold',
                                                   'tr',
                                                   'angle_mode',
                                                   'angle_mean',
                                                   'bin_mode', 'bin_mean',
                                                   'prediction',
                                                   'permutation']]
    path = out_dir + '/' + sub_id + '_clf_prediction_permut.tsv'
    clf_prediction_permut.to_csv(path, sep='\t', index=False)
    # Accuracy
    clf_accuracy_permut = clf_accuracy_permut[['sub_id', 'mask',
                                               'buffer',
                                               'fold',
                                               'accuracy_mode',
                                               'accuracy_mean',
                                               'correlation_mode',
                                               'correlation_mean',
                                               'permutation']]
    path = out_dir + '/' + sub_id + '_clf_accuracy_permut.tsv'
    clf_accuracy_permut.to_csv(path, sep='\t', index=False)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Basic plotting
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#    for mask_count, mask_id in enumerate(mask_names):
#
#        fig = plt.figure(figsize=(8,8))
#        gs = gridspec.GridSpec(2,3, wspace=0.3, hspace=0.2)
#
#        mask_results = clf_prediction.loc[clf_prediction['mask'] == mask_id,:]
#
#        for bin_count, bin_id in enumerate(np.arange(1,7)):
#
#            pred_dist = mask_results.loc[mask_results['bin_mode'] == bin_id,
#                                         'prediction']
#
#            ax = fig.add_subplot(gs[bin_count])
#            ax.hist(pred_dist, bins=np.arange(1,8), density=True, rwidth=0.8)
#            ax.set_xticks(np.arange(1,7)+0.5)
#            ax.set_xticklabels(np.arange(1,7))
#            ax.set_ylim(bottom=0, top=1)
#            ax.set_title('Direction ' + str(bin_id), weight='bold')
#            fig.add_subplot(ax)
#        plt.show()

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

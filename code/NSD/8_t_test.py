#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:07:09 2023

@author: amysentis
"""

def load_file(input_path, dataset_names, flags):
    """
    load hdf5 file
    """

    f = h5py.File(input_path, 'r')
    data = []

    for (dataset_name, flag) in zip(dataset_names, flags):
        if flag:
            data_tmp = f[dataset_name][()]
        else:
            data_tmp = f[dataset_name][:]
        data.append(data_tmp)

    f.close()

    return data


def save_file(output_path, output_filename, files, dataset_names):
    """
    save hdf5 file

    inputs: output path to save to, output filename, output file
    """

    file_outpath = os.path.join(output_path, output_filename)

    f = h5py.File(file_outpath, "w")

    for (file, dataset_name) in zip(files, dataset_names):
        f.create_dataset(dataset_name, data=file)

    f.close()
    print("saved to file:", file_outpath)



#imports
import h5py
import os
import numpy as np
import pandas as pd
import time
from nilearn import image, masking
from scipy.stats import t, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection, multipletests


start = time.time()

data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"
subjects = [1,2,3,4,5,6,7,8]
#subjects = [2,3,4,5,8]
output_data_path = "/home/amysentis/dynamicHR/NSD/analysis/"

#parameters
time_shift = 6
time_snippet = "time" + str(time_shift)
n_features = 1271436

union_gm_mask_path = os.path.join(data_path, "group", "union_gm_mask-MNI-nearest.nii.gz")
union_gm_mask = image.load_img(union_gm_mask_path)


############### within subject one-sided t-tests, and average probability images ###############

print("")
print ("within subject one-sided t-tests")
print("")

#subjects = [1]

for subject in subjects:

    subj = "subj0" + str(subject)

    print("=========================")
    print("")
    print(subj)

    subj_file_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/" + subj + "/" + subj + "-session-run_list.txt"
    subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
    str_subj_list = subj_list.tolist()
    n_runs = len(str_subj_list)

    projections = np.zeros((n_runs, n_features))

    print("loading data...")

    for i, entry in enumerate(str_subj_list):

        #print(entry)

        subj, session, run = entry.split('-')

        proj_filename = session + "_" + run + "_var_projections_union_MNI_" + time_snippet + ".hdf5"
        proj_path = os.path.join(output_data_path, subj, "models", "outputs", proj_filename)
        [proj] = load_file(proj_path, ['projections'], [False])
        projections[i,:] = proj


    #t-test
    print("")
    print("calculating one-sided t-tests...")
    # t_less, p_less =  np.apply_along_axis(ttest_1samp, axis=0, arr=projections, popmean=0, alternative='less')
    # print(np.nanmin(t_less), np.nanmax(t_less))
    # t_greater, p_greater =  np.apply_along_axis(ttest_1samp, axis=0, arr=projections, popmean=0, alternative='greater')
    # print(np.nanmin(t_greater), np.nanmax(t_greater))
    t, p =  np.apply_along_axis(ttest_1samp, axis=0, arr=projections, popmean=0)
    index1 = np.nanargmin(t)
    index2 = np.nanargmax(t)
    #print("t", t[index1-5:index1+5])
    #print("p", p[index1-5:index1+5])
    #print("t", np.nanmin(t), np.nanmax(t))
    t_neg = np.where(t<0, t, 0)
    #print("t neg", np.nanmin(t_neg), np.nanmax(t_neg))
    p_neg = np.where(t<0, p, np.nan)
    #print("t neg", t_neg[index1-5:index1+5])
    #print("p neg", p_neg[index1-5:index1+5])
    t_pos = np.where(t>0, t, 0)
    #print("t pos", np.nanmin(t_pos), np.nanmax(t_pos))
    p_pos = np.where(t>0, p, np.nan)
    #print("t pos", t_pos[index2-5:index2+5])
    #print("p pos", p_pos[index2-5:index2+5])

    #fdr correction 
    print("fdr correction...")
    alphas = ['0.05', '0.01']

    #fixing nan propagation issue in fdr correction
    # p_less_mask = np.isfinite(p_less)
    # p_less_values_corrected = np.full(p_less.shape, np.nan)
    # rejected_less = np.full(p_less.shape, True)
    # p_greater_mask = np.isfinite(p_greater)
    # p_greater_values_corrected = np.full(p_greater.shape, np.nan)
    # rejected_greater = np.full(p_greater.shape, True)
    p_neg_mask = np.isfinite(p_neg)
    p_neg_values_corrected = np.full(p_neg.shape, np.nan)
    rejected_neg = np.full(p_neg.shape, True)
    p_pos_mask = np.isfinite(p_pos)
    p_pos_values_corrected = np.full(p_pos.shape, np.nan)
    rejected_pos = np.full(p_pos.shape, True)

    for alpha in alphas:

        print(alpha)

        # rejected_less[p_less_mask], p_less_values_corrected[p_less_mask], a1, a2 = multipletests(p_less[p_less_mask], alpha=float(alpha), method='fdr_bh', is_sorted=False, returnsorted=False)
        # t_less_thresholded = np.where(rejected_less, t_less, 0)
        # rejected_greater[p_greater_mask], p_greater_values_corrected[p_greater_mask], a1, a2 = multipletests(p_greater[p_greater_mask], alpha=float(alpha), method='fdr_bh', is_sorted=False, returnsorted=False)
        # t_greater_thresholded = np.where(rejected_greater, t_greater, 0)
        # print(t_greater_thresholded)
        rejected_neg[p_neg_mask], p_neg_values_corrected[p_neg_mask], a1, a2 = multipletests(p_neg[p_neg_mask], alpha=float(alpha), method='fdr_bh', is_sorted=False, returnsorted=False)
        t_neg_thresholded = np.where(rejected_neg, t_neg, 0)
        #print("t neg thresholded", t_neg_thresholded[index1-5:index1+5])
        rejected_pos[p_pos_mask], p_pos_values_corrected[p_pos_mask], a1, a2 = multipletests(p_pos[p_pos_mask], alpha=float(alpha), method='fdr_bh', is_sorted=False, returnsorted=False)
        t_pos_thresholded = np.where(rejected_pos, t_pos, 0)
        #print("t pos thresholded", t_pos_thresholded[index2-5:index2+5])

        #generate binary mask
        mask_negative = np.where(t_neg_thresholded < 0, 1, 0)
        #print("mask neg", mask_negative[index1-5:index1+5])
        mask_positive = np.where(t_pos_thresholded > 0, 1, 0)
        #print("mask pos", mask_positive[index2-5:index2+5])

        #save as hdf5 file
        hdf5_output_path = os.path.join(output_data_path, subj, "models", "outputs")
        str_alpha = alpha.split('.')[1]
        #output_filename = subj + "_t_test_less_a" + str_alpha + ".hdf5"
        #files = [t_less, p_less, rejected_less, p_less_values_corrected, t_less_thresholded, mask_negative]
        output_filename = subj + "_t_test_neg_a" + str_alpha + ".hdf5"
        files = [t_neg, p_neg, rejected_neg, p_neg_values_corrected, t_neg_thresholded, mask_negative]
        dataset_names = ['t', 'p', 'rejected', 'p_corrected', 't_thresholded', 'mask']
        save_file(hdf5_output_path, output_filename, files, dataset_names)
        #output_filename = subj + "_t_test_greater_a" + str_alpha + ".hdf5"
        #files = [t_greater, p_greater, rejected_greater, p_greater_values_corrected, t_greater_thresholded, mask_positive]
        output_filename = subj + "_t_test_pos_a" + str_alpha + ".hdf5"
        files = [t_pos, p_pos, rejected_pos, p_pos_values_corrected, t_pos_thresholded, mask_positive]
        save_file(hdf5_output_path, output_filename, files, dataset_names)


for alpha in alphas:

    print(alpha)

    all_masks_positive = np.zeros((len(subjects), n_features))
    all_masks_negative = np.zeros((len(subjects), n_features))
    str_alpha = alpha.split('.')[1]

    for i, subject in enumerate(subjects):

        subj = "subj0" + str(subject)

        # mask_less_filename = subj + "_t_test_less_a" + str_alpha + ".hdf5"
        # mask_less_path = os.path.join(output_data_path, subj, "models", "outputs", mask_less_filename)
        # [mask_less] = load_file(mask_less_path, ['mask'], [False])
        # all_masks_negative[i,:] = mask_less
        # mask_greater_filename = subj + "_t_test_greater_a" + str_alpha + ".hdf5"
        # mask_greater_path = os.path.join(output_data_path, subj, "models", "outputs", mask_greater_filename)
        # [mask_greater] = load_file(mask_greater_path, ['mask'], [False])
        # all_masks_positive[i,:] = mask_greater
        # print(all_masks_positive)
        mask_neg_filename = subj + "_t_test_neg_a" + str_alpha + ".hdf5"
        mask_neg_path = os.path.join(output_data_path, subj, "models", "outputs", mask_neg_filename)
        [mask_neg] = load_file(mask_neg_path, ['mask'], [False])
        all_masks_negative[i,:] = mask_neg
        mask_pos_filename = subj + "_t_test_pos_a" + str_alpha + ".hdf5"
        mask_pos_path = os.path.join(output_data_path, subj, "models", "outputs", mask_pos_filename)
        [mask_pos] = load_file(mask_pos_path, ['mask'], [False])
        all_masks_positive[i,:] = mask_pos
        #print(all_masks_positive)

    ave_mask_negative = np.mean(all_masks_negative, axis=0)
    ave_mask_positive = np.mean(all_masks_positive, axis=0)
    #print(ave_mask_positive.shape)
    #print(ave_mask_positive)

    #save
    hdf5_output_path = os.path.join(output_data_path, "group")
    output_filename = "t_test_negative_prob_a" + str_alpha + ".hdf5"
    save_file(hdf5_output_path, output_filename, [ave_mask_negative], ['ave_mask'])
    output_filename = "t_test_positive_prob_a" + str_alpha + ".hdf5"
    save_file(hdf5_output_path, output_filename, [ave_mask_positive], ['ave_mask'])

    ave_mask_neg_img = masking.unmask(ave_mask_negative, union_gm_mask, order='F')
    output_path = os.path.join(output_data_path, "group", "t_test_negative_prob_a" + str_alpha + ".nii.gz")
    ave_mask_neg_img.to_filename(output_path)
    print("saving nifti:", output_path) 
    ave_mask_pos_img = masking.unmask(ave_mask_positive, union_gm_mask, order='F')
    output_path = os.path.join(output_data_path, "group", "t_test_positive_prob_a" + str_alpha + ".nii.gz")
    ave_mask_pos_img.to_filename(output_path)
    print("saving nifti:", output_path) 


end = time.time()
print("")
print("Total time:", end - start)

###############  ###############
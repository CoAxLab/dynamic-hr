#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute and save residuals from GLM of variability regressors (X) onto voxel data (Y)

"""

#imports
import h5py
import numpy as np
import os
import pandas as pd
import time
from nilearn import image, masking
from nilearn.glm.first_level import FirstLevelModel


#data files
input_data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/data/nsddata/ppdata/"
output_data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"

subj_file_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/subj267-session-run_list.txt"
subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
str_subj_list = subj_list.tolist()
#str_subj_list = ["subj01-session21-run02", "subj01-session21-run04", "subj02-session21-run02", "subj03-session21-run02"]
last_subj = "subj0"

#parameters
tr = 1.333171
n_vols = 226
frame_times = np.arange(n_vols)*tr

start = time.time()

for entry in str_subj_list:

    print(entry)

    subj, session, run = entry.split('-')

    if subj != last_subj:
        #load subj gm mask
        print(subj, "- loading gm mask")
        mask_filename = subj + "_gm_mask.nii.gz"
        mask_path = os.path.join(output_data_path, subj, mask_filename)
        mask_img = image.load_img(mask_path)

    print(session, run)

    #load motion regressors
    motion_filename = "motion_" + session + "_" + run + ".tsv"
    motion_path = os.path.join(input_data_path, subj, "func1pt8mm", "motion", motion_filename)
    motion_df = pd.read_csv(motion_path, sep = '\t', header=None)

    #load timeseries nii file
    timeseries_filename = "timeseries_" + session + "_" + run + ".nii.gz"
    timeseries_path = os.path.join(input_data_path, subj, "func1pt8mm", "timeseries", timeseries_filename)
    timeseries_img = image.load_img(timeseries_path)

    #load niphlem's var regressor file
    # var_filename = session + "_" + run + "_var_regressors.hdf5"
    # var_path = os.path.join(output_data_path, subj, "physio", var_filename)
    # f = h5py.File(var_path, 'r')
    # hv_reg = f['HV_regressors'][:]
    # rv_reg = f['RV_regressors'][:]
    # f.close()

    #load niphlem's retroicor regressor file
    retro_filename = session + "_" + run + "_retro_regressors.hdf5"
    retro_path = os.path.join(output_data_path, subj, "physio", retro_filename)
    f = h5py.File(retro_path, 'r')
    retro_puls_reg = f['retro_puls_regressors'][:]
    retro_resp_reg = f['retro_resp_regressors'][:]
    f.close()

    #set up design matrix
    # dm_var = np.column_stack((np.ones(len(frame_times)), #intercept
    #                         motion_df.to_numpy(), 
    #                         hv_reg, #HR variability regressors
    #                         rv_reg #resp variability regressors
    #                         ))
    # dm_var_df = pd.DataFrame(dm_var, 
    #                         columns=["intercept", 
    #                                 "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", 
    #                                 "hv", "rv"]
    #                         )
    # dm_var_df.index=frame_times
    #print(dm_var_df.head())

    dm_retro = np.column_stack((np.ones(len(frame_times)), #intercept
                                motion_df.to_numpy(), 
                                retro_puls_reg, #HR retro regressors
                                retro_resp_reg #resp retro regressors
                                ))
    dm_retro_df = pd.DataFrame(dm_retro, 
                                columns=["intercept", 
                                        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", 
                                        "cardiac_sin1", "cardiac_cos1", "cardiac_sin2", "cardiac_cos2",
                                        "resp_sin", "resp_cos"]
                                )
    dm_retro_df.index=frame_times

    #glm and residuals
    # first_level_var = FirstLevelModel(t_r=tr, drift_model='polynomial', drift_order=1, mask_img=mask_img, standardize=True,
    #                                     signal_scaling=False, smoothing_fwhm=4, minimize_memory=False, hrf_model=None)
    # first_level_var.fit(run_imgs=timeseries_img, design_matrices=dm_var_df)
    # residuals_var = masking.apply_mask(first_level_var.residuals[0], mask_img, dtype='f', smoothing_fwhm=None, ensure_finite=True)
    #print(residuals_var.shape)

    first_level_retro = FirstLevelModel(t_r=tr, drift_model='polynomial', drift_order=1, mask_img=mask_img, standardize=True,
                                        signal_scaling=False, smoothing_fwhm=4, minimize_memory=False, hrf_model=None)
    first_level_retro.fit(run_imgs=timeseries_img, design_matrices=dm_retro_df)
    residuals_retro = masking.apply_mask(first_level_retro.residuals[0], mask_img, dtype='f', smoothing_fwhm=None, ensure_finite=True)

    #save as hdf5 file
    # var_res_filename = session + "_" + run + "_var_residuals.hdf5"
    # var_res_path = os.path.join(output_data_path, subj, "timeseries", var_res_filename)
    # f = h5py.File(var_res_path, "w")
    # f.create_dataset("var_residuals", data = residuals_var)
    # f.close()
    # print("residuals saved to file:", var_res_path)

    retro_res_filename = session + "_" + run + "_retro_residuals.hdf5"
    retro_res_path = os.path.join(output_data_path, subj, "timeseries", retro_res_filename)
    f = h5py.File(retro_res_path, "w")
    f.create_dataset("retro_residuals", data = residuals_retro)
    f.close()
    print("residuals saved to file:", retro_res_path)

    print("")

    last_subj = subj



end = time.time()
print("")
print("Total time:", end - start)

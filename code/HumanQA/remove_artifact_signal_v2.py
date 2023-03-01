#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute and save residuals, contrast map and beta maps 
from GLM of RETROICOR regressors (X) onto voxel data (Y)
- 2 phase expansion cardiac
- 1 phase expansion respiration

"""

#imports
import numpy as np
import os
import pandas as pd
from copy import copy
from nilearn import image, masking
from nilearn.glm.first_level import FirstLevelModel


#data files
input_data_path = "/data/Amy/dynamicHR/HumanQA/data/BIDS/human-qa/derivatives/fmriprep/sub-06/"

sessions = ["ses-03", "ses-04", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", \
            "ses-10", "ses-12", "ses-13", "ses-14", "ses-15", "ses-17", "ses-18"]

rest_img_filename = "_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

task01_img_filenames = ["_task-randomagent_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                        "_task-randomagent_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
                        "_task-randomagent_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
				        "_task-shooterturn2_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
                        "_task-shooterturn2_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
				        "_task-randomagent_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
                        "_task-baityk7_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
				        "_task-baityk7_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
				        "_task-shootermirror1_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
				        "_task-shootermirror1_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
				        "_task-random_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
                        "_task-random_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                        "_task-shooterturn2_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                        "_task-diffusivebandit_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"] 

task02_img_filenames = ["_task-randomagent_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                        "_task-randomagent_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
                        "_task-randomagent_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
				        "_task-shooterturn2_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
                        "_task-shooterturn2_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
				        "_task-randomagent_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
                        "_task-baityk7_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
				        "_task-baityk7_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
				        "_task-shootermirror1_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
				        "_task-shootermirror1_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
				        "_task-random_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", 
                        "_task-random_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                        "_task-shooterturn2_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                        "_task-diffusivebandit_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"] 

rest_conf_filename = "_task-rest_desc-confounds_regressors.tsv"

task01_conf_filenames = ["_task-randomagent_run-1_desc-confounds_regressors.tsv",
                         "_task-randomagent_run-1_desc-confounds_regressors.tsv", 
                         "_task-randomagent_run-1_desc-confounds_regressors.tsv", 
				         "_task-shooterturn2_run-1_desc-confounds_regressors.tsv", 
                         "_task-shooterturn2_run-1_desc-confounds_regressors.tsv", 
				         "_task-randomagent_run-1_desc-confounds_regressors.tsv", 
                         "_task-baityk7_run-1_desc-confounds_regressors.tsv",
				         "_task-baityk7_run-1_desc-confounds_regressors.tsv", 
				         "_task-shootermirror1_run-1_desc-confounds_regressors.tsv", 
				         "_task-shootermirror1_run-1_desc-confounds_regressors.tsv", 
				         "_task-random_run-1_desc-confounds_regressors.tsv", 
                         "_task-random_run-1_desc-confounds_regressors.tsv",
                         "_task-shooterturn2_run-1_desc-confounds_regressors.tsv",
                         "_task-diffusivebandit_run-1_desc-confounds_regressors.tsv"] 

task02_conf_filenames = ["_task-randomagent_run-2_desc-confounds_regressors.tsv",
                         "_task-randomagent_run-2_desc-confounds_regressors.tsv", 
                         "_task-randomagent_run-2_desc-confounds_regressors.tsv", 
				         "_task-shooterturn2_run-2_desc-confounds_regressors.tsv", 
                         "_task-shooterturn2_run-2_desc-confounds_regressors.tsv", 
				         "_task-randomagent_run-2_desc-confounds_regressors.tsv", 
                         "_task-baityk7_run-2_desc-confounds_regressors.tsv",
				         "_task-baityk7_run-2_desc-confounds_regressors.tsv", 
				         "_task-shootermirror1_run-2_desc-confounds_regressors.tsv", 
				         "_task-shootermirror1_run-2_desc-confounds_regressors.tsv", 
				         "_task-random_run-2_desc-confounds_regressors.tsv", 
                         "_task-random_run-2_desc-confounds_regressors.tsv",
                         "_task-shooterturn2_run-2_desc-confounds_regressors.tsv",
                         "_task-diffusivebandit_run-2_desc-confounds_regressors.tsv"] 


img_files = []
conf_files = []
folder_names = []

# the below is definitely not the cleanest way - need to fix up
for (session, task01_img, task02_img, task01_conf, task02_conf) in \
		zip(sessions, task01_img_filenames, task02_img_filenames, \
            task01_conf_filenames, task02_conf_filenames):
	
    img_data_path = os.path.join(input_data_path, session, "func")
    
    rest_img_file = "sub-06_" + session + rest_img_filename 
    rest_img = os.path.join(img_data_path, rest_img_file)
    rest_conf_file = "sub-06_" + session + rest_conf_filename 
    rest_conf = os.path.join(img_data_path, rest_conf_file)
    
    task01_img_file = "sub-06_" + session + task01_img
    task01_img = os.path.join(img_data_path, task01_img_file)
    task01_conf_file = "sub-06_" + session + task01_conf
    task01_conf = os.path.join(img_data_path, task01_conf_file)
    
    task02_img_file = "sub-06_" + session + task02_img
    task02_img = os.path.join(img_data_path, task02_img_file)
    task02_conf_file = "sub-06_" + session + task02_conf
    task02_conf = os.path.join(img_data_path, task02_conf_file)

    img_files.extend((rest_img, task01_img, task02_img))
    conf_files.extend((rest_conf, task01_conf, task02_conf))
    folder_names.extend((os.path.join(session, "physio", "rest"), \
        os.path.join(session, "physio", "task01"), os.path.join(session, "physio", "task02")))
    

reg_data_path = "/data/Amy/dynamicHR/HumanQA/analysis/full_analysis/"

output_data_path = "/data/Amy/dynamicHR/HumanQA/analysis/final_analysis/"

output_folder_names = ["rest", "task01", "task02"]

#parameters
tr = 1.5
n_vols = 353
frame_times = np.arange(n_vols)*tr

#load gm mask
gm_filename = "sub-06_space-MNI152NLin2009cAsym_label-GM_mask_02.nii.gz"
gm_mask = image.load_img(output_data_path + gm_filename)


for (img_file, conf_file, run) in zip(img_files, conf_files, folder_names): 

    task = os.path.basename(run)
    session = run.split('/')[0]

    #load image
    print("loading image:", img_file)
    img = image.load_img(img_file)

    #load confounds
    print("loading confounds:", conf_file)
    conf_all_df = pd.read_csv(conf_file, sep="\t")

    #select confounds
    conf_vars = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y',
                    'rot_z', 'global_signal', 'csf', 'white_matter']

    conf_df = conf_all_df[conf_vars]

    cardiac_retro_reg_file = task + "_cardiac_RETROICOR_regressors.txt"
    print("loading cardiac RETORICOR regressors:", cardiac_retro_reg_file)
    cardiac_retro_reg_path = os.path.join(reg_data_path, run, cardiac_retro_reg_file)
    cardiac_retro_reg_data = np.loadtxt(cardiac_retro_reg_path)

    resp_retro_reg_file = task + "_resp_RETROICOR_regressors.txt"
    print("loading respiration RETORICOR regressors:", resp_retro_reg_file)
    resp_retro_reg_path = os.path.join(reg_data_path, run, resp_retro_reg_file)
    resp_retro_reg_data = np.loadtxt(resp_retro_reg_path)

    hv_reg_file = task + "_HV_regressors.txt"
    print("loading HR variability regressors:", hv_reg_file)
    hv_reg_path = os.path.join(reg_data_path, run, hv_reg_file)
    hv_reg_data = np.loadtxt(hv_reg_path)

    rv_reg_file = task + "_RV_regressors.txt"
    print("loading respiration variability regressors:", rv_reg_file)
    rv_reg_path = os.path.join(reg_data_path, run, rv_reg_file)
    rv_reg_data = np.loadtxt(rv_reg_path)

    print("data loaded")

    #set up design matrix
    dm_retroicor = np.column_stack((np.ones(len(frame_times)), #intercept
                                    conf_df.to_numpy(), 
                                    cardiac_retro_reg_data, #cardiac RETROICOR regressors
                                    resp_retro_reg_data #resp RETROICOR regressors
                                    ))
    dm_retroicor_df = pd.DataFrame(dm_retroicor, 
                                    columns=["intercept", 
                                            "trans_x", "trans_y", "trans_z", "rot_x", "rot_y",
                                            "rot_z", "global_signal", "csf", "white_matter",
                                            "cardiac_sin1", "cardiac_cos1", "cardiac_sin2", "cardiac_cos2",
                                            "resp_sin", "resp_cos"]
                                    )
    dm_retroicor_df.index=frame_times

    dm_var = np.column_stack((np.ones(len(frame_times)), #intercept
                            conf_df.to_numpy(), 
                            hv_reg_data, #HR variability regressors
                            rv_reg_data #resp variability regressors
                            ))
    dm_var_df = pd.DataFrame(dm_var, 
                            columns=["intercept", 
                                    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y",
                                    "rot_z", "global_signal", "csf", "white_matter",
                                    "hv", "rv"]
                            )
    dm_var_df.index=frame_times

    dm_retroicor_var = np.column_stack((np.ones(len(frame_times)), #intercept
                                        conf_df.to_numpy(), 
                                        cardiac_retro_reg_data, #cardiac RETROICOR regressors
                                        resp_retro_reg_data, #resp RETROICOR regressors
                                        hv_reg_data, #HR variability regressors
                                        rv_reg_data #resp variability regressors
                                        ))
    dm_retroicor_var_df = pd.DataFrame(dm_retroicor_var, 
                                      columns=["intercept", 
                                               "trans_x", "trans_y", "trans_z", "rot_x", "rot_y",
                                               "rot_z", "global_signal", "csf", "white_matter",
                                               "cardiac_sin1", "cardiac_cos1", "cardiac_sin2", "cardiac_cos2",
                                               "resp_sin", "resp_cos",
                                               "hv", "rv"]
                                      )
    dm_retroicor_var_df.index=frame_times

    #GLM
    first_level_retro = FirstLevelModel(t_r=tr, drift_model='polynomial', drift_order=1, mask_img=gm_mask, standardize=True,
                                        signal_scaling=False, smoothing_fwhm=4, minimize_memory=False, hrf_model=None) 
    #check if can get rid of smoothing
    
    first_level_var = copy(first_level_retro)
    first_level_retro_var = copy(first_level_retro)

    first_level_retro.fit(run_imgs=img, design_matrices=dm_retroicor_df) 
    first_level_var.fit(run_imgs=img, design_matrices=dm_var_df) 
    first_level_retro_var.fit(run_imgs=img, design_matrices=dm_retroicor_var_df) 

    #residuals
    residuals_retro = masking.apply_mask(first_level_retro.residuals[0], gm_mask, dtype='f', smoothing_fwhm=None, ensure_finite=True)
    residuals_var = masking.apply_mask(first_level_var.residuals[0], gm_mask, dtype='f', smoothing_fwhm=None, ensure_finite=True)
    residuals_retro_var = masking.apply_mask(first_level_retro_var.residuals[0], gm_mask, dtype='f', smoothing_fwhm=None, ensure_finite=True)


    #contrast map
    retroicor_map = first_level_retro.compute_contrast("cardiac_sin1+cardiac_cos1+cardiac_sin2+cardiac_cos2+resp_sin+resp_cos", 
                                                        stat_type="F", output_type="z_score")
    var_map = first_level_var.compute_contrast("hv+rv", stat_type="F", output_type="z_score")
    retroicor_var_map = first_level_retro_var.compute_contrast("cardiac_sin1+cardiac_cos1+cardiac_sin2+cardiac_cos2+resp_sin+resp_cos+hv+rv", 
                                                                stat_type="F", output_type="z_score")

    #beta maps - Q: is this correct input to compute_contrast?
    cardiac_sin1_beta = first_level_retro.compute_contrast("cardiac_sin1", output_type="effect_size")
    cardiac_sin2_beta = first_level_retro.compute_contrast("cardiac_sin2", output_type="effect_size")
    cardiac_cos1_beta = first_level_retro.compute_contrast("cardiac_cos1", output_type="effect_size")
    cardiac_cos2_beta = first_level_retro.compute_contrast("cardiac_cos2", output_type="effect_size")
    resp_sin_beta = first_level_retro.compute_contrast("resp_sin", output_type="effect_size")
    resp_cos_beta = first_level_retro.compute_contrast("resp_cos", output_type="effect_size")

    hv_beta = first_level_var.compute_contrast("hv", output_type="effect_size")
    rv_beta = first_level_var.compute_contrast("rv", output_type="effect_size")


    #save
    retro_res_outfile = task + "_RETROICOR_residuals.txt"
    retro_res_outpath = os.path.join(output_data_path, session, "time_series", retro_res_outfile)
    np.savetxt(retro_res_outpath, residuals_retro)
    print("residuals saved to file:", retro_res_outpath)

    var_res_outfile = task + "_var_residuals.txt"
    var_res_outpath = os.path.join(output_data_path, session, "time_series", var_res_outfile)
    np.savetxt(var_res_outpath, residuals_var)
    print("residuals saved to file:", var_res_outpath)

    retro_var_res_outfile = task + "_RETROICOR_var_residuals.txt"
    retro_var_res_outpath = os.path.join(output_data_path, session, "time_series", retro_var_res_outfile)
    np.savetxt(retro_var_res_outpath, residuals_retro_var)
    print("residuals saved to file:", retro_var_res_outpath)

    retro_con_outfile = task + "_RETROICOR_contrast_map.nii.gz"
    retro_con_outpath = os.path.join(output_data_path, session, "time_series", retro_con_outfile)
    retroicor_map.to_filename(retro_con_outpath)
    print("contrast map saved to file:", retro_con_outpath)

    var_con_outfile = task + "_var_contrast_map.nii.gz"
    var_con_outpath = os.path.join(output_data_path, session, "time_series", var_con_outfile)
    var_map.to_filename(var_con_outpath)
    print("contrast map saved to file:", retro_con_outpath)

    retro_var_con_outfile = task + "_RETROICOR_var_contrast_map.nii.gz"
    retro_var_con_outpath = os.path.join(output_data_path, session, "time_series", retro_var_con_outfile)
    retroicor_var_map.to_filename(retro_var_con_outpath)
    print("contrast map saved to file:", retro_var_con_outpath)

    beta_names = ['RETROICOR_cardiac_sin1', 'RETROICOR_cardiac_sin2', 'RETROICOR_cardiac_cos1', 'RETROICOR_cardiac_cos2', 
                    'RETROICOR_resp_sin', 'RETROICOR_resp_cos', 'hv', 'rv']
    beta_images = [cardiac_sin1_beta, cardiac_sin2_beta, cardiac_cos1_beta, cardiac_cos2_beta, 
                    resp_sin_beta, resp_cos_beta, hv_beta, rv_beta]

    for (beta, beta_img) in zip(beta_names, beta_images):
        beta_outfile = task + "_beta_map_" + beta + ".nii.gz"
        beta_outpath = os.path.join(output_data_path, session, "time_series", beta_outfile)
        beta_img.to_filename(beta_outpath)
        print("beta map save to file:", beta_outpath)





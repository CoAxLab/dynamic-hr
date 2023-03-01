#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Extract time series from nifti image:
- using fmriprep output in  MNI152NLin2009cAsym coordinate space 
- using nilearn to extract timeseries and regress confounds

"""

#imports
import os
import pandas as pd
from nilearn import image, masking, signal

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



output_data_path = "/data/Amy/dynamicHR/HumanQA/analysis/final_analysis/"

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

    # select confounds
    conf_vars = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y',
                    'rot_z', 'global_signal', 'csf', 'white_matter']

    conf_df = conf_all_df[conf_vars]

    #extract time series (for each voxel)
    print("beginning time series extraction...")
    time_series = masking.apply_mask(img, gm_mask, dtype='f', smoothing_fwhm=None, ensure_finite=True)
    #regressing out confounds
    cleaned_time_series = signal.clean(time_series, detrend=True, standardize='zscore', confounds=conf_df, standardize_confounds=True, filter=False, t_r=1.5)
    print("... extraction complete")
    cleaned_time_series_df = pd.DataFrame(cleaned_time_series)


    #save df
    out_file = task + "_raw_timeseries.csv"
    outpath = os.path.join(output_data_path, session, "time_series", out_file)
    cleaned_time_series_df.to_csv(outpath)
    print("saved to file:", outpath)
    print("")












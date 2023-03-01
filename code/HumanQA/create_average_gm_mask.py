#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Load standard space reference gm tissue probability image from fmriprep output
- Binarize to generate mask
"""

#imports
import numpy as np
from nilearn import image

#data files
gm_input_data_path = "/data/Amy/dynamicHR/HumanQA/data/BIDS/human-qa/derivatives/fmriprep/sub-06/anat/"
gm_prob_filename = "sub-06_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz"
func_input_data_path = "/data/Amy/dynamicHR/HumanQA/data/BIDS/human-qa/derivatives/fmriprep/sub-06/ses-03/func/"
func_filename = "sub-06_ses-03_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
output_data_path = "/data/Amy/dynamicHR/HumanQA/analysis/final_analysis/"

#load images
gm_prob = image.load_img(gm_input_data_path + gm_prob_filename)
func_img = image.load_img(func_input_data_path + func_filename)

#binarize
gm_mask_02 = image.math_img('img1 > 0.2', img1=gm_prob)

#resample to func resolution
gm_mask_02_resampled = image.resample_to_img(gm_mask_02, func_img, interpolation='nearest')

#save
output_file = "sub-06_space-MNI152NLin2009cAsym_label-GM_mask_02.nii.gz"
gm_mask_02_resampled.to_filename(output_data_path + output_file)
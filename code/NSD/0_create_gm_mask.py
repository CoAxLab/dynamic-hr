#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Load standard space HCP parcellation (using functional version bc same resolution as timeseries data)
- Binarize to generate mask for each subject
"""

#imports
import os
from nilearn import image

#data files
input_data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/data/nsddata/ppdata/"
output_data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"
parcellation_filename = "HCP_MMP1.nii.gz"

subjects = ["subj01", "subj02", "subj03", "subj04", \
            "subj05", "subj06", "subj07", "subj08"]

for subj in subjects:

    print(subj)

    #load parcellation image
    parcellation_path = os.path.join(input_data_path, subj, parcellation_filename)
    parcellation_img = image.load_img(parcellation_path)

    #convert to binary mask
    mask_img = image.math_img('img1 > 0', img1=parcellation_img)

    #save
    output_filename = subj + "_gm_mask.nii.gz"
    output_path = os.path.join(output_data_path, subj, output_filename)
    mask_img.to_filename(output_path)
    print("gm mask saved to file:", output_path)

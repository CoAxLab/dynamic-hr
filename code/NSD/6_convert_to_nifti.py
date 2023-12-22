#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert projected weights to nifti images (in subject space). Run for each subj 
individually.

Note: weights correspond to the training set of each test run, 
e.g. subj01, session21, run01 = the mean of projected weights from all other 
sessions and runs for this subj.
"""


#functions

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


#imports
import h5py
import os
import time
import pandas as pd
from nilearn import image, masking


data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"
subj = "subj08"
subj_file_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/" + subj + "/" + subj + "-session-run_list.txt"
subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
str_subj_list = subj_list.tolist()
#str_subj_list = ["subj01-session21-run01", "subj01-session21-run02", "subj01-session22-run03"]

#parameters
n_runs = len(str_subj_list)
time_shifts = [6, 5]

start = time.time()

gm_mask_path = os.path.join(data_path, subj, subj + "_gm_mask.nii.gz")
gm_mask = image.load_img(gm_mask_path)


for time_shift in time_shifts:

    print("")
    print("=========================")
    print("time shift: ", time_shift)

    time_snippet = "time" + str(time_shift)

    for i, entry in enumerate(str_subj_list):

        print("")
        print(entry)

        subj, session, run = entry.split('-')

        filename = session + "_" + run + "_var_projections_" + time_snippet + ".hdf5"
        filepath = os.path.join(data_path, subj, "models", "outputs", filename)
        [proj] = load_file(filepath, ['projections'], [False])
        
        proj_img = masking.unmask(proj, gm_mask, order='F')
        output_filename = session + "_" + run + "_var_projected_weights_" + time_snippet + ".nii.gz"
        output_path = os.path.join(data_path, subj, "models", "outputs", output_filename)
        proj_img.to_filename(output_path)
        print("Nifti image saved:", output_path)


    print("=========================")

end = time.time()
print("")
print("Total time:", end - start)

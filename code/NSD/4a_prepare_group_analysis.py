#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- create union gm mask across subjects (in MNI space)
- convert lasso betas to nifti, transform to MNI, remask with union mask --> save this hdf5 file
- do the same with varibility residuals
^ do this for all time shifts for all subjects!

NOTE: the transform to MNI portion needs to be done with the nsdcode utility, i.e. in nsdcode environment
- perhaps all of this can be done in that environment to keep it all in one script?


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


def save_file(output_path, output_filename, files, dataset_names):
    """
    save hdf5 file 

    inputs: output path to save to, output filename, output file
    """

    file_outpath = os.path.join(output_path, output_filename)

    f = h5py.File(file_outpath, "w")

    for (file, dataset_name) in zip(files, dataset_names):
        f.create_dataset(dataset_name, data = file)
    
    f.close()
    #print("saved to file:", file_outpath)


#imports
import h5py
import numpy as np
import os
import pandas as pd
import time
from nilearn import image, masking




#data files
data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"

subjects = [1,2,3,4,5,6,7,8]

#parameters
RANDOM_STATE = 2
#time_shifts = (np.arange(-10,3)) * (-1)
time_shifts = [6]
n_trs_total = 226

start = time.time()

#create union gm mask
# gm_masks = []

# for subject in subjects:
    
#     subj = "subj0" + str(subject)
    
#     print("")
#     print("=========================")
#     print("subject: ", subj)
    
#     gm_path = os.path.join(data_path, subj, subj + "_gm_mask-MNI-nearest.nii.gz")
#     gm_mask = image.load_img(gm_path)
#     gm_masks.append(gm_mask)
    
# union_gm_mask = masking.intersect_masks(gm_masks, threshold=0)
# union_path = os.path.join(data_path, "group", "union_gm_mask-MNI-nearest.nii.gz")
# union_gm_mask.to_filename(union_path)


#convert lasso betas and X data (residuals) to nifti, MNI, then union mask

#testing:
#subjects = [1]
#time_shifts = [10]

#---------------------------
#section 2: convert to nifti
#---------------------------

# for subject in subjects:
    
#     subj = "subj0" + str(subject)
    
#     print("")
#     print("=========================")
#     print("subject: ", subj)
    
#     subj_file_path = os.path.join(data_path, subj, subj + "-session-run_list.txt")
#     subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
#     str_subj_list = subj_list.tolist()
#     #str_subj_list = ["subj01-session21-run01", "subj01-session21-run02"]
    
#     #load subject space gm mask
#     gm_mask_path = os.path.join(data_path, subj, subj + "_gm_mask.nii.gz")
#     gm_mask = image.load_img(gm_mask_path)
    
    
#     for time_shift in time_shifts:
        
#         print("")
#         print("-------------------------")
#         print("time shift: ", time_shift)
        
#         time_snippet = "time" + str(time_shift)
#         n_trs = n_trs_total - abs(time_shift)
        
        
#         for i, entry in enumerate(str_subj_list):

#             print("")
#             print(entry)

#             subj, session, run = entry.split('-')
            
#             #convert to nifti and save
        
#             brain_file = session + "_" + run + "_var_residuals_" + time_snippet + ".hdf5" 
#             brain_path = os.path.join(data_path, subj, "models", "inputs", brain_file)
#             [X] = load_file(brain_path, ["time"+str(time_shift)], [False])
            
#             X_img = masking.unmask(X, gm_mask, order='F')
#             X_output_filename = session + "_" + run + "_var_residuals_" + time_snippet + ".nii.gz"
#             X_output_path = os.path.join(data_path, "group", subj, "models", "inputs", X_output_filename)
#             X_img.to_filename(X_output_path)
#             print("residuals saved as nifti:", X_output_path)
                
#             model_filename = session + "_" + run + "_var_trained_model_" + time_snippet + ".hdf5"
#             model_path = os.path.join(data_path, subj, "models", "outputs", model_filename)
#             dataset_names = ['betas', 'X_mean']
#             scalor_flags = [False, False]
#             [lasso_betas, X_mean] = load_file(model_path, dataset_names, scalor_flags)
        
#             beta_img = masking.unmask(lasso_betas, gm_mask, order='F')
#             beta_output_filename = session + "_" + run + "_var_betas_" + time_snippet + ".nii.gz"
#             beta_output_path = os.path.join(data_path, "group", subj, "models", "outputs", beta_output_filename)
#             beta_img.to_filename(beta_output_path)
#             print("betas saved as nifti:", beta_output_path)
            
#             X_mean_img = masking.unmask(X_mean, gm_mask, order='F')
#             X_mean_output_filename = session + "_" + run + "_var_X_mean_" + time_snippet + ".nii.gz"
#             X_mean_output_path = os.path.join(data_path, "group", subj, "models", "outputs", X_mean_output_filename)
#             X_mean_img.to_filename(X_mean_output_path)
#             print("X mean saved as nifti:", X_mean_output_path)
    
#         print("-------------------------")
        
#     print("=========================")


# end = time.time()
# print("")
# print("Total time:", end - start)


#=====================================================
#STOP HERE AND TRANSFORM TO MNI IN NSDCODE ENVIRONMENT
#conda activate nsdcode
#=====================================================

#---------------------------
#section 3: transform to MNI
#---------------------------

# import os
# import pandas as pd
# import time
# from nsdcode.nsd_mapdata import NSDmapdata
# from nsdcode.nsd_datalocation import nsd_datalocation

# data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"
# base_path = os.path.join('/media/amysentis/data2/Amy/dynamicHR/NSD/data/')

# # initiate NSDmapdata
# nsd = NSDmapdata(base_path)
# nsd_dir = nsd_datalocation(base_path=base_path)
# sourcespace = 'func1pt8'
# targetspace = 'MNI'
# interpmethod = 'nearest'

# input_files = ["var_residuals", "var_betas", "var_X_mean"]
# folders = ["inputs", "outputs", "outputs"]

# subjects = [1,2,3,4,5,6,7,8]
# #time_shifts = (np.arange(-10,3)) * (-1)

# #testing:
# #subjects = [1]
# #time_shifts = [10]

# start = time.time()

# for subject in subjects:
    
#     subjix = subject
#     subj = "subj0" + str(subject)
    
#     print("")
#     print("=========================")
#     print("subject: ", subj)
    
#     subj_file_path = os.path.join(data_path, subj, subj + "-session-run_list.txt")
#     subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
#     str_subj_list = subj_list.tolist()
#     #str_subj_list = ["subj01-session21-run01", "subj01-session21-run02"]

#     for time_shift in time_shifts:

#         print("")
#         print("-------------------------")
#         print("time shift: ", time_shift)

#         time_snippet = "time" + str(time_shift)

#         for i, entry in enumerate(str_subj_list):

#             print("")
#             print(entry)

#             subj, session, run = entry.split('-')
            
#             for (input_file, folder) in zip(input_files, folders):
                
#                 print(input_file)

#                 # input_filename = session + "_" + run + "_var_residuals_" + time_snippet + ".nii.gz"
#                 input_filename = session + "_" + run + "_" + input_file + "_" + time_snippet + ".nii.gz"
#                 # output_filename = session + "_" + run + "_var_residuals_MNI_" + time_snippet + ".nii.gz"
#                 output_filename = session + "_" + run + "_" + input_file + "_MNI_" + time_snippet + ".nii.gz"
#                 # output_path = os.path.join(data_path, "group", subj, "models", "inputs", output_filename)
#                 output_path = os.path.join(data_path, "group", subj, "models", folder, output_filename)
                
#                 # sourcedata = os.path.join(data_path, "group", subj, "models", "inputs", input_filename)
#                 sourcedata = os.path.join(data_path, "group", subj, "models", folder, input_filename)
    
#                 nsd.fit(
#                     subjix,
#                     sourcespace,
#                     targetspace,
#                     sourcedata,
#                     interptype=interpmethod,
#                     badval=0,
#                     outputfile=output_path)
            
#         print("-------------------------")
        
#     print("=========================")

# end = time.time()
# print("")
# print("Total time:", end - start)
    
    
    
#========================================
#STOP HERE AND RETURN TO BASE ENVIRONMENT
#conda deactivate
#========================================
 
#testing:
# subjects = [1]
# time_shifts = [10]

#---------------------------
# #section 4: extract matrices in common MNI space using union mask
#---------------------------

#load union MNI space gm mask
union_gm_mask_path = os.path.join(data_path, "group", "union_gm_mask-MNI-nearest.nii.gz")
union_gm_mask = image.load_img(union_gm_mask_path)

start = time.time()

for subject in subjects:
    
    subj = "subj0" + str(subject)
    
    print("")
    print("=========================")
    print("subject: ", subj)
    
    subj_file_path = os.path.join(data_path, subj, subj + "-session-run_list.txt")
    subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
    str_subj_list = subj_list.tolist()
    #str_subj_list = ["subj01-session21-run01", "subj01-session21-run02"]
    
    
    for time_shift in time_shifts:
        
        print("")
        print("-------------------------")
        print("time shift: ", time_shift)
        
        time_snippet = "time" + str(time_shift)
        n_trs = n_trs_total - abs(time_shift)
        
        
        for i, entry in enumerate(str_subj_list):

            print("")
            print(entry)

            subj, session, run = entry.split('-')
            
            #mask nifti to matrix and save
        
            brain_file = session + "_" + run + "_var_residuals_MNI_" + time_snippet + ".nii.gz" 
            brain_path = os.path.join(data_path, "group", subj, "models", "inputs", brain_file)
            X_img = image.load_img(brain_path)
            X_masked = masking.apply_mask(X_img, union_gm_mask, dtype='f', smoothing_fwhm=None, ensure_finite=True)
            X_output_filename = session + "_" + run + "_var_residuals_union_MNI_" + time_snippet + ".hdf5"
            X_output_path = os.path.join(data_path, "group", subj, "models", "inputs")
            save_file(X_output_path, X_output_filename, [X_masked], ['X'])
            print("masked union-MNI residuals saved as:", X_output_path, X_output_filename)
                
            beta_filename = session + "_" + run + "_var_betas_MNI_" + time_snippet + ".nii.gz"
            beta_path = os.path.join(data_path, "group", subj, "models", "outputs", beta_filename)
            beta_img = image.load_img(beta_path)
            beta_masked = masking.apply_mask(beta_img, union_gm_mask, dtype='f', smoothing_fwhm=None, ensure_finite=True)
            
            X_mean_filename = session + "_" + run + "_var_X_mean_MNI_" + time_snippet + ".nii.gz"
            X_mean_path = os.path.join(data_path, "group", subj, "models", "outputs", X_mean_filename)
            X_mean_img = image.load_img(X_mean_path)
            X_mean_masked = masking.apply_mask(X_mean_img, union_gm_mask, dtype='f', smoothing_fwhm=None, ensure_finite=True)
            
            model_output_filename = session + "_" + run + "_var_trained_model_union_MNI_" + time_snippet + ".hdf5"
            model_output_path = os.path.join(data_path, "group", subj, "models", "outputs")
            save_file(model_output_path, model_output_filename, [beta_masked, X_mean_masked], ['betas', 'X_mean'])
            print("masked union-MNI betas, X_mean saved as:", model_output_path, model_output_filename)

    
        print("-------------------------")
        
    print("=========================")



end = time.time()
print("")
print("Total time:", end - start)




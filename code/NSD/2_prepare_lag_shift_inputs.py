#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Prepare model input files for 13 different lag shifts for all noise models:
- up to 10 units backwards in time (inclusive)
- at time zero
- up to 2 units forwards in time (inclusive)

Save output as hdf5 files

"""

def load_file(input_path, dataset_name):
    """
    load hdf5 file 
    """

    f = h5py.File(input_path, 'r')
    data = f[dataset_name][:]
    f.close()

    return data



def save_file(output_path, output_filename, file, dataset_name):
    """
    save hdf5 file 

    inputs: output path to save to, output filename, output file
    """

    file_outpath = os.path.join(output_path, output_filename)

    f = h5py.File(file_outpath, "w")
    f.create_dataset(dataset_name, data = file)
    f.close()
    print("saved to file:", file_outpath)


#imports
import h5py
import numpy as np
import os
import pandas as pd

#load data
data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"

subj_file_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/subj267-session-run_list.txt"
subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
str_subj_list = subj_list.tolist()
#str_subj_list = ["subj01-session21-run02", "subj01-session22-run03", "subj02-session21-run01"]

data1_folder = "timeseries"
#data2_folder = "physio"


for entry in str_subj_list:

    print(entry)

    subj, session, run = entry.split('-')

    outpath = os.path.join(data_path, subj, "models", "inputs")

    # data1_filename = session + "_" + run + "_var_residuals.hdf5"
    # data1_path = os.path.join(data_path, subj, data1_folder, data1_filename)
    # data1 = load_file(data1_path, 'var_residuals')

    data1_filename = session + "_" + run + "_retro_residuals.hdf5"
    data1_path = os.path.join(data_path, subj, data1_folder, data1_filename)
    data1 = load_file(data1_path, 'retro_residuals')

    # data2_filename = session + "_" + run + "_downsampled_rr.hdf5"
    # data2_path = os.path.join(data_path, subj, data2_folder, data2_filename)
    # data2 = load_file(data2_path, 'downsampled_rr')

    #save full files as time0
    data1_filename_snippet = data1_filename.split('.')[0]
    new_data1_filename = data1_filename_snippet + "_time0.hdf5"
    save_file(outpath, new_data1_filename, data1, "time0")
    # data2_filename_snippet = data2_filename.split('.')[0]
    # new_data2_filename = data2_filename_snippet + "_time0.hdf5"
    # save_file(outpath, new_data2_filename, data2, "time0")

    #forwards time shift
    for i in np.arange(1,11):
        #new data1 np matrix
        new_data1 = data1[i:,:]
        # new data2 np vector
        #new_data2 = data2[:-i]
    
        #save
        new_data1_filename = data1_filename_snippet + "_time" + str(i) + ".hdf5"
        save_file(outpath, new_data1_filename, new_data1, "time"+str(i))
        #new_data2_filename = data2_filename_snippet + "_time" + str(i) + ".hdf5"
        #save_file(outpath, new_data2_filename, new_data2, "time"+str(i))

    #backwards time shift
    for j in [1,2]:
        #new data1 np matrix
        new_data1 = data1[:-j,:]
        # new data2 np vector
        #new_data2 = data2[j:]
        
        #save
        new_data1_filename = data1_filename_snippet + "_time-" + str(j) + ".hdf5"
        save_file(outpath, new_data1_filename, new_data1, "time-"+str(j))
        #new_data2_filename = data2_filename_snippet + "_time-" + str(j) + ".hdf5"
        #save_file(outpath, new_data2_filename, new_data2, "time-"+str(j))


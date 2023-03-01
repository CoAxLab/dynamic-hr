#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Prepare model input files for 13 different lag shifts for raw data:
- up to 10 units backwards in time (inclusive)
- at time zero
- up to 2 units forwards in time (inclusive)

Save output as txt files

"""

def save_file(output_path, output_filename, file):
    """
    save txt file 

    inputs: output path to save to, output filename, output file
    """

    file_outpath = os.path.join(output_path, output_filename)
    np.savetxt(file_outpath, file)
    print("saved to file:", file_outpath)


#imports
import numpy as np
import os
import pandas as pd

#load data
input_data1_path = "/data/Amy/dynamicHR/HumanQA/analysis/final_analysis/"
sessions = ["ses-03", "ses-04", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", \
            "ses-10", "ses-12", "ses-13", "ses-14", "ses-15", "ses-17", "ses-18"]
# = ["ses-10", "ses-12", "ses-13", "ses-14", "ses-15", "ses-17", "ses-18"]
data1_folder = "time_series"
input_data2_path = "/data/Amy/dynamicHR/HumanQA/analysis/full_analysis/"
data2_folder = "physio"
data2_filename = "downsampled_rr.txt"

folder_names = ["rest", "task01", "task02"]

output_data_path = "/data/Amy/dynamicHR/HumanQA/analysis/final_analysis/"


for session in sessions:

    outpath = os.path.join(output_data_path, session, "models", "inputs")

    for run in folder_names:

        # data1_raw_filename = run + "_raw_timeseries.csv"
        # data1_raw_path = os.path.join(input_data1_path, session, data1_folder, data1_raw_filename)
        # data1_raw_tmp = np.genfromtxt(data1_raw_path, delimiter=",")
        # data1_raw = data1_raw_tmp[1:,1:] #gets rid of header row and index column

        data2_path = os.path.join(input_data2_path, session, data2_folder, run, data2_filename)
        data2 = np.loadtxt(data2_path)

        #save full files as time0
        # data1_raw_filename_snippet = data1_raw_filename.split('.')[0]
        # new_data1_raw_filename = data1_raw_filename_snippet + "_time0.txt"
        data2_filename_snippet = data2_filename.split('.')[0]
        #new_data2_filename = run + "_" + data2_filename_snippet + "_time0.txt"
        #save_file(outpath, new_data1_raw_filename, data1_raw)
        #save_file(outpath, new_data2_filename, data2)

        #backwards time shift
        #for i in np.arange(1,11):
        for i in np.arange(6,11):
            #new data1_raw np matrix
            #new_data1_raw = data1_raw[i:,:]

            # new data2 np vector
            new_data2 = data2[:-i]
            
            #save
            #new_data1_raw_filename = data1_raw_filename_snippet + "_time-" + str(i) + ".txt"
            new_data2_filename = run + "_" + data2_filename_snippet + "_time-" + str(i) + ".txt"
            #save_file(outpath, new_data1_raw_filename, new_data1_raw)
            save_file(outpath, new_data2_filename, new_data2)

        #forwards time shift
        # for j in [1,2]:
        #     #new data1_raw np matrix
        #     new_data1_raw = data1_raw[:-j,:]

        #     # new data2 np vector
        #     #new_data2 = data2[j:]
            
        #     #save
        #     new_data1_raw_filename = data1_raw_filename_snippet + "_time" + str(j) + ".txt"
        #     #new_data2_filename = run + "_" + data2_filename_snippet + "_time" + str(j) + ".txt"
        #     save_file(outpath, new_data1_raw_filename, new_data1_raw)
        #     #save_file(outpath, new_data2_filename, new_data2)

        
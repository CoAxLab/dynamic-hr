#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate Haufe projections for each training set:
- load lasso weights
- load and center X data
- compute: projections = covariance(X) * weights
- for each modular training set, i.e. each run separately
- old: average projections for training sets corresponding to each test run
- save hdf5 projections

Note: need to run separate script for each subject
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
        f.create_dataset(dataset_name, data=file)

    f.close()
    print("saved to file:", file_outpath)



#imports
import h5py
import os
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler


data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"
subj = "subj08"
subj_file_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/" + subj + "/" + subj + "-session-run_list.txt"
subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
str_subj_list = subj_list.tolist()
#str_subj_list = ["subj01-session21-run01", "subj01-session21-run02", "subj01-session22-run03"]

#parameters
n_runs = len(str_subj_list)
time_shifts = [6, 5]
n_trs_total = 226

if "1" in subj:
    n_features = 102533 #subj01
elif "2" in subj:
    n_features = 103533 #subj02
elif "3" in subj:
    n_features = 106358 #subj03
elif "4" in subj:
    n_features = 97029 #subj04
elif "5" in subj:
    n_features = 90299 #subj05
elif "6" in subj:
    n_features = 115845 #subj06
elif "7" in subj:
    n_features = 86549 #subj07
else:
    n_features = 104392 #subj08

dof = n_features - 1

start = time.time()

for time_shift in time_shifts:

    print("")
    print("=========================")
    print("time shift: ", time_shift)

    n_trs = n_trs_total - abs(time_shift)
    time_snippet = "time" + str(time_shift)

    #projections = np.zeros((n_features, n_runs))

    for i, entry in enumerate(str_subj_list):

        print("")
        print(entry)

        subj, session, run = entry.split('-')

        print("loading data...")

        model_filename = session + "_" + run + "_var_trained_model_" + time_snippet + ".hdf5"
        model_path = os.path.join(data_path, subj, "models", "outputs", model_filename)
        [lasso_betas] = load_file(model_path, ['betas'], [False])

        brain_file = session + "_" + run + "_var_residuals_" + time_snippet + ".hdf5"
        brain_path = os.path.join(data_path, subj, "models", "inputs", brain_file)
        [X] = load_file(brain_path, [time_snippet], [False])
        normalize = StandardScaler(with_std=False)
        X_centered = normalize.fit_transform(X)

        print("calculating projections...")
        #cov_X = np.cov(X.T) # memory problem
        #cov_X = (1/dof) * (X.T @ X ) # <-- same memory problem
        projections = (1/dof) * (X_centered.T @ (X_centered @ lasso_betas)) #this ensures that always multiplying by a vector (instead of matrix)

        #save projections
        proj_file = session + "_" + run + "_var_projections_" + time_snippet + ".hdf5"
        output_path = os.path.join(data_path, subj, "models", "outputs")
        save_file(output_path, proj_file, [projections], ['projections'])

    
    print("=========================")

end = time.time()
print("")
print("Total time:", end - start)

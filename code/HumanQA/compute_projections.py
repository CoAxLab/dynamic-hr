#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate Haufe projections for each training set:
- load lasso weights
- load and center X data
- compute: projections = covariance(X) * weights
- average projections for training sets corresponding to each test run
- save pickled projections

"""


#functions

def load_pickle(path):
    
    f = open(path, 'rb')
    data = pickle.load(f) 
    f.close()

    return data


def save_pickle(output_path, output_filename, data):
    """
    save data to HDF5 file (this needs to be implemented/changed)

    inputs: output path to save to, output filename, list of output data
    """

    file_outpath = os.path.join(output_path, output_filename)
    f = open(file_outpath, 'wb')
    pickle.dump(data,f)
    f.close()
    print("data saved as", output_filename, "in: ", output_path)



#imports
import gc
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler


data_path = "/data/Amy/dynamicHR/HumanQA/analysis/final_analysis/"
sessions = ["ses-03", "ses-04", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", \
            "ses-10", "ses-12", "ses-13", "ses-14", "ses-15", "ses-17", "ses-18"]
#sessions = ["ses-03"]
folder_names = ["rest", "task01", "task02"]


runs = []
for session in sessions:
    for folder in folder_names:
        runs.append(session + "_" + folder)

#parameters
n_runs = len(sessions) * len(folder_names)
time_shifts = [-6, -5]
n_trs_total = 353
n_features = 152498
dof = n_features - 1


for time_shift in time_shifts:

    print("")
    print("=========================")
    print("time shift: ", time_shift)
    print("")

    n_trs = n_trs_total - abs(time_shift)

    projections = np.zeros((n_features, n_runs))

    for i, run in enumerate(runs):

        print("")
        print("computing individual haufe projections: run ", i, ", ", run)
        print("")
        session, task = run.split('_')

        print("loading data...")

        time_snippet = "time" + str(time_shift)

        model_filename = task + "_var_mask02_trained_lasso_cj_" + time_snippet + ".pckl"
        print(model_filename)
        model_path = os.path.join(data_path, session, "models", "outputs", model_filename)

        lasso_alpha, lasso_betas, lasso_intercept_tmp = load_pickle(model_path)

        brain_file = task + "_var_residuals_time" + str(time_shift) + ".txt"
        print(brain_file)
        brain_path = os.path.join(data_path, session, "models", "inputs", brain_file)

        X = np.loadtxt(brain_path)
        normalize = StandardScaler(with_std=False)
        X_centered = normalize.fit_transform(X)

        print("... loaded")
        print("")

        print("calculating...")
        #cov_X = np.cov(X.T) # memory problem
        #cov_X = (1/dof) * (X.T @ X ) # <-- same memory problem
        projections[:,i] = (1/dof) * (X_centered.T @ (X_centered @ lasso_betas)) #this ensures that always multiplying by a vector (instead of matrix)
        print("...done")
        print("")


    
    for i, run in enumerate(runs):

        print("")
        print("averaging projections for all except: run ", i, ", ", run)
        print("")

        session, task = run.split('_')

        indices = np.arange(n_runs) #len(runs)
        indices1 = np.delete(indices,i)
        indices1_list = indices1.tolist()
        #print("indices1_list", indices1_list)
        
        proj_sum = projections[:,indices1_list].sum(axis=1)
        proj_mean = proj_sum / (n_runs-1)
        #print("projections", projections[0,:])
        #print("proj_sum", proj_sum[0])
        #print("proj_mean", proj_mean[0])
        
        print("...done")
        print("")

        #save projections
        proj_file = task + "_var_mask02_modular_projections_cj_" + time_snippet + ".pckl"
        output_path = os.path.join(data_path, session, "models", "outputs")
        save_pickle(output_path, proj_file, [proj_mean])
        #print("saving")
        #print(output_path, proj_file)


    
    print("=========================")
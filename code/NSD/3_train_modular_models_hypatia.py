#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load input data and train individual LASSO-PCR model for each run, for each time shift
- (for now) not using next 5-fold cv to select lambda parameter for LASSO --> commented out
- saves model weights (LASSO coefficients) from each run and clears memory before starting the next run

Note: right now, this code will only work if all runs for a given model are of the same length

"""


#functions

def load_file(input_path, dataset_name):
    """
    load hdf5 file 
    """

    f = h5py.File(input_path, 'r')
    data = f[dataset_name][:]
    f.close()

    return data


def train_lasso(X_train, y_train):
    """
    Pipeline: normalize, PCR, LASSO
    Run 5-fold cv to select optimal LASSO parameters (on training set)
    Fit LASSO model with optimal parameters using training set
    Get predictions on test set

    Returns: test set predictions, optimal alpha value for LASSO models
    """

    # javi's class w scale only - saving mean for each run
    lasso_cv = LassoPCR(scale=False, cv=None, n_alphas=100, lasso_kws={'max_iter':1e5, 'random_state':RANDOM_STATE}) #default cv=5
    lasso_cv.fit(X_train, y_train.ravel())
    opt_alpha = lasso_cv.alpha_
    #save these to get see variation across runs

    # fit optimal model
    normalize = StandardScaler(with_std=False)
    pca = PCA()
    lasso_opt = Lasso(alpha=opt_alpha, normalize=False, max_iter=1e5, random_state=RANDOM_STATE)
    pipeline_list_opt = [('normalize', normalize), ('pca', pca), ('lasso_opt', lasso_opt)]
    pipe_opt = Pipeline(pipeline_list_opt)
    pipe_opt.fit(X_train, y_train.ravel()) 
    X_mean = pipe_opt.named_steps['normalize'].mean_
    pca_u, pca_s, pca_v = pipe_opt.named_steps['pca']._fit(X_train)
    lasso_coef = pipe_opt.named_steps['lasso_opt'].coef_ #lasso cost function weight
    lasso_betas = lasso_coef.dot(pca_v)
    lasso_intercept = pipe_opt.named_steps['lasso_opt'].intercept_

    return opt_alpha, X_mean, lasso_betas, lasso_intercept


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
    print("saved to file:", file_outpath)


#imports
import h5py
import numpy as np
import os
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#for javi's class
import sys
sys.path.append("/data/Amy/my-scikit-tools/")
from my_sklearn_tools.pca_regressors import LassoPCR

#data files
data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"
subj_file_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/data/subj01-session-run_list.txt"
subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
str_subj_list = subj_list.tolist()
#str_subj_list = ["subj01-session21-run01", "subj01-session21-run02", "subj01-session22-run03"]

#parameters
RANDOM_STATE = 2
n_runs = len(str_subj_list)
time_shifts = (np.arange(-10,3)) * (-1)
#time_shifts = [10]
n_trs_total = 226
n_features = 102533


for time_shift in time_shifts:

    print("")
    print("=========================")
    print("time shift: ", time_shift)
    print("")

    n_trs = n_trs_total - abs(time_shift)

    for entry in str_subj_list:

        print(entry)

        subj, session, run = entry.split('-')

        #load data
        print("loading data...")

        hr_file = session + "_" + run + "_downsampled_rr_time" + str(time_shift) + ".hdf5"
        brain_file = session + "_" + run + "_var_residuals_time" + str(time_shift) + ".hdf5" 

        hr_path = os.path.join(data_path, subj, "models", "inputs", hr_file)
        brain_path = os.path.join(data_path, subj, "models", "inputs", brain_file)

        X = load_file(brain_path, "time"+str(time_shift))
        y = load_file(hr_path, "time"+str(time_shift))
        
        #train model
        print("training model...")
        lasso_alpha, X_mean, lasso_betas, lasso_intercept = train_lasso(X, y)

        #save model
        output_path = os.path.join(data_path, subj, "models", "outputs")
        out_snippet = "time" + str(time_shift)
        output_filename = session + "_" + run + "_var_trained_lasso_" + out_snippet + ".hdf5"
        lasso_files = [lasso_alpha, lasso_betas, lasso_intercept]
        lasso_dataset_names = ['alpha', 'betas', 'intercept']
        save_file(output_path, output_filename, lasso_files, lasso_dataset_names)

        output_filename = session + "_" + run + "_var_X_mean_" + out_snippet + ".hdf5"
        X_files = [X_mean]
        X_dataset_names = ['X_mean']
        save_file(output_path, output_filename, X_files, X_dataset_names)
        print("")
            

    
    print("=========================")

    









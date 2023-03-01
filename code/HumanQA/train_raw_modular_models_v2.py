#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load input data and train individual LASSO-PCR model for each run, for each time shift
- (for now) not using next 5-fold cv to select lambda parameter for LASSO --> commented out
- saves model weights (LASSO coefficients) from each run and clears memory before starting the next run

Note: right now, this code will only work if all runs for a given model are of the same length

"""


#functions

def load_data(y_file, X_file):
    """
    load X and y data from file

    Returns: x matrix (numpy format), y vector (numpy format)
    """

    y = np.loadtxt(y_file)
    X = np.loadtxt(X_file)

    return X, y



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
import numpy as np
import os
import pickle

from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#for javi's class
import sys
sys.path.append("/data/Amy/my-scikit-tools/")
from my_sklearn_tools.pca_regressors import LassoPCR

#data files
input_data_path = "/data/Amy/dynamicHR/HumanQA/analysis/full_analysis/"
sessions = ["ses-03", "ses-04", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", \
            "ses-10", "ses-12", "ses-13", "ses-14", "ses-15", "ses-17", "ses-18"]
folder_names = ["rest", "task01", "task02"]

output_data_path = "/data/Amy/dynamicHR/HumanQA/analysis/final_analysis/"

#parameters
RANDOM_STATE = 2
n_runs = len(sessions) * len(folder_names)
n_folds = n_runs
time_shifts = np.arange(-10,3)
n_trs_total = 353
n_features = 152498 


for time_shift in time_shifts:

    print("")
    print("=========================")
    print("time shift: ", time_shift)
    print("")

    n_trs = n_trs_total - abs(time_shift)

    for session in sessions:

        print("session: ", session)

        for run in folder_names:

            print("run: ", run)

            #load data
            print("loading data...")

            hr_file = run + "_downsampled_rr_time" + str(time_shift) + ".txt"
            brain_file = run + "_raw_timeseries_time" + str(time_shift) + ".txt"

            hr_path = os.path.join(input_data_path, session, "models", "inputs", hr_file)
            brain_path = os.path.join(input_data_path, session, "models", "inputs", brain_file)

            X, y = load_data(hr_path, brain_path)
            print("data loaded")
            
            #train model
            print("training model...")
            lasso_alpha, X_mean, lasso_betas, lasso_intercept = train_lasso(X, y)
            print("model trained")

            #save model
            print("saving model data...")
            output_path = os.path.join(output_data_path, session, "models", "outputs", "raw")
            out_snippet = "time" + str(time_shift)
            output_filename = run + "_raw_mask02_trained_lasso_cj_" + out_snippet + ".pckl"
            save_pickle(output_path, output_filename, [lasso_alpha, lasso_betas, lasso_intercept])

            output_filename = run + "_raw_mask02_X_mean_" + out_snippet + ".pckl"
            save_pickle(output_path, output_filename, X_mean)

    
    print("=========================")

    









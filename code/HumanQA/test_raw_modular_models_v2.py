#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load input data and trained individual LASSO-PCR model for each run, for each time shift
- averages model weights from the relevant trained models to use as model weights for testing
- implement leave-one-run out testing (could also use 5/10 fold cv)

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


def load_pickle(path):
    
    f = open(path, 'rb')
    data = pickle.load(f) 
    f.close()

    return data


def test_lasso(X, y, lasso_betas, lasso_intercept):
    """
    Pipeline: normalize, PCR, LASSO
    Run 5-fold cv to select optimal LASSO parameters (on training set)
    Fit LASSO model with optimal parameters using training set
    Get predictions on test set

    Returns: test set predictions, optimal alpha value for LASSO models
    """

    # fixed normalization step - according to train data
    # same code for scale only step - according to train data
    lasso = Lasso(fit_intercept=False, normalize=False, max_iter=1e5, random_state=RANDOM_STATE)
    pipeline_list = [('lasso', lasso)]
    pipe = Pipeline(pipeline_list)
    pipe.named_steps['lasso'].coef_ = lasso_betas
    pipe.named_steps['lasso'].intercept_ = lasso_intercept

    y_predicted_test = pipe.predict(X)  

    return y_predicted_test


def evaluate_model(y, y_pred):
    """
    Evaluate model predictions using:
    - Pearson correlation coefficient, r
    - Coefficient of determination, R-squared
    - RMSE
    - add in: AIC/BIC?

    Returns: r, p, R-squared, RMSE
    """

    r, p = pearsonr(y, y_pred)

    r2 = r2_score(y, y_pred)

    n = y.shape[0]
    mse = (1/n)*np.sum(np.square(y-y_pred))
    rmse = np.sqrt(mse)

    return r, p, r2, rmse


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

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr


#data files
input_data_path = "/data/Amy/dynamicHR/HumanQA/analysis/full_analysis/"
sessions = ["ses-03", "ses-04", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", \
            "ses-10", "ses-12", "ses-13", "ses-14", "ses-15", "ses-17", "ses-18"]
folder_names = ["rest", "task01", "task02"]

output_data_path = "/data/Amy/dynamicHR/HumanQA/analysis/final_analysis/"

runs = []
for session in sessions:
    for folder in folder_names:
        runs.append(session + "_" + folder)

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

    corrs = np.zeros((n_runs,))
    corrs_p = np.zeros((n_runs,))
    r2s = np.zeros((n_runs,))
    rmses = np.zeros((n_runs,))

    for i, run in enumerate(runs):

        print("")
        print("run: ", run)
        session, task = run.split('_')

        #load test data
        print("loading data...")

        hr_file = task + "_downsampled_rr_time" + str(time_shift) + ".txt"
        brain_file = task + "_raw_timeseries_time" + str(time_shift) + ".txt"

        hr_path = os.path.join(input_data_path, session, "models", "inputs", hr_file)
        brain_path = os.path.join(input_data_path, session, "models", "inputs", brain_file)

        X, y = load_data(hr_path, brain_path)

        train_runs = runs.copy()
        train_runs.remove(run)

        lasso_betas = np.zeros(X.shape[1]) #or zeros_like(X[0,:])
        X_means = np.zeros(X.shape[1])
        X_stds = np.zeros(X.shape[1])
        lasso_intercept = 0

        for train_run in train_runs:

            session1, task1 = train_run.split('_')

            time_snippet = "time" + str(time_shift)
            model_filename = task1 + "_raw_mask02_trained_lasso_cj_" + time_snippet + ".pckl"
            model_path = os.path.join(output_data_path, session1, "models", "outputs", "raw", model_filename)

            lasso_alpha, lasso_betas_tmp, lasso_intercept_tmp = load_pickle(model_path)
            lasso_betas += lasso_betas_tmp
            lasso_intercept += lasso_intercept_tmp

            #normalization_filename = task1 + "_raw_X_mean_std_" + time_snippet + "_v2.pckl"
            scale_filename = task1 + "_raw_mask02_X_mean_" + time_snippet + ".pckl"
            #normalization_path = os.path.join(input_data_path, session1, "models", "outputs", "raw", normalization_filename)
            scale_path = os.path.join(output_data_path, session1, "models", "outputs", "raw", scale_filename)

            #X_mean_tmp, X_std_tmp = load_pickle(normalization_path)
            X_mean_tmp = load_pickle(scale_path)
            X_means += X_mean_tmp
            #X_stds += X_std_tmp
        
        lasso_betas = lasso_betas / len(train_runs)
        lasso_intercept = lasso_intercept / len(train_runs)

        X_means = X_means / len(train_runs)
        #X_stds = X_stds / len(train_runs)

        #X_normalized = (X - X_means) / X_stds
        X_centered = X - X_means
        
        print("testing model...")
        y_predicted = test_lasso(X_centered, y, lasso_betas, lasso_intercept)

        r, p, r2, rmse = evaluate_model(y, y_predicted)
        corrs[i] = r
        corrs_p[i] = p
        r2s[i] = r2
        rmses[i] = rmse
        print("Pearson r: ", r, "p-value: ", p)
        print("R-squared: ", r2)
        print("RMSE: ", rmse)

        #save y observed and y predicted
        y_pred_file = task + "_raw_mask02_modular_y_predicted_test_cj_" + time_snippet + ".pckl"
        output_path = os.path.join(output_data_path, session, "models", "outputs", "raw")
        save_pickle(output_path, y_pred_file, [y, y_predicted])
            
    #save cv metrics
    output_path = os.path.join(output_data_path, "results", "raw", time_snippet)
    cv_file = 'raw_mask02_modular_cj_cv_r_p_r2_rmse.pckl'
    save_pickle(output_path, cv_file, [corrs, corrs_p, r2s, rmses])

    
    print("=========================")

    









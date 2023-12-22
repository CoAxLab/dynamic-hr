#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load input data and trained individual LASSO-PCR model for each run, for each time shift
- averages model weights from the relevant trained models to use as model weights for testing
- implement leave-one-run out testing (could also use 5/10 fold cv)

Note: right now, this code will only work if all runs for a given model are of the same length

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


def test_lasso(X, lasso_betas, lasso_intercept):
    """
    Pipeline: normalize, PCR, LASSO
    Run 5-fold cv to select optimal LASSO parameters (on training set)
    Fit LASSO model with optimal parameters using training set
    Get predictions on test set

    Returns: test set predictions, optimal alpha value for LASSO models
    """
    
    
    # fixed normalization step - according to train data
    # same code for scale only step - according to train data
    #lasso = Lasso(fit_intercept=False, normalize=False, max_iter=100000, random_state=RANDOM_STATE)
    lasso = Lasso(fit_intercept=False, max_iter=100000, random_state=RANDOM_STATE)
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
import time

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr


#data files
data_path1 = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"
data_path2 = "/home/amysentis/dynamicHR/NSD/analysis/"
#subj_file_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/subj02/subj02-session-run_list.txt"
#subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
#str_subj_list = subj_list.tolist()
#str_subj_list = ["subj02-session21-run01", "subj02-session21-run02", "subj02-session22-run03"]

#subjects = [1,2,3,4,5,6,7,8]
subjects = [2,6,7]

#parameters
RANDOM_STATE = 2

time_shifts = (np.arange(-10,3)) * (-1)
#time_shifts = [-1, -2]
n_trs_total = 226
#n_features = 102533

start = time.time()

for subject in subjects:
    
    subj = "subj0" + str(subject)
    subj_file_path = os.path.join(data_path1, subj, subj + "-session-run_list.txt")
    subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
    str_subj_list = subj_list.tolist()

    n_runs = len(str_subj_list)

    print("")
    print("=========================")
    print("subject: ", subj)
    print("")

    for time_shift in time_shifts:

        print("")
        print("=========================")
        print("time shift: ", time_shift)
        print("")

        time_snippet = "time" + str(time_shift)
        n_trs = n_trs_total - abs(time_shift)

        corrs = np.zeros((n_runs,))
        corrs_p = np.zeros((n_runs,))
        r2s = np.zeros((n_runs,))
        rmses = np.zeros((n_runs,))

        #for i, run in enumerate(runs):
        for i, entry in enumerate(str_subj_list):
            #subj01-session21-run01
            #subj01-session21-run02
            print(entry)

            subj, session, run = entry.split('-')

            #load test data
            print("loading data...")

            hr_file = session + "_" + run + "_downsampled_rr_time" + str(time_shift) + ".hdf5"
            #brain_file = session + "_" + run + "_var_residuals_time" + str(time_shift) + ".hdf5" 
            brain_file = session + "_" + run + "_retro_residuals_time" + str(time_shift) + ".hdf5" 

            hr_path = os.path.join(data_path1, subj, "models", "inputs", hr_file)
            brain_path = os.path.join(data_path2, subj, "models", "inputs", brain_file)

            [X] = load_file(brain_path, ["time"+str(time_shift)], [False])
            [y] = load_file(hr_path, ["time"+str(time_shift)], [False])

            train_entries = str_subj_list.copy()
            train_entries.remove(entry)

            lasso_betas = np.zeros(X.shape[1]) 
            X_means = np.zeros(X.shape[1])
            X_stds = np.zeros(X.shape[1])
            lasso_intercept = 0

            for train_entry in train_entries:

                #print(train_entry)

                subj, session1, run1 = train_entry.split('-')

                #time_snippet = "time" + str(time_shift)

                #model_filename = session1 + "_" + run1 + "_var_trained_model_" + time_snippet + ".hdf5"
                model_filename = session1 + "_" + run1 + "_retro_trained_model_" + time_snippet + ".hdf5"
                model_path = os.path.join(data_path2, subj, "models", "outputs", model_filename)
                #print(model_path)
                dataset_names = ['alpha', 'betas', 'intercept', 'X_mean']
                scalor_flags = [True, False, True, False]
                [lasso_alpha, lasso_betas_tmp, lasso_intercept_tmp, X_mean_tmp] = load_file(model_path, dataset_names, scalor_flags)
                #model_filename = session1 + "_" + run1 + "_var_trained_lasso_" + time_snippet + ".hdf5"
                #model_path = os.path.join(data_path, subj, "models", "outputs", model_filename)

                #lasso_dataset_names = ['alpha', 'betas', 'intercept']
                #lasso_flags = [True, False, True]
                #[lasso_alpha, lasso_betas_tmp, lasso_intercept_tmp] = load_file(model_path, lasso_dataset_names, lasso_flags)
                lasso_betas += lasso_betas_tmp
                lasso_intercept += lasso_intercept_tmp

                #scale_filename = session1 + "_" + run1 + "_var_X_mean_" + time_snippet + ".hdf5"
                #scale_path = os.path.join(data_path, subj, "models", "outputs", scale_filename)

                #X_mean_dataset_names = ['X_mean']
                #X_mean_flags = [False]
                #[X_mean_tmp] = load_file(scale_path, X_mean_dataset_names, X_mean_flags)
                X_means += X_mean_tmp
            
            lasso_betas = lasso_betas / len(train_entries)
            lasso_intercept = lasso_intercept / len(train_entries)

            X_means = X_means / len(train_entries)

            X_centered = X - X_means
            
            print("testing model...")
            y_predicted = test_lasso(X_centered, lasso_betas, lasso_intercept)

            r, p, r2, rmse = evaluate_model(y, y_predicted)
            corrs[i] = r
            corrs_p[i] = p
            r2s[i] = r2
            rmses[i] = rmse
            print("Pearson r: ", r, "p-value: ", p)
            print("R-squared: ", r2)
            print("RMSE: ", rmse)

            #save y observed and y predicted
            #y_pred_file = session + "_" + run + "_var_y_predicted_test_" + time_snippet + "_v2.hdf5"
            y_pred_file = session + "_" + run + "_retro_y_predicted_test_" + time_snippet + ".hdf5"
            output_path = os.path.join(data_path2, subj, "models", "outputs")
            save_file(output_path, y_pred_file, [y, y_predicted], ['y', 'y_predicted'])
            print("")
                
        #save cv metrics
        output_path = os.path.join(data_path2, subj, "models", "outputs")
        #cv_file = "var_cv_r_p_r2_rmse_" + time_snippet + "_v2.hdf5"
        cv_file = "retro_cv_r_p_r2_rmse_" + time_snippet + ".hdf5"
        save_file(output_path, cv_file, [corrs, corrs_p, r2s, rmses], ['corr', 'p', 'r2', 'rmse'])
        
        print("=========================")

    print("")
    print("=========================")

    
end = time.time()
print("")
print("Total time:", end - start)



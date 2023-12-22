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
data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"
subjects = [1,2,3,4,5,6,7,8]
#subjects = [1]

output_data_path = "/home/amysentis/dynamicHR/NSD/analysis/"

#parameters
RANDOM_STATE = 2
time_shift = 6
#n_trs_total = 226
time_snippet = "time" + str(time_shift)
#n_trs = n_trs_total - abs(time_shift)
n_features = 1271436

start = time.time()

for subject in subjects:
    
    subj = "subj0" + str(subject)
    
    print("")
    print("=========================")
    print("subject:", subj)
    
    subj_file_path = os.path.join(data_path, subj, subj + "-session-run_list.txt")
    subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
    str_subj_list = subj_list.tolist()
    #str_subj_list = ["subj01-session21-run01", "subj01-session21-run02", "subj01-session22-run03"]
    
    n_runs = len(str_subj_list)
    corrs = np.zeros((n_runs,))
    corrs_p = np.zeros((n_runs,))
    r2s = np.zeros((n_runs,))
    rmses = np.zeros((n_runs,))
    
    #for i, run in enumerate(runs):
    for i, entry in enumerate(str_subj_list):
    
        print(entry)
    
        subj, session, run = entry.split('-')
    
        #load test data
        print("loading test data...")
    
        hr_file = session + "_" + run + "_downsampled_rr_" + time_snippet + ".hdf5"
        brain_file = session + "_" + run + "_var_residuals_union_MNI_" + time_snippet + ".hdf5" 
    
        hr_path = os.path.join(data_path, subj, "models", "inputs", hr_file)
        brain_path = os.path.join(data_path, "group", subj, "models", "inputs", brain_file)

        #print(hr_path)
        #print(brain_path)
    
        [X] = load_file(brain_path, ['X'], [False])
        [y] = load_file(hr_path, [time_snippet], [False])
    
        train_entries = str_subj_list.copy()
        train_entries.remove(entry)
    
        lasso_betas = np.zeros(X.shape[1]) 
        X_means = np.zeros(X.shape[1])
        lasso_intercept = 0

        print("loading train data...")
    
        for train_entry in train_entries:
    
            print(train_entry)
    
            subj, session1, run1 = train_entry.split('-')
            
            model_filename = session1 + "_" + run1 + "_var_trained_model_union_MNI_" + time_snippet + ".hdf5"
            model_path = os.path.join(data_path, "group", subj, "models", "outputs", model_filename)
            #print(model_path)
            dataset_names = ['betas', 'X_mean']
            scalor_flags = [False, False]
            [lasso_betas_tmp, X_mean_tmp] = load_file(model_path, dataset_names, scalor_flags)
            intercept_filename = session1 + "_" + run1 + "_var_trained_model_" + time_snippet + ".hdf5"
            intercept_path = os.path.join(data_path, subj, "models", "outputs", intercept_filename)
            #print(intercept_path)
            [lasso_intercept_tmp] = load_file(intercept_path, ['intercept'], [True])
            
            lasso_betas += lasso_betas_tmp
            #print(lasso_betas)
            lasso_intercept += lasso_intercept_tmp
            #print(lasso_intercept)
            X_means += X_mean_tmp
            #print(X_means)
        
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
        # print("Pearson r: ", r, "p-value: ", p)
        # print("R-squared: ", r2)
        # print("RMSE: ", rmse)
    
        #save y observed and y predicted
        y_pred_file = session + "-" + run + "_var_y_predicted_test_MNI_" + time_snippet + ".hdf5"
        output_path = os.path.join(output_data_path, subj, "models", "outputs")
        #print(output_path, y_pred_file)
        save_file(output_path, y_pred_file, [y, y_predicted], ['y', 'y_predicted'])
        print("")
            
    #save cv metrics
    output_path = os.path.join(output_data_path, subj, "models", "outputs")
    cv_file = "var_cv_r_p_r2_rmse_MNI_" + time_snippet + ".hdf5"
    #print(output_path, cv_file)
    save_file(output_path, cv_file, [corrs, corrs_p, r2s, rmses], ['corr', 'p', 'r2', 'rmse'])
    
    print("=========================")

    
end = time.time()
print("")
print("Total time:", end - start)



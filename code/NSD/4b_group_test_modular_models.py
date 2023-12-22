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

#parameters
RANDOM_STATE = 2
#time_shifts = (np.arange(-10,3)) * (-1)
n_trs_total = 226
time_shift = 6
time_snippet = "time" + str(time_shift)
n_trs = n_trs_total - abs(time_shift)
n_features = 1271436


start = time.time()

#testing
#subjects = [1,2]

for train_subject in subjects:
    
    train_subj = "subj0" + str(train_subject)
    
    print("")
    print("=========================")
    print("train subject: ", train_subj)
    
    train_subj_file_path = os.path.join(data_path, train_subj, train_subj + "-session-run_list.txt")
    #train_subj_file_path = os.path.join(data_path, train_subj, train_subj + "-session-run_list_test0509.txt")
    train_subj_list = pd.read_csv(train_subj_file_path, header=None).squeeze("columns")
    train_str_subj_list = train_subj_list.tolist()
    #train_str_subj_list = ["subj01-session21-run01", "subj01-session21-run02"]
    
    #load train data
    #print("load train data")
    lasso_betas = np.zeros(n_features) 
    X_means = np.zeros(n_features)
    lasso_intercept = 0
    
    for train_entry in train_str_subj_list:

        print(train_entry)

        train_subj, train_session, train_run = train_entry.split('-')

        model_filename = train_session + "_" + train_run + "_var_trained_model_union_MNI_" + time_snippet + ".hdf5"
        model_path = os.path.join(data_path, "group", train_subj, "models", "outputs", model_filename)
        #print(model_path)
        dataset_names = ['betas', 'X_mean']
        scalor_flags = [False, False]
        [lasso_betas_tmp, X_mean_tmp] = load_file(model_path, dataset_names, scalor_flags)
        intercept_filename = train_session + "_" + train_run + "_var_trained_model_" + time_snippet + ".hdf5"
        intercept_path = os.path.join(data_path, train_subj, "models", "outputs", intercept_filename)
        #print(intercept_path)
        [lasso_intercept_tmp] = load_file(intercept_path, ['intercept'], [True])
        lasso_betas += lasso_betas_tmp
        lasso_intercept += lasso_intercept_tmp
        #print(lasso_intercept)
        X_means += X_mean_tmp
    
    lasso_betas = lasso_betas / len(train_str_subj_list)
    lasso_intercept = lasso_intercept / len(train_str_subj_list)

    X_means = X_means / len(train_str_subj_list)

       
    #load test data
    test_subjects = subjects.copy()
    #print(test_subjects)
    test_subjects.remove(train_subject)
    #print(test_subjects)
    
    for test_subject in test_subjects:
        
        test_subj = "subj0" + str(test_subject)
        print("")
        print("-------------------------")
        print("test subject: ", test_subj)
        
        test_subj_file_path = os.path.join(data_path, test_subj, test_subj + "-session-run_list.txt")
        #test_subj_file_path = os.path.join(data_path, test_subj, test_subj + "-session-run_list_test0509.txt")
        test_subj_list = pd.read_csv(test_subj_file_path, header=None).squeeze("columns")
        test_str_subj_list = test_subj_list.tolist()
        #test_str_subj_list = ["subj02-session21-run01", "subj02-session21-run02"]
        
        n_runs = len(test_str_subj_list)
        
        corrs = np.zeros((n_runs,))
        corrs_p = np.zeros((n_runs,))
        r2s = np.zeros((n_runs,))
        rmses = np.zeros((n_runs,))
        
        #load test data
        for i, test_entry in enumerate(test_str_subj_list):

            print(test_entry)

            test_subj, test_session, test_run = test_entry.split('-')

            #load test data
            #print("loading data...")

            hr_file = test_session + "_" + test_run + "_downsampled_rr_" + time_snippet + ".hdf5"
            brain_file = test_session + "_" + test_run + "_var_residuals_union_MNI_" + time_snippet + ".hdf5" 

            hr_path = os.path.join(data_path, test_subj, "models", "inputs", hr_file)
            brain_path = os.path.join(data_path, "group", test_subj, "models", "inputs", brain_file)
            #print(brain_path)

            [X] = load_file(brain_path, ['X'], [False])
            [y] = load_file(hr_path, [time_snippet], [False])
            
            X_centered = X - X_means
            
            #print("testing model...")
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
            y_pred_file = "train-" + train_subj + "_test-" + test_subj + "-" + test_session + "-" + test_run + "_var_y_predicted_test_" + time_snippet + ".hdf5"
            output_path = os.path.join(data_path, "group", test_subj, "models", "outputs")
            save_file(output_path, y_pred_file, [y, y_predicted], ['y', 'y_predicted'])
            #print("")
                
        #save cv metrics
        print("")
        output_path = os.path.join(data_path, "group", test_subj, "models", "outputs")
        cv_file = "train-" + train_subj + "_test-" + test_subj + "_var_cv_r_p_r2_rmse_" + time_snippet + ".hdf5"
        save_file(output_path, cv_file, [corrs, corrs_p, r2s, rmses], ['corr', 'p', 'r2', 'rmse'])
        
        print("-------------------------")
    
    print("=========================")

        
end = time.time()
print("")
print("Total time:", end - start)


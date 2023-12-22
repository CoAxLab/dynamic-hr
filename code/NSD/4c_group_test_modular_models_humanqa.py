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
data_path_nsd = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"

#subjects = [1,2,3,4,5,8]
subjects = [1,2,3,4,5,6,7,8]

#parameters
RANDOM_STATE = 2
time_shift = 6
time_snippet = "time" + str(time_shift)
n_features = 1271436

lasso_betas = np.zeros(n_features) 
X_means = np.zeros(n_features)
lasso_intercept = 0
n_runs = 0

start = time.time()

print("=========================")
print("")
print("load train data")

for train_subject in subjects:
    
    train_subj = "subj0" + str(train_subject)
    
    print("")
    print("-------------------------")
    print("train subject: ", train_subj)
    
    train_subj_file_path = os.path.join(data_path_nsd, train_subj, train_subj + "-session-run_list.txt")
    train_subj_list = pd.read_csv(train_subj_file_path, header=None).squeeze("columns")
    train_str_subj_list = train_subj_list.tolist()
    #train_str_subj_list = ["subj01-session21-run01", "subj01-session21-run02"]
    n_runs += len(train_str_subj_list)
    #print(n_runs)
    
    #load train data
    for train_entry in train_str_subj_list:

        print(train_entry)

        train_subj, train_session, train_run = train_entry.split('-')

        model_filename = train_session + "_" + train_run + "_var_trained_model_union_MNI_" + time_snippet + ".hdf5"
        model_path = os.path.join(data_path_nsd, "group", train_subj, "models", "outputs", model_filename)
        #print(model_path)
        dataset_names = ['betas', 'X_mean']
        scalor_flags = [False, False]
        [lasso_betas_tmp, X_mean_tmp] = load_file(model_path, dataset_names, scalor_flags)
        intercept_filename = train_session + "_" + train_run + "_var_trained_model_" + time_snippet + ".hdf5"
        intercept_path = os.path.join(data_path_nsd, train_subj, "models", "outputs", intercept_filename)
        #print(intercept_path)
        [lasso_intercept_tmp] = load_file(intercept_path, ['intercept'], [True])
        lasso_betas += lasso_betas_tmp
        #print(lasso_betas.shape, lasso_betas)
        lasso_intercept += lasso_intercept_tmp
        #print(lasso_intercept)
        X_means += X_mean_tmp
    
lasso_betas = lasso_betas / n_runs
lasso_intercept = lasso_intercept / n_runs
X_means = X_means / n_runs

print("")
print("-------------------------")

       
#load test data
data_path_qa_brain = "/media/amysentis/ExtraDrive1/Amy/dynamicHR/HumanQA/analysis/group_analysis/"
data_path_qa_hp = "/media/amysentis/ExtraDrive1/Amy/dynamicHR/HumanQA/analysis/final_analysis/"
sessions = ["ses-03", "ses-04", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", \
            "ses-10", "ses-12", "ses-13", "ses-14", "ses-15", "ses-17", "ses-18"]
folder_names = ["rest", "task01", "task02"]

time_shift_qa = 5
time_snippet_qa_brain = "time" + str(time_shift_qa)
time_snippet_qa_hp = "time-" + str(time_shift_qa)
n_runs_qa = 42
corrs = np.zeros((n_runs_qa,))
corrs_p = np.zeros((n_runs_qa,))
r2s = np.zeros((n_runs_qa,))
rmses = np.zeros((n_runs_qa,))

i = 0

#testing
# sessions = ["ses-03", "ses-04"]
# folder_names = ["rest", "task01"]
# n_runs_qa = 4
# corrs = np.zeros((n_runs_qa,))
# corrs_p = np.zeros((n_runs_qa,))
# r2s = np.zeros((n_runs_qa,))
# rmses = np.zeros((n_runs_qa,))

print("")
print("=========================")
print("")
print("load test data")
    
for session in sessions:

    for task in folder_names:
    
        print("")
        print("-------------------------")
        print("Human QA test run: ", session, task)
        
        hr_file = task + "_downsampled_rr_" + time_snippet_qa_hp + ".txt"
        brain_file = session + "_" + task + "_var_residuals_mask-nsd-union_" + time_snippet_qa_brain + ".hdf5" 

        hr_path = os.path.join(data_path_qa_hp, session, "models", "inputs", hr_file)
        #print(hr_path)
        brain_path = os.path.join(data_path_qa_brain, "models", "inputs", brain_file)
        #print(brain_path)

        [X] = load_file(brain_path, [time_snippet_qa_brain], [False])
        y = np.loadtxt(hr_path)
        
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
        #y_pred_file = "train-NSDave_test-" + session + "-" + task + "_var_y_predicted_test_" + time_snippet_qa_brain + ".hdf5"
        y_pred_file = "train-NSD-all-ave_test-" + session + "-" + task + "_var_y_predicted_test_" + time_snippet_qa_brain + ".hdf5"
        output_path = os.path.join(data_path_qa_brain, "models", "outputs")
        #print(output_path)
        save_file(output_path, y_pred_file, [y, y_predicted], ['y', 'y_predicted'])
        print("")
        print("-------------------------")

        i += 1
                
#save cv metrics
print("")
output_path = os.path.join(data_path_qa_brain, "models", "outputs")
#cv_file = "train-NSDave_test-QA_var_cv_r_p_r2_rmse_" + time_snippet_qa_brain + ".hdf5"
cv_file = "train-NSD-all-ave_test-QA_var_cv_r_p_r2_rmse_" + time_snippet_qa_brain + ".hdf5"
save_file(output_path, cv_file, [corrs, corrs_p, r2s, rmses], ['corr', 'p', 'r2', 'rmse'])


print("=========================")

        
end = time.time()
print("")
print("Total time:", end - start)


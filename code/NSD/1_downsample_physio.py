#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Average hr signal across each TR to get same number of timepoints as MR signal

Standardize data using z-score

"""

#imports
import h5py
import numpy as np
import os
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d


#data files
data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"

subj_file_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/data/subj-session-run_list.txt"
subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
str_subj_list = subj_list.tolist()

#parameters
fs = 400
tr = 1.333171 #pre-processed upsampled tr for NSD 1.8mm timeseries data
#1.33171?
n_trs = 226 #pre-processed upsampled n vols for NSD 1.8mm timeseries data


str_subj_list = ["subj01-session21-run02"]

for entry in str_subj_list:

	print(entry)

	subj, session, run = entry.split('-')

	#load data
	peak_file = session + "_" + run + "_puls_signal_peaks.hdf5"
	peak_path = os.path.join(data_path, subj, "physio", peak_file)
	f = h5py.File(peak_path, 'r')
	peaks = f['puls_peaks_high'][:]
	f.close()

	tr_cutoffs = np.arange(0, fs*tr*(peaks.size+1), fs*tr)
	#compute rr interval
	rr = abs(np.diff(peaks))

	#print("shapes:", peaks.shape, inst_hr.shape, tr_cutoffs.shape)

	#for i, i_tr in enumerate(tr_cutoffs, start=1):
	downsampled_rr = np.zeros((n_trs))
	for i in range(n_trs):
		#print(i)
		indices, = np.where(np.logical_and(peaks > tr_cutoffs[i], peaks < tr_cutoffs[i+1]))
		#print(indices, type(indices))
		if indices.size == 0:
			downsampled_rr[i] = np.nan 
		elif indices.size > 1:
			#print(indices-1)
			#print(rr[indices-1])
			downsampled_rr[i] = np.mean(rr[indices-1]) 
			#print(downsampled_rr[i])
		else:
			downsampled_rr[i] = rr[indices-1]

	# for testing - see downsampling_notes excel spreadsheet
	#print("tr cutoffs", tr_cutoffs[:20])
	#print("peaks", peaks[:25])
	#print("rr", rr[:25])
	#print("downsampled", downsampled_rr)

	#print("shapes:", peaks.shape, rr.shape, tr_cutoffs.shape, downsampled_rr.shape)


	# standardize (z-score)
	final_rr = stats.zscore(downsampled_rr, nan_policy='omit')
	#linear regression by TR number (to detrend global linear trends)
	# ^look at zscored data to see if there are global trends (linear) - plot (usually not needed for physio data)
	# ^ if so, do lin reg and use to residuals (and szcore)

	#impute missing values using cubic spline interpolation
	if np.isnan(np.sum(final_rr)):
		final_rr_df_temp = pd.DataFrame(final_rr)
		final_rr_df = final_rr_df_temp.interpolate(method='spline', order=3)
		final_rr = final_rr_df.to_numpy()
		print("hr data point(s) imputed with cubic spline interpolation")
		#reshape back to array instead of matrix with one column
		final_rr = final_rr.reshape(-1)

	print("")

	#save 
	final_rr_filename = session + "_" + run + "_downsampled_rr.hdf5"
	final_rr_path = os.path.join(data_path, subj, "physio", final_rr_filename)
	f = h5py.File(final_rr_path, "w")
	f.create_dataset("downsampled_rr", data = final_rr)
	f.close()
	print("saved to file:", final_rr_path)












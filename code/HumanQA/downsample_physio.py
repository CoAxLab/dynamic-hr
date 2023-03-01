#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Average ecg signal across each TR to get same number of timepoints as MR signal

Standardize data using z-score

"""

#imports
import numpy as np
import os
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d


#data files
data_path = "/data/Amy/dynamicHR/HumanQA/analysis/full_analysis/"

peak_file = "corrected_peaks_ecg.txt"

sessions = ["ses-03", "ses-04", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", \
            "ses-10", "ses-12", "ses-13", "ses-14", "ses-15", "ses-17", "ses-18"]
#sessions = ["ses-03"]

folder_names = ["rest", "task01", "task02"]
#folder_names = ["task02"]

fs = 400 #hz
tr = 1.5 #s

n_trs = 353 #number of total trs

for session in sessions:

	for run in folder_names:

		#load data
		peak_path = os.path.join(data_path, session, "physio", run, peak_file)
		peaks = np.loadtxt(peak_path)

		tr_cutoffs = np.arange(0, fs*tr*(peaks.size+1), fs*tr)
		#compute rr interval
		rr = abs(np.diff(peaks))

		#calculate inst hr (in bpm)
		#inst_hr = (fs/rr)*60 

		#print("shapes:", peaks.shape, inst_hr.shape, tr_cutoffs.shape)

		#for i, i_tr in enumerate(tr_cutoffs, start=1):
		downsampled_rr = np.zeros((n_trs)) #correct?
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
		outpath = os.path.join(data_path, session, "physio", run, "downsampled_rr.txt")
		np.savetxt(outpath, final_rr)
		print("saved to file:", outpath)












#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run niphlem resp, pulse QC report for NSD
- save html qc report
- save hdf5 file with high filtered pulse ox signal and corrected peaks (hr)

Generate variability model regressors (both cardiac and respiration)

"""

#imports
import h5py
import numpy as np
import os
import pandas as pd

from niphlem.models import HVPhysio, RVPhysio, RetroicorPhysio
from niphlem.report import make_pulseox_report, make_resp_report
from scipy.interpolate import interp1d


#data files
input_data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/data/nsddata/ppdata/"
output_data_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/"

subj_file_path = "/media/amysentis/data2/Amy/dynamicHR/NSD/analysis/subj02/subj02-session-run_list.txt"
subj_list = pd.read_csv(subj_file_path, header=None).squeeze("columns")
str_subj_list = subj_list.tolist()

#parameters - all
fs1 = 50
fs2 = 400
tr = 1.333171 #pre-processed upsampled tr for NSD 1.8mm timeseries data
n_vols = 226 #pre-processed upsampled n vols for NSD 1.8mm timeseries data
#frame_times1 = np.arange(0, 300.8, 1/fs1) #for current data at 50hz
frame_times2 = np.arange(0, 300.8, 1/fs2) #for upsampling data to 400hz

#parameters - puls
puls_delta = 200
puls_peak_rise = 0.75 
puls_high_pass = 0.6 #change to 0.1?
puls_low_pass = 5.0

#parameters - resp
resp_delta = 800
resp_peak_rise = 0.5
resp_high_pass = 0.1
resp_low_pass = 0.5

#str_subj_list = ["subj01-session21-run03", "subj01-session21-run04", "subj01-session22-run01", "subj02-session21-run03"]

for entry in str_subj_list:

    print(entry)

    subj, session, run = entry.split('-')

    resp_filename = "physio_" + session + "_" + run + "_resp.tsv"
    resp_file_path = os.path.join(input_data_path, subj, "physio", resp_filename)
    puls_filename = "physio_" + session + "_" + run + "_puls.tsv"
    puls_file_path = os.path.join(input_data_path, subj, "physio", puls_filename)

    #load resp and puls signal using niphlem and upsample to 400hz
    resp_signal = np.loadtxt(resp_file_path)
    #print(resp_signal.shape)
    frame_times1_resp = np.linspace(0,300.8,resp_signal.shape[0])
    #print(frame_times1_resp.shape)
    resp_data400 = interp1d(frame_times1_resp, resp_signal, fill_value="extrapolate")(frame_times2)

    puls_signal = np.loadtxt(puls_file_path)
    #print(puls_signal.shape)
    frame_times1_puls = np.linspace(0,300.8,puls_signal.shape[0])
    #print(frame_times1_puls.shape)
    puls_data400 = interp1d(frame_times1_puls, puls_signal, fill_value="extrapolate")(frame_times2)

    outpath = os.path.join(output_data_path, subj, "physio")

    resp_report, resp_dict = make_resp_report(resp_data400,
                                                fs=fs2,
                                                delta=resp_delta,
                                                high_pass=resp_high_pass,
                                                low_pass=resp_low_pass)
                                                #outpath=outpath)
    # resp_report_path = os.path.join(outpath, session + "_" + run + "_resp_qc.html")
    # resp_report.save_as_html(resp_report_path)
    # print("Respiration QC report generated and saved to file:", resp_report_path)

    puls_report, puls_dict = make_pulseox_report(puls_data400,
                                                   fs=fs2,
                                                   delta_low=resp_delta,
                                                   delta_high=puls_delta,
                                                   peak_rise_low=resp_peak_rise, 
                                                   peak_rise_high=puls_peak_rise, 
                                                   resp_high_pass=resp_high_pass,
                                                   resp_low_pass=resp_low_pass,
                                                   hr_high_pass=puls_high_pass,
                                                   hr_low_pass=puls_low_pass)
                                                   #outpath=outpath)
    # puls_report_path = os.path.join(outpath, session + "_" + run + "_puls_qc.html")
    # puls_report.save_as_html(puls_report_path)
    # print("Pulse-oximetry QC report generated and saved to file:", puls_report_path)
    # ^note only care about cardiac signal recovered from pulse-ox (not resp)

    puls_filt_high = puls_dict['high_filtered_signal']
    puls_peaks_high = puls_dict['peaks_high']
    # puls_signal_path = os.path.join(outpath, session + "_" + run + "_puls_signal_peaks.hdf5")
    # f = h5py.File(puls_signal_path, "w")
    # f.create_dataset("puls_filt_high", data = puls_filt_high)
    # f.create_dataset("puls_peaks_high", data = puls_peaks_high)
    # f.close()
    # print("Cardiac signal and peaks from pulse-oximetry saved to file:", puls_signal_path)

    # hv_puls = HVPhysio(physio_rate=fs2,
    #                     t_r=tr,
    #                     delta=puls_delta,
    #                     peak_rise=puls_peak_rise,
    #                     time_window=4.5, #~3 trs
    #                     low_pass=puls_low_pass, 
    #                     high_pass=puls_high_pass)

    # rv_resp = RVPhysio(physio_rate=fs2,
    #                     t_r=tr,
    #                     time_window=4.5, #~3 trs
    #                     low_pass=resp_low_pass, 
    #                     high_pass=resp_high_pass)

    retro_puls = RetroicorPhysio(physio_rate=fs2, 
                            	t_r=tr, 
                            	delta=puls_delta,
                            	peak_rise=puls_peak_rise,
                            	low_pass=puls_low_pass, 
                            	high_pass=puls_high_pass, 
                            	order=2 # order 2 of retroicor expansion
                           		)

    retro_resp = RetroicorPhysio(physio_rate=fs2, 
                            	 t_r=tr, 
                            	 delta=resp_delta,
                            	 peak_rise=resp_peak_rise,
                            	 low_pass=resp_low_pass, 
                            	 high_pass=resp_high_pass, 
                            	 order=1 # order 1 of retroicor expansion
                           		 )

    frame_times = np.arange(n_vols)*tr

    # hv_regressors = hv_puls.compute_regressors(signal=puls_data400, time_scan=frame_times)
    # rv_regressors = rv_resp.compute_regressors(signal=resp_data400, time_scan=frame_times)
    retro_puls_regressors = retro_puls.compute_regressors(signal=puls_data400, time_scan=frame_times)
    retro_resp_regressors = retro_resp.compute_regressors(signal=resp_data400, time_scan=frame_times)

    # out_filename = session + "_" + run + "_var_regressors.hdf5"
    out_filename = session + "_" + run + "_retro_regressors.hdf5"
    reg_outpath = os.path.join(outpath, out_filename)

    # combine pulse, resp together into 1 hdf5 file for each run
    f = h5py.File(reg_outpath, "w")
    # f.create_dataset("HV_regressors", data = hv_regressors)
    # f.create_dataset("RV_regressors", data = rv_regressors)
    f.create_dataset("retro_puls_regressors", data = retro_puls_regressors)
    f.create_dataset("retro_resp_regressors", data = retro_resp_regressors)
    f.close()
    #print("Cardiac, resp variability regressors computed and saved to file:", reg_outpath)
    print("Cardiac, resp retroicor regressors computed and saved to file:", reg_outpath)

  

    print("")


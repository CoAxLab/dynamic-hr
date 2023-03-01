#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run niphlem ECG QC report for Human QA data

Generate RETROICOR regressors (both cardiac and respiration)

"""

#imports
import numpy as np
import os

from niphlem.input_data import load_cmrr_data, load_cmrr_info
from niphlem.models import RetroicorPhysio, HVPhysio, RVPhysio
from niphlem.report import make_ecg_report, make_resp_report

#data files
#data_path = "/Users/amy/Documents/PittCMU/G3/dynamicHR/HumanQA/ses-18/physio/"
input_data_path = "/data/Amy/dynamicHR/HumanQA/data/flywheel/qalab/Prisma-Human-QA/sub-06/"

#output_data_path = "/data/Amy/dynamicHR/HumanQA/data/prelim_analysis/"
output_data_path = "/data/Amy/dynamicHR/HumanQA/analysis/full_analysis/"

#sessions = ["ses-04", "ses-06", "ses-08", "ses-10", "ses-12", "ses-13", "ses-14", "ses-18"]
#sessions = ["ses-03", "ses-05", "ses-07", "ses-09", "ses-15", "ses-17"]
sessions = ["ses-03", "ses-04", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", \
            "ses-10", "ses-12", "ses-13", "ses-14", "ses-15", "ses-17", "ses-18"]

rest_folder = "func-bold_task-resting_run-01_PhysioLog"

rest_file_names = ["Physio_20210315_141521_58e9ae42-7480-4512-809d-6893fa884479",
 					"Physio_20210322_140315_89a222d1-4c24-4caf-a898-f06c6bfd2342",
					 "Physio_20210329_141226_9046fd48-be3e-4185-a577-15b58020fa5e",
 			  	     "Physio_20210405_140008_a1364c13-c8ce-4692-a102-e677e778a15c",
					 "Physio_20210412_134828_4319c2dc-568c-4f60-825c-b2b9fb0eace6",
 			         "Physio_20210419_140257_e4622db0-d1b9-48de-aa88-eb6f21710a6d",
					 "Physio_20210426_140508_3bbb42bf-e8b2-4586-935b-156f3e0eb98b",
 			         "Physio_20210503_135746_f9718345-ae6c-48be-a219-1541ba8c5100",
 			         "Physio_20210517_135623_18055d14-8610-4f7f-8eaa-11ac82194614",
 			         "Physio_20210607_135323_bd0abac1-1487-4c3c-98c2-166d9c91ab03",
 			         "Physio_20210614_140425_cc77e7ab-a49f-478c-87fd-4c069a551733",
					 "Physio_20210621_140517_87a9f2fc-1513-435b-8356-e09f25c6038a",
			         "Physio_20210716_113127_a2bccc55-74c9-44d2-8574-d19c8eeb8360",
 			         "Physio_20210726_140318_a30567eb-1bc2-4b93-85e0-c91ad52215c0"]

task01_folders = ["func-bold_task-randomagent_run-01_PhysioLog",
				  "func-bold_task-baityk3_run-01_PhysioLog",
				  "func-bold_task-baityk3_run-01_PhysioLog",
				  "func-bold_task-shooterturn2_run-01_PhysioLog",
				  "func-bold_task-shooterturn2_run-01_PhysioLog",
				  "func-bold_task-randomagent_run-01_PhysioLog",
				  "func-bold_task-baityk7_run-01_PhysioLog",
				  "func-bold_task-baityk7_run-01_PhysioLog",
				  "func-bold_task-shootermirror1_run-01_PhysioLog",
				  "func-bold_task-shootermirror1_run-01_PhysioLog",
				  "func-bold_task-random_run-01_PhysioLog",
				  "func-bold_task-random_run-01_PhysioLog",
			      "func-bold_task-shooterturn2_run-01_PhysioLog",
				  "func-bold_task-diffusivebandit_run-01_PhysioLog"]

task01_file_names = ["Physio_20210315_142501_009d4b64-db94-48d0-840f-ad1c2676310b",
					 "Physio_20210322_141247_80350a05-871c-4564-968c-b90e7987a319",
					 "Physio_20210329_142201_9c552968-88a0-45c3-8a4c-49d9f69bcafc",
			  	     "Physio_20210405_141158_88f99747-62e9-443a-915b-05bea267655b",
					 "Physio_20210412_135800_4aaa4178-fefc-4317-bfce-86582d17d5e5",
			         "Physio_20210419_141228_e8cd29ef-9797-4b35-9940-0281c2203244",
					 "Physio_20210426_141447_a8c9baa6-062a-4e27-ba54-c9843ce3e08f",
			         "Physio_20210503_140720_17062698-45d5-4316-81de-909985337054",
			         "Physio_20210517_140554_094e3c2f-31b6-4156-bca2-ae45428feaa0",
			         "Physio_20210607_140255_f4d2b66d-a362-414f-b7c2-1fae04e4b1d9",
			         "Physio_20210614_141356_94aca764-e607-43a1-ba22-0ab4a0bf7a38",
					 "Physio_20210621_141448_c3838e6a-52ee-411c-bb88-dfe11ed74166",
			      	 "Physio_20210716_114057_54351ee7-cdd8-4e11-a5af-146be26fe94a",
			         "Physio_20210726_141249_40f4698a-ea11-467e-ba9d-eafe78317dae"]

task02_folders = ["func-bold_task-randomagent_run-02_PhysioLog",
				  "func-bold_task-baityk3_run-02_PhysioLog",
				  "func-bold_task-baityk3_run-02_PhysioLog",
				  "func-bold_task-shooterturn2_run-02_PhysioLog",
			      "func-bold_task-shooterturn2_run-02_PhysioLog",
				  "func-bold_task-randomagent_run-02_PhysioLog",
				  "func-bold_task-baityk7_run-02_PhysioLog",
				  "func-bold_task-baityk7_run-02_PhysioLog",
				  "func-bold_task-shootermirror1_run-02_PhysioLog",
				  "func-bold_task-shootermirror1_run-02_PhysioLog",
				  "func-bold_task-random_run-02_PhysioLog",
				  "func-bold_task-random_run-02_PhysioLog",
			      "func-bold_task-shooterturn2_run-02_PhysioLog",
				  "func-bold_task-diffusivebandit_run-02_PhysioLog"]

task02_file_names = ["Physio_20210315_143507_32eadc53-2bdf-4d45-beb0-692af08450e6",
					 "Physio_20210322_142314_42d1c4a9-c882-47c6-af56-25754b9aec97",
					 "Physio_20210329_143206_27636298-b2b8-4c6c-90fb-0b7ff0a2486f",
			  	     "Physio_20210405_142103_d5621c54-0a57-453a-8315-ca402fee60f6",
					 "Physio_20210412_140800_c5ed01ed-614e-4307-8215-6a183c0c1eac",
			         "Physio_20210419_142225_36dd33b8-e959-474d-9d7a-04e1a64ffa6b",
					 "Physio_20210426_142434_599f8751-6335-4f41-93e1-22f83bc022b2",
			         "Physio_20210503_141852_0a31dbea-2018-4061-a09b-2a1a92a3f6e0",
			         "Physio_20210517_141532_d9df2f02-03c2-4317-bfc1-1d951281f349",
			         "Physio_20210607_141257_ace84342-fdc0-4b7f-934b-948783a7c042",
			         "Physio_20210614_142330_6e3ba158-c111-4f5e-a23c-071c2d01da57",
					 "Physio_20210621_142424_9f9ee4a2-e461-404f-ae3c-b5ced4bfb4f3",
			      	 "Physio_20210716_115104_732478c9-b7fa-4b73-86f8-843247fdfacf",
			         "Physio_20210726_142305_6cfe913f-9864-4f8f-8bf3-f28d7f8bfb34"]

info_log = "_Info.log"
ecg_log = "_ECG.log"
resp_log = "_RESP.log"

info_files = []
ecg_files = []
resp_files = []
folder_names = []

# the below is definitely not the cleanest way - need to fix up
for (session, rest_file, task01_folder, task02_folder, task01_file, task02_file) in \
		zip(sessions, rest_file_names, task01_folders, task02_folders, task01_file_names, task02_file_names):
	
	rest_info_filename = rest_file + info_log
	rest_info_log = os.path.join(input_data_path, session, rest_folder, rest_info_filename)
	rest_ecg_filename = rest_file + ecg_log
	rest_ecg_log = os.path.join(input_data_path, session, rest_folder, rest_ecg_filename)
	rest_resp_filename = rest_file + resp_log
	rest_resp_log = os.path.join(input_data_path, session, rest_folder, rest_resp_filename)
	task01_info_filename = task01_file + info_log
	task01_info_log = os.path.join(input_data_path, session, task01_folder, task01_info_filename)
	task01_ecg_filename = task01_file + ecg_log
	task01_ecg_log = os.path.join(input_data_path, session, task01_folder, task01_ecg_filename)
	task01_resp_filename = task01_file + resp_log
	task01_resp_log = os.path.join(input_data_path, session, task01_folder, task01_resp_filename)
	task02_info_filename = task02_file + info_log
	task02_info_log = os.path.join(input_data_path, session, task02_folder, task02_info_filename)
	task02_ecg_filename = task02_file + ecg_log
	task02_ecg_log = os.path.join(input_data_path, session, task02_folder, task02_ecg_filename)
	task02_resp_filename = task02_file + resp_log
	task02_resp_log = os.path.join(input_data_path, session, task02_folder, task02_resp_filename)

	info_files.extend((rest_info_log, task01_info_log, task02_info_log))
	ecg_files.extend((rest_ecg_log, task01_ecg_log, task02_ecg_log))
	resp_files.extend((rest_resp_log, task01_resp_log, task02_resp_log))
	folder_names.extend((os.path.join(session, "physio", "rest"), \
		os.path.join(session, "physio", "task01"), os.path.join(session, "physio", "task02")))
	#folder_names.extend((os.path.join(output_data_path, session, "rest"), \
	#	os.path.join(output_data_path, session, "task01"), os.path.join(output_data_path, session, "task02")))


#parameters - all
fs = 400
tr = 1.5
n_vols = 353

#parameters - ecg
ecg_delta = 200
ecg_peak_rise = 0.75
ground = 0
ecg_high_pass = 0.6
ecg_low_pass = 5.0

#parameters - resp
resp_delta = 800
resp_peak_rise = 0.5
resp_high_pass = 0.1
resp_low_pass = 0.5

#generate qc report and regressors for each run
for (info, ecg_data, resp_data, run) in zip(info_files, ecg_files, resp_files, folder_names): 

	print("run: ", run)
	
	task = os.path.basename(run)

	time_traces, meta_info = load_cmrr_info(info)

	ecg_signal, meta_info = load_cmrr_data(ecg_data, info_dict=meta_info, sig_type="ECG")
	#^note: physio signal aligned to start of mr signal (by default)

	resp_signal, meta_info = load_cmrr_data(resp_data, info_dict=meta_info, sig_type="RESP")

	outpath = os.path.join(output_data_path, run)

	# ecg_report, output_dict = make_ecg_report(ecg_signal,
	#                                           fs=fs,
	#                                           delta=ecg_delta,
	#                                           ground=ground,
	#                                           high_pass=ecg_high_pass,
	#                                           low_pass=ecg_low_pass,
	#                                           outpath=outpath)

	# print("ECG QC report generated and save to file:", outpath, "ecg_qc.html")

	# resp_report, output_dict = make_resp_report(resp_signal,
	#                                             fs=fs,
	#                                             delta=resp_delta,
	#                                             high_pass=resp_high_pass,
	#                                             low_pass=resp_low_pass,
	#                                             outpath=outpath)

	# print("Respiration QC report generated and save to file:", outpath, "resp_qc.html")

	# retro_ecg = RetroicorPhysio(physio_rate=fs, 
    #                         	t_r=tr, 
    #                         	delta=ecg_delta,
    #                         	peak_rise=ecg_peak_rise,
    #                         	columns="mean", 
    #                         	low_pass=ecg_low_pass, 
    #                         	high_pass=ecg_high_pass, 
    #                         	order=2 # order 2 of retroicor expansion
    #                        		)

	# retro_resp = RetroicorPhysio(physio_rate=fs, 
    #                         	 t_r=tr, 
    #                         	 delta=resp_delta,
    #                         	 peak_rise=resp_peak_rise,
    #                         	 low_pass=resp_low_pass, 
    #                         	 high_pass=resp_high_pass, 
    #                         	 order=1 # order 1 of retroicor expansion
    #                        		 )

	hv_ecg = HVPhysio(physio_rate=fs,
					  t_r=tr,
					  delta=ecg_delta,
					  peak_rise=ecg_peak_rise,
					  columns="mean",
					  time_window=4.5, #check this
					  low_pass=ecg_low_pass, 
                      high_pass=ecg_high_pass)

	rv_resp = RVPhysio(physio_rate=fs,
					   t_r=tr,
					   time_window=4.5, #check this
					   low_pass=resp_low_pass, 
                       high_pass=resp_high_pass)

	frame_times = np.arange(n_vols)*tr

	# cardiac_retroicor_regressors = retro_ecg.compute_regressors(signal=ecg_signal, time_scan=frame_times)

	# cardiac_out_filename = task + "_cardiac_RETROICOR_regressors.txt"
	# reg_outpath = os.path.join(outpath, cardiac_out_filename)
	# np.savetxt(reg_outpath, cardiac_retroicor_regressors)
	# print("cardiac RETROICOR regressors computed and saved to file:", reg_outpath)

	# resp_retroicor_regressors = retro_resp.compute_regressors(signal=resp_signal, time_scan=frame_times)

	# resp_out_filename = task + "_resp_RETROICOR_regressors.txt"
	# reg_outpath = os.path.join(outpath, resp_out_filename)
	# np.savetxt(reg_outpath, resp_retroicor_regressors)
	# print("respiration RETROICOR regressors computed and saved to file:", reg_outpath)

	hv_regressors = hv_ecg.compute_regressors(signal=ecg_signal, time_scan=frame_times)

	hv_out_filename = task + "_HV_regressors.txt"
	reg_outpath = os.path.join(outpath, hv_out_filename)
	np.savetxt(reg_outpath, hv_regressors)
	print("HR variability regressors computed and saved to file:", reg_outpath)

	rv_regressors = rv_resp.compute_regressors(signal=resp_signal, time_scan=frame_times)

	rv_out_filename = task + "_RV_regressors.txt"
	reg_outpath = os.path.join(outpath, rv_out_filename)
	np.savetxt(reg_outpath, rv_regressors)
	print("respiration variability regressors computed and saved to file:", reg_outpath)

	print("")


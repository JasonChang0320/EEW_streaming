import h5py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
import time

def triggered_stations(waveforms, stations, p_picks, mask_sec, sampling_rate=200, max_station_num=25):
    if (
        len(stations) < max_station_num
    ):  # triggered station < max_station_number (default:25) zero padding
        for zero_pad_num in range(max_station_num - len(stations)):
            # print(f"second {waveform.shape}")
            waveforms = np.concatenate(
                (waveforms, np.expand_dims(np.zeros_like(waveforms[0]), axis=0)),
                axis=0,
            )
            stations = np.concatenate(
                (stations, np.expand_dims(np.zeros_like(stations[0]), axis=0)),
                axis=0,
            )
    else:
        waveforms = waveforms[:25, :, :]
        stations = stations[:25, :]
        p_picks = p_picks[:25]
    # mask
    waveforms[:, p_picks[0] + (mask_sec * sampling_rate) :, :] = 0
    for i in range(len(p_picks)):
        if p_picks[i] > p_picks[0] + (mask_sec * sampling_rate):
            waveforms[i, :, :] = 0
            stations[i] = np.zeros_like(stations[0])
    return waveforms, stations


data_path = "../TTSAM/2016_sample.hdf5"
output_path="./waveforms/"
init_event_metadata = pd.read_hdf(data_path, "metadata/event_metadata")
sample_eqid = init_event_metadata.query("year==2016")["EQ_ID"]
data_length_sec = 15
sampling_rate = 200

with h5py.File(data_path, "r") as origin:
    for eqid in sample_eqid.values:

        data = origin["data"][str(eqid)]
        waveforms = np.array(
            data["acc_traces"][:, : (data_length_sec * sampling_rate), :]
        )
        stations = np.array(data["station_location"])
        vs30 = np.array(data["Vs30"])
        p_picks = np.array(data["p_picks"])
        event_time = (datetime.now()+ timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        for mask_sec in [3,5,7,10]:
            if mask_sec==3:
                time.sleep(3)
            cut_waveforms, cut_stations = triggered_stations(
                waveforms, stations, p_picks, mask_sec=mask_sec
            )
            # generate json file
            data = '{{"event_time": "{}", "waveform": {}, "station": {}}}\n'
            content = ""
            content += data.format(event_time, cut_waveforms.tolist(), cut_stations.tolist())
            file_name=f"{eqid}_{mask_sec}_sec.json"
            with open(output_path + file_name, "wt", encoding="utf-8") as f:
                f.write(content)
            if mask_sec==7:
                time.sleep(3)
            else:
                time.sleep(2)
    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)



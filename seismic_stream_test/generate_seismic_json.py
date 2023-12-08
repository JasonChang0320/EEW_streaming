import json
import numpy as np
from datetime import datetime, timedelta
import time
import os
import shutil

def generate_random_json(output_path,file_name):
    data='{{"event_time": "{}", "waveform": {}, "station": {}}}\n'
    content = ''
    event_time = (datetime.now()+ timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
    waveform = np.random.rand(25, 3000, 3).tolist()
    station = np.random.rand(25, 3).tolist()

    content += data.format(
                    event_time,
                    waveform,
                    station)
    
    with open(output_path + file_name,"wt", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    output_path="./waveforms/"
    for i in range(50):
        file_name=f"random_seismic_data{i}.json"
        generate_random_json(output_path,file_name)
        time.sleep(3)
    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)
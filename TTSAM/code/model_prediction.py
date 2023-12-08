import h5py
import matplotlib.pyplot as plt
import argparse

plt.subplots()
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from CNN_Transformer_Mixtureoutput_TEAM import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding,
    TransformerEncoder,
    full_model,
)

from multiple_sta_dataset import multiple_station_dataset

from plot_predict_map import true_predicted

#######TODO
# parser = argparse.ArgumentParser()
# parser.add_argument("-s", "--sec", help="masked data length", type=int, required=True)
# args = parser.parse_args()

# mask_after_sec = args.sec
mask_after_sec = 3
#######
label = "pga"
data = multiple_station_dataset(
    "../2016_sample.hdf5",
    mode="test",
    mask_waveform_sec=mask_after_sec,
    test_year=2016,
    label_key=label,
    mag_threshold=0,
    input_type="acc",
    data_length_sec=15,
)

for num in [5,10]:
    path = f"../model/model{num}.pt"
    emb_dim = 150
    mlp_dims = (150, 100, 50, 30, 10)
    CNN_model = CNN(mlp_input=5665)
    pos_emb_model = PositionEmbedding(emb_dim=emb_dim)
    transformer_model = TransformerEncoder()
    mlp_model = MLP(input_shape=(emb_dim,), dims=mlp_dims)
    mdn_model = MDN(input_shape=(mlp_dims[-1],))
    full_Model = full_model(
        CNN_model,
        pos_emb_model,
        transformer_model,
        mlp_model,
        mdn_model,
        pga_targets=25,
        data_length=3000,
    )
    full_Model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    loader = DataLoader(dataset=data, batch_size=1)

    Mixture_mu = []
    Label = []
    P_picks = []
    EQ_ID = []
    Label_time = []
    Sta_name = []
    Lat = []
    Lon = []
    Elev = []
    for j, sample in enumerate(loader):
        picks = sample["p_picks"].flatten().numpy().tolist()
        label_time = sample[f"{label}_time"].flatten().numpy().tolist()
        lat = sample["target"][:, :, 0].flatten().tolist()
        lon = sample["target"][:, :, 1].flatten().tolist()
        elev = sample["target"][:, :, 2].flatten().tolist()
        P_picks.extend(picks)
        P_picks.extend([np.nan] * (25 - len(picks)))
        Label_time.extend(label_time)
        Label_time.extend([np.nan] * (25 - len(label_time)))
        Lat.extend(lat)
        Lon.extend(lon)
        Elev.extend(elev)

        eq_id = sample["EQ_ID"][:, :, 0].flatten().numpy().tolist()
        EQ_ID.extend(eq_id)
        EQ_ID.extend([np.nan] * (25 - len(eq_id)))
        weight, sigma, mu = full_Model(sample)

        weight = weight.cpu()
        sigma = sigma.cpu()
        mu = mu.cpu()
        if j == 0:
            Mixture_mu = torch.sum(weight * mu, dim=2).cpu().detach().numpy()
            Label = sample["label"].cpu().detach().numpy()
        else:
            Mixture_mu = np.concatenate(
                [Mixture_mu, torch.sum(weight * mu, dim=2).cpu().detach().numpy()],
                axis=1,
            )
            Label = np.concatenate(
                [Label, sample["label"].cpu().detach().numpy()], axis=1
            )
    Label = Label.flatten()
    Mixture_mu = Mixture_mu.flatten()
    output = {
        "EQ_ID": EQ_ID,
        "p_picks": P_picks,
        f"{label}_time": Label_time,
        "predict": Mixture_mu,
        "answer": Label,
        "latitude": Lat,
        "longitude": Lon,
        "elevation": Elev,
    }
    output_df = pd.DataFrame(output)
    output_df = output_df[output_df["answer"] != 0]
    output_df.to_csv(
        f"./predict/model_{num}_{mask_after_sec}_sec_prediction.csv", index=False
    )
    fig, ax = true_predicted(
            y_true=output_df["answer"],
            y_pred=output_df["predict"],
            time=mask_after_sec,
            quantile=False,
            agg="point",
            point_size=12,
            target=label,
        )
    ax.set_title(
        f"{mask_after_sec}s True Predict Plot, 2016 data",
        fontsize=20,
    )
    fig.savefig(f"./predict/model_{num}_{mask_after_sec}_sec.png")
    plt.close()
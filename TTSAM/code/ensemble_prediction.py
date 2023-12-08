import argparse
import pandas as pd
from plot_predict_map import true_predicted

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sec", help="masked data length", type=int, required=True)
args = parser.parse_args()

mask_after_sec = args.sec

label="pga"
predict5 = pd.read_csv(
    f"./predict/model_5_{mask_after_sec}_sec_prediction.csv"
)
predict10 = pd.read_csv(
    f"./predict/model_10_{mask_after_sec}_sec_prediction.csv"
)

ensemble_predict = (predict5 + predict10) / 2
fig, ax = true_predicted(
    y_true=ensemble_predict["answer"],
    y_pred=ensemble_predict["predict"],
    time=mask_after_sec,
    quantile=False,
    agg="point",
    point_size=20,
    target=label,
    title=f"{mask_after_sec}s Ensemble Prediction, 2016 data",
)

ensemble_predict.to_csv(f"./predict/{mask_after_sec}_sec_ensemble_prediction.csv",index=False)
fig.savefig(f"./predict/{mask_after_sec}_sec_ensemble_predict_plot.png",dpi=300)
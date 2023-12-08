from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    FloatType,
    TimestampType,
    ArrayType,
)
import torch
import sys

sys.path.append("../TTSAM/code")

from CNN_Transformer_Mixtureoutput_TEAM import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding,
    TransformerEncoder,
    full_model,
)

AI_path = "../TTSAM"
num = 5
# load model
path = f"{AI_path}/model/model{num}.pt"
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
full_Model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

# SparkSession
spark = SparkSession.builder.appName("example").getOrCreate()

# schema
seismic_waveform_schema = StructType(
    [
        StructField("event_time", TimestampType(), True),
        StructField(
            "waveform",
            ArrayType(ArrayType(ArrayType(FloatType(), True), True), True),
            True,
        ),
        StructField("station", ArrayType(ArrayType(FloatType(), True), True), True),
    ]
)

json_data = spark.read.json(
    "./waveforms/random_seismic_data.json", schema=seismic_waveform_schema
)

# collect from table
waveform = torch.tensor(
    json_data.select("waveform").rdd.flatMap(lambda x: x).collect()
).to(torch.double)
input_station = torch.tensor(
    json_data.select("station").rdd.flatMap(lambda x: x).collect()
).to(torch.double)
target_station = input_station

# model predict
sample = {"waveform": waveform, "sta": input_station, "target": target_station}
weight, sigma, mu = full_Model(sample)
pga = torch.sum(weight * mu, dim=2).cpu().detach().numpy()

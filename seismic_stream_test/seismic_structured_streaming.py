import os
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


def process_batch(batch_df, epoch_id):
    print(f"Processing batch with epoch ID {epoch_id}")
    batch_df.select("event_time").show(truncate=False)
    waveform = torch.tensor(
        batch_df.select("waveform").rdd.flatMap(lambda x: x).collect()
    ).to(torch.double)
    input_station = torch.tensor(
        batch_df.select("station").rdd.flatMap(lambda x: x).collect()
    ).to(torch.double)
    target_station = input_station
    # model predict
    sample = {"waveform": waveform, "sta": input_station, "target": target_station}
    weight, sigma, mu = full_Model(sample)
    pga = torch.sum(weight * mu, dim=2).cpu().detach().numpy()
    print("model predicting...")
    print(f"predicted_pga:{pga}")


# 定義JSON文件的路徑
TEST_DATA_DIR_SPARK = f"{os.getcwd()}/waveforms"
print(TEST_DATA_DIR_SPARK)

# 建立 SparkSession
spark = (
    SparkSession.builder.appName("SeismicStreamExample")
    .getOrCreate()
)

# 定義 JSON Schema
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

# 讀取資料夾中的 JSON 檔案
json_stream_df = (
    spark.readStream.format("json")
    .schema(seismic_waveform_schema)
    .option("maxFilesPerTrigger", 1)  # 每次處理一個檔案
    .load(TEST_DATA_DIR_SPARK)  # 更換為你的實際路徑
)

# 處理 JSON 檔案中的資料
# processed_df = json_stream_df.select("event_time")

# 可設定不同query來分流需predict 的target stations
query = (
    json_stream_df.writeStream.outputMode(
        "append"
    )  # 輸出模式可以是 'append', 'complete', 或 'update'
    # .trigger(processingTime="5 seconds")
    .foreachBatch(process_batch).start()  # 使用自定義函數處理每一批資料
)
query.awaitTermination()

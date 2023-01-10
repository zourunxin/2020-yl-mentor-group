import pandas as pd

result_dir = "../result/1.10/GraphSAGE_result.csv"

df_res = pd.read_csv(result_dir)
df_data = pd.read_csv("../output/datasource_1228_new.csv")

df_res["summary"] = df_res["name"].apply(lambda x: df_data[df_data["name"] == x].values.tolist()[0][4])
df_res["description"] = df_res["name"].apply(lambda x: df_data[df_data["name"] == x].values.tolist()[0][5])

df_res = df_res[["name", "label", "predict", "summary", "description"]]
df_res = df_res[df_res["label"] != df_res["predict"]]

print(len(df_res))
df_res.to_csv("GraphSAGE_result_sample_200.csv", index=False)
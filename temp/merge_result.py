import pandas as pd

df_traget = pd.read_csv("correct_label.csv")
df_source = pd.read_csv("../traditional_classification/RandomForest_result_error.csv")

print(len(df_source))
df_diff = df_source[~ df_source["name"].isin(df_traget["name"])]
print(len(df_diff))
df_diff.to_csv("diff.csv", index=False)
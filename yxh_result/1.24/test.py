import pandas as pd

df_bert = pd.read_csv("bert_result_error.csv")
df_rf = pd.read_csv("RandomForest_result_error.csv")

bert_set = set(list(df_bert["name"]))
rf_set = set(list(df_rf["name"]))

bert_only = list(bert_set - rf_set)
rf_only = list(rf_set - bert_set)

df_bert_output = df_bert[df_bert["name"].isin(bert_only)]
df_rf_output = df_rf[df_rf["name"].isin(rf_only)]

df_bert_output.to_csv("bert_only.csv", index=False)
df_rf_output.to_csv("rf_only.csv", index=False)

print(len(df_bert_output))
print(len(df_rf_output))

both = bert_set & rf_set
df_both = df_bert[df_bert["name"].isin(both)]
df_both = df_both.rename(columns={"predict": "bert_pred"})

df_both = df_both.merge(df_rf[['name','predict']], left_on='name', right_on='name', how='left').rename(columns={"predict": "rf_pred"}).reindex(columns=["name", "label", "bert_pred", "rf_pred", "summary", "description"])
df_both.to_csv("both.csv", index=False)
print(len(df_both))

df_unique = df_both[df_both["rf_pred"] != df_both["bert_pred"]]
df_unique.to_csv("unique.csv", index=False)
print(len(df_unique))
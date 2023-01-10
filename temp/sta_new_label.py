import pandas as pd


# name,class_true,class_pred,summary,description,人工判别,说明
df_data = pd.read_csv("../result/1.9/error_list_分类.csv")

df_data["人工判别"] = df_data["人工判别"].apply(lambda x: x.split("/")[-1])
df_unique = df_data[df_data["class_true"] != df_data["人工判别"]]
df_correct = df_data[df_data["人工判别"] == df_data["class_pred"]]


print(len(df_data))
print(len(df_unique))
print(len(df_unique) / len(df_data))
print(len(df_correct))

print(df_correct)
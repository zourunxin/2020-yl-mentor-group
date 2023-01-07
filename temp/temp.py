import sys
sys.path.append("../")
import pandas as pd
import utils.CommonUtils as CommonUtils

df_old = pd.read_csv("../output/datasource_1128.csv")
df_new = pd.read_csv("../output/datasource_1228.csv")

df_old["label"] = df_old["label"].apply(lambda x: CommonUtils.convert_label(x.split("/")[0].strip(), mode="class"))

old_list = df_old.values.tolist()
new_list = df_new.values.tolist()

df_new_pred = pd.read_csv("../output/GraphSAGE_result.csv")
df_old_pred = pd.read_csv("../result/12.31/GraphSAGE_result_class.csv")

print(old_list[0])
print(new_list[0])
unique = []

for data in old_list:
    for new_data in new_list:
        if data[0] == new_data[1]:
            if data[1] != new_data[2]:
                _list = df_old_pred.loc[df_old_pred["name"] == data[0]].values.tolist()
                old_predict = _list[0][3] if len(_list) == 1 else "none"
                _list = df_new_pred.loc[df_new_pred["name"] == data[0]].values.tolist()
                new_predict = _list[0][3] if len(_list) == 1 else "none"
                if old_predict != "none" and new_predict != "none":
                    unique.append([data[0], data[1], new_data[2], old_predict, new_predict])

df_unique = pd.DataFrame(unique, columns=["name", "old_label", "new_label", "old_predict", "new_predict"])
df_unique.to_csv("unique_error.csv")


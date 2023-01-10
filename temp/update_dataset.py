import pandas as pd

df_old = pd.read_csv("../output/datasource_1228.csv")
df_res = pd.read_csv("../result/1.9/error_list_分类.csv")

df_res["人工判别"] = df_res["人工判别"].apply(lambda x: x.split("/")[-1])


old_list = df_old.values.tolist()
res_list = df_res.values.tolist()

new_list = []
for data in old_list:
    name = data[1]
    for res in res_list:
        if name == res[0]:
            data[2] = res[5]
    new_list.append(data[1:])

pd.DataFrame(new_list, columns=['name', 'label', 'text', 'summary', 'description']).to_csv("../output/datasource_1228_new.csv", index=False)
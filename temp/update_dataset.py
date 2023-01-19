import pandas as pd

df_old = pd.read_csv("../output/datasource_1228.csv")
df_res = pd.read_csv("../result/1.17/error_list_分类.csv")
df_feedback = pd.read_csv("../result/1.17/标签核对_feedback.csv")
df_my = pd.read_csv("../result/1.17/correct_label.csv")

def is_valid_label(label):
    return label in ["库", "工具", "服务", "其它"]

df_res["人工判别"] = df_res["人工判别"].apply(lambda x: x.split("/")[-1])


old_list = df_old.values.tolist()
res_list = df_res.values.tolist()
feedback_list = df_feedback.values.tolist()
my_list = df_my.values.tolist()


new_list = []
for data in old_list:
    name = data[0]
    for res in res_list:
        if name == res[0]:
            data[1] = res[5]
    if name == "adwaita-gtk2-theme":
        print(data)
    new_list.append(data)

for data in new_list:
    name = data[0]
    for feedback in feedback_list:
        if name == feedback[0]:
            data[1] = feedback[8]
    if name == "adwaita-gtk2-theme":
        print(data)


for data in new_list:
    if not is_valid_label(data[1]):
        print(data[0])



pd.DataFrame(new_list, columns=['name', 'label', 'text', 'summary', 'description']).to_csv("datasource_1228_new_new.csv", index=False)
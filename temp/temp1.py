import pandas as pd

# rpm_name,description,zero_summary,zero_description,summary,src_name,分层分类,
# src_name,zero_description,zero_summary,description,summary,primary_rpm_name,primary_description,primary_summary,rpm_name,分层分类,,,源码包名,
df_rpm = pd.read_csv("../data_resources/1228_bin.csv")
df_src = pd.read_csv("../data_resources/1228_src.csv")


df_res = pd.read_csv("../result/1.9/error_list_分类.csv")
df_res["人工判别"] = df_res["人工判别"].apply(lambda x: x.split("/")[-1])
res_list = df_res.values.tolist()

for res in res_list:
    df_rpm.loc[df_rpm["rpm_name"] == res[0], "分层分类"] = res[5]

src_rpm_map = {}
src_label_map = {}
rpm_list = df_rpm.values.tolist()
src_list = df_src.values.tolist()


for data in rpm_list:
    if not data[5] in src_rpm_map:
        src_rpm_map[data[5]] = []
    src_rpm_map[data[5]].append(data[0])

for data in src_list:
    if not data[0] in src_label_map:
        src_label_map[data[0]] = data[9]

print(len(src_label_map))

final_list = []
for src in src_rpm_map:
    rpms = src_rpm_map[src]
    if len(rpms) >= 0:
        rpm_map = {}
        for rpm in rpms:
            rpm_map[rpm] = df_rpm.loc[df_rpm["rpm_name"] == rpm].values.tolist()[0][6]
        final_list.append([src, rpm_map])

same_list = []
unique_list = []

for src_rpm in final_list:
    rpm_map = src_rpm[1]
    unique = False
    if src_rpm[0] in src_label_map:
        for key in rpm_map:
            if rpm_map[key] != src_label_map[src_rpm[0]]:
                unique = True
        src_rpm[0] = src_rpm[0] + ": " + src_label_map[src_rpm[0]]
        if unique:
            unique_list.append(src_rpm + [len(src_rpm[1])])
        else:
            same_list.append(src_rpm + [len(src_rpm[1])])
    # else:
    #     print("{} (对应在数据集中的二进制包:{})".format(src_rpm[0], list(src_rpm[1].keys())))

# srcs = src_label_map.keys()
# for src in srcs:
#     if not src in src_rpm_map:
#         print(src)

df_unique = pd.DataFrame(unique_list, columns=["src", "bin", "num_of_bin"])
df_same = pd.DataFrame(same_list, columns=["src", "bin", "num_of_bin"])


for i in range(4):
    df_ui = df_unique[df_unique["num_of_bin"] == i + 1]
    # df_si = df_same[df_same["num_of_bin"] == i + 1]
    print("含有 {} 个二进制包的源码包有 {} 个, 占比 {}".format(i + 1, len(df_ui), len(df_ui)/len(df_unique)))

df_ui = df_unique[df_unique["num_of_bin"] >= 5]
print("含有 5 个及以上二进制包的源码包有 {} 个, 占比 {}".format(len(df_ui), len(df_ui)/len(df_unique)))

df_unique.to_csv("../output/unique_label.csv", index=False)
df_same.to_csv("../output/same_label.csv", index=False)
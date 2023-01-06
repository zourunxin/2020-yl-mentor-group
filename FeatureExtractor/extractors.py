from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import csv
import numpy as np

def extract_name_feat(label):
    label = str(label)
    feat = []
    # 以lib开头的软件包，如libXXX
    feat.append(1 if label.startswith("lib") else 0)
    # 以lib结尾的软件包，如XXXlib
    feat.append(1 if label.endswith("lib") else 0)
    # 以libs结尾的软件包，如XXX-libs
    feat.append(1 if label.endswith("libs") else 0)
    # 以devel结尾的软件包，如XXX-devel
    feat.append(1 if label.endswith("devel") else 0)
    # 以static结尾的软件包，如XXX-static
    feat.append(1 if label.endswith("static") else 0)
    # 以perl-为前缀的软件包，如perl-XXX
    feat.append(1 if label.startswith("perl") else 0)
    # 以python-为前缀的软件包，如python-XXX
    feat.append(1 if label.startswith("python") else 0)
    # 以-java为后缀的软件包，如XXX-java
    feat.append(1 if label.endswith("java") else 0)
    # 以-headers作为后缀的软件包，如XXX-headers
    feat.append(1 if label.endswith("headers") else 0)
    # 以-api为后缀的软件包，如XXX-api
    feat.append(1 if label.endswith("api") else 0)
    # 以tool结尾的软件包，如XXXtool或XXX-tool
    feat.append(1 if label.endswith("tool") else 0)
    # 以tools结尾的软件包，如XXXtools或XXX-tools
    feat.append(1 if label.endswith("tools") else 0)
    # 以toolset结尾的软件包，如XXXtoolset或XXX-toolset
    feat.append(1 if label.endswith("toolset") else 0)
    # util作为中间字符串的软件包，如XXX-util-XXX
    feat.append(1 if label.find("util") != -1 else 0)
    # 以util结尾的软件包，如XXXutil或XXX-util
    feat.append(1 if label.endswith("util") else 0)
    # 以utils结尾的软件包，如XXXutils或XXX-utils
    feat.append(1 if label.endswith("utils") else 0)
    # 以utility结尾的软件包，如XXXutility或XXX-utility
    feat.append(1 if label.endswith("utility") else 0)
    # 以progs结尾的软件包，如XXX-progs
    feat.append(1 if label.endswith("progs") else 0)
    # 以-cli结尾的软件包，如XXX-cli
    feat.append(1 if label.endswith("cli") else 0)
    # 以-client结尾的软件包，如XXX-client
    feat.append(1 if label.endswith("client") else 0)
    # 以cmd结尾的软件包，如XXXcmd或XXX-cmd
    feat.append(1 if label.endswith("cmd") else 0)
    # 以ctl结尾的软件包，如XXXctl
    feat.append(1 if label.endswith("ctl") else 0)
    # 以server结尾的软件包，如XXXserver或XXX-server
    feat.append(1 if label.endswith("server") else 0)
    # 以service结尾的软件包，如XXXservice或XXX-service
    feat.append(1 if label.endswith("service") else 0)
    # 以serv结尾的软件包，如XXXserv
    feat.append(1 if label.endswith("serv") else 0)
    # 以manager结尾的软件包，如XXXmanager或XXX-manager
    feat.append(1 if label.endswith("manager") else 0)

    return feat

def extract_key_feat(text):
    text = str(text)
    feat = []
    # 这里不一定都是用 find, 所以不用 key_list 做循环
    # 描述中包含library关键字的软件包
    feat.append(1 if text.find("library") != -1 else 0)
    # 描述中包含libraries关键字的软件包
    feat.append(1 if text.find("libraries") != -1 else 0)
    # 描述中包含command line关键字的软件包
    feat.append(1 if text.find("command") != -1 else 0)
    # 描述中包含utility关键字的软件包
    feat.append(1 if text.find("utility") != -1 else 0)
    # 描述中包含utilities关键字的软件包
    feat.append(1 if text.find("utilities") != -1 else 0)
    # 描述中包含tool关键字的软件包
    feat.append(1 if text.find("tool") != -1 else 0)
    # 描述中包含tools关键字的软件包
    feat.append(1 if text.find("tools") != -1 else 0)
    # 描述中包含application关键字的软件包
    feat.append(1 if text.find("application") != -1 else 0)
    # 描述中包含commands关键字的软件包
    feat.append(1 if text.find("commands") != -1 else 0)
    # 描述中包含service关键字的软件包
    feat.append(1 if text.find("service") != -1 else 0)

    return feat



def name_feat_extractor(label_list):
    print("提取包名特征")
    feats = []
    for label in label_list:
        feats.append(extract_name_feat(label))

    return feats

def keyword_feat_extractor(text_list):
    print("提取描述关键词特征")
    feats = []
    for text in text_list:
        feats.append(extract_key_feat(text))

    return feats

def bow_feat_extractor(text_list):
    # 建立词袋模型
    print("建立词袋模型")
    cv = CountVectorizer(min_df=5, binary=True, stop_words="english")
    cv_matrix = cv.fit_transform(text_list)
    word_vecs = cv_matrix.toarray()
    bag = cv.get_feature_names_out()
    print("词袋大小： {}".format(len(bag)))

    with open("../output/wordbag.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(bag)

    count_list = [list(vec).count(1) for vec in word_vecs]

    print("词袋模型建立完毕，词袋大小为: {}, 单词出现平均占比: {}".format(len(word_vecs[0]), np.mean(count_list) / len(word_vecs[0])))

    return word_vecs

def tfidf_feat_extractor(text_list):
    # 建立词袋模型
    print("开始提取 TF-IDF 特征")
    cv = CountVectorizer(min_df= 5, binary=False, stop_words="english")
    cv_matrix = cv.fit_transform(text_list)
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(cv_matrix)
    weights=tfidf.toarray() # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重 

    return weights
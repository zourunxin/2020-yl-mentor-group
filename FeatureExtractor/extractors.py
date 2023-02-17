import pdb
import sys

from utils.CommonUtils import convert_label

sys.path.append("../")
import csv
import numpy as np
import pandas as pd
import utils.NLPUtils as NLPUtils

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from gensim.models.word2vec import Word2Vec
import pdb
from utils.FileUtil import xlrd_reader, csv_reader

def _extract_name_feat(label):
    label = str(label)
    feat = []
    # 以lib开头的软件包，如libXXX
    # feat.append(1 if label.startswith("lib") else 0)
    # 以lib结尾的软件包，如XXXlib
    feat.append(1 if label.find("lib") != -1 else 0)
    # 以libs结尾的软件包，如XXX-libs
    # feat.append(1 if label.endswith("libs") else 0)
    # 以devel结尾的软件包，如XXX-devel
    feat.append(1 if label.endswith("devel") else 0)
    # 以static结尾的软件包，如XXX-static
    feat.append(1 if label.endswith("static") else 0)
    # 以perl-为前缀的软件包，如perl-XXX
    feat.append(1 if label.find("perl") != -1 else 0)
    # 以python-为前缀的软件包，如python-XXX
    feat.append(1 if label.find("python") != -1 else 0)
    # 以-java为后缀的软件包，如XXX-java
    feat.append(1 if label.find("java") != -1 else 0)
    # 以-headers作为后缀的软件包，如XXX-headers
    feat.append(1 if label.endswith("headers") else 0)
    # 以-api为后缀的软件包，如XXX-api
    feat.append(1 if label.find("api") != -1 else 0)
    # 以tool结尾的软件包，如XXXtool或XXX-tool
    feat.append(1 if label.find("tool") != -1 else 0)
    # 以tools结尾的软件包，如XXXtools或XXX-tools
    # feat.append(1 if label.endswith("tools") else 0)
    # # 以toolset结尾的软件包，如XXXtoolset或XXX-toolset
    # feat.append(1 if label.endswith("toolset") else 0)
    # util作为中间字符串的软件包，如XXX-util-XXX
    feat.append(1 if label.find("util") != -1 else 0)
    # 以util结尾的软件包，如XXXutil或XXX-util
    # feat.append(1 if label.endswith("util") else 0)
    # 以utils结尾的软件包，如XXXutils或XXX-utils
    # feat.append(1 if label.endswith("utils") else 0)
    # 以utility结尾的软件包，如XXXutility或XXX-utility
    # feat.append(1 if label.endswith("utility") else 0)
    # 以progs结尾的软件包，如XXX-progs
    feat.append(1 if label.endswith("progs") else 0)
    # 以-cli结尾的软件包，如XXX-cli
    feat.append(1 if label.find("cli") != -1 else 0)
    # 以-client结尾的软件包，如XXX-client
    # feat.append(1 if label.endswith("client") else 0)
    # 以cmd结尾的软件包，如XXXcmd或XXX-cmd
    feat.append(1 if label.find("cmd") != -1 else 0)
    # 以ctl结尾的软件包，如XXXctl
    feat.append(1 if label.find("ctl") != -1 else 0)
    # 以server结尾的软件包，如XXXserver或XXX-server
    # feat.append(1 if label.endswith("server") else 0)
    # 以service结尾的软件包，如XXXservice或XXX-service
    # feat.append(1 if label.endswith("service") else 0)
    # 以serv结尾的软件包，如XXXserv
    feat.append(1 if label.find("serv") != -1 else 0)
    # 以manager结尾的软件包，如XXXmanager或XXX-manager
    feat.append(1 if label.find("manager") != -1 else 0)

    return feat

def _extract_key_feat(text):
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
    # 描述中包含deamon关键字的软件包
    feat.append(1 if text.find("deamon") != -1 else 0)
    feat.append(1 if text.find("dictionary") != -1 else 0)
    feat.append(1 if text.find("driver") != -1 else 0)
    feat.append(1 if text.find("runtime") != -1 else 0)
    feat.append(1 if text.find("function") != -1 else 0)
    feat.append(1 if text.find("performance") != -1 else 0)
    feat.append(1 if text.find("manager") != -1 else 0)
    feat.append(1 if text.find("plugin") != -1 else 0)
    feat.append(1 if text.find("network") != -1 else 0)
    feat.append(1 if text.find("java") != -1 else 0)
    feat.append(1 if text.find("python") != -1 else 0)
    feat.append(1 if text.find("implement") != -1 else 0)
    return feat


def name_feat_extractor(label_list):
    print("提取包名特征")
    feats = []
    for label in label_list:
        feats.append(_extract_name_feat(label))

    return feats

def keyword_feat_extractor(text_list):
    print("提取描述关键词特征")
    feats = []
    for text in text_list:
        feats.append(_extract_key_feat(text))

    return feats

def meta_feat_extractor(_df_data, df_edges):
    df_data = _df_data
    print("提取元特征")
    in_list = list(df_edges['in'])
    out_list = list(df_edges['out'])
    df_data["processed_summary"] = df_data["summary"].apply(lambda x: NLPUtils.preprocess_text(x))
    df_data["processed_description"] = df_data["description"].apply(lambda x: NLPUtils.preprocess_text(x))
    df_data["summary_len"] = df_data["processed_summary"].apply(lambda x: len(x.split(" ")))
    df_data["description_len"] = df_data["processed_description"].apply(lambda x: len(x.split(" ")))
    df_data["indegree"] = df_data["name"].apply(lambda x: in_list.count(x))
    df_data["outdegree"] = df_data["name"].apply(lambda x: out_list.count(x))
    df_data["degree"] = df_data[['indegree', 'outdegree']].apply(lambda x: x['indegree'] + x['outdegree'], axis=1)
    df_data["name_len"] = df_data["name"].apply(lambda x: len(x))

    df_feat = df_data[["summary_len", "description_len", "indegree", "outdegree", "degree", "name_len"]]
    
    return df_feat.values.tolist()

def unique_feat_extractor(text_list, max_features=3000, speci_layer='all'):
    """
    核心：'text + text'    aa aa   cnt=2
    系统：'text + text'    bb bb bb   cnt=3
    应用：'text + text'
    其它：'text + text'
    """
    def get_fixed_num_word(layer_text, layer_cnt=0.8):
        feature_names = tv.get_feature_names_out()
        X = tfidf.toarray()
        unique_word = []
        for layer, line in zip(layer_text.keys(), X):
            layer_unique_word = [a[1] for a in sorted([(x, feature_names[i]) for i, x in enumerate(line) if x != 0], reverse=True)]
            # pdb.set_trace()
            unique_word.extend(layer_unique_word[:int(len(layer_unique_word) * layer_cnt)])
            print('{}层的 unique word 数量有: {}'.format(layer, len(layer_unique_word)))
            print('{}层的 unique word 分别是: {}'.format(layer, layer_unique_word[:int(len(layer_unique_word) * layer_cnt)]))
        return unique_word

    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/datasource_1228_rpm.csv')
    layer_text = {}
    for line in reader:
        layer = convert_label(line[1], mode='layer')
        text = " ".join(set(s.lower() for s in NLPUtils.preprocess_text(line[2]).split(' ')))
        lText = layer_text.get(layer, "") + " " + text
        layer_text[layer] = lText
    # 建立词袋模型，获得 unique word
    print("开始提取 unique word 特征(每个类的预料为一个文档)")
    tv = TfidfVectorizer(max_features=max_features, max_df=1, stop_words="english")
    tfidf = tv.fit_transform(layer_text.values())
    unique_word = get_fixed_num_word(layer_text)
    # unique_word = tv.get_feature_names()

    print("提取分层人工特征")
    feats = []
    for text in text_list:
        feat = []
        for word in unique_word:
            feat.append(1 if word in text.split(' ') else 0)
        feats.append(feat)
    return feats


def bow_feat_extractor(text_list):
    # 建立词袋模型
    print("建立词袋模型")
    cv = CountVectorizer(max_features=9999, binary=False, stop_words="english")
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

def tfidf_feat_extractor(text_list, label_list=None, feature_num=1000):
    '''
     传入顺序一致的text列表 和 one-hot 形式的标签列表
    '''
    # 建立 tf-idf 特征
    # print("开始提取 TF-IDF 特征(每个包的语料为一个文档)")
    # tv = TfidfVectorizer(max_features=9999, stop_words="english")
    # tfidf = tv.fit_transform(text_list)
    # skb = SelectKBest(chi2, k=feature_num)# 选择 k 个最佳特征
    # tfidf = skb.fit_transform(tfidf, np.argmax(label_list, axis=1))
    #
    # return tfidf.toarray()
    # 建立词袋模型
    print("开始提取 TF-IDF 特征(每个包的语料为一个文档)")
    tv = TfidfVectorizer(max_features=feature_num, stop_words="english")
    tfidf = tv.fit_transform(text_list)
    weights=tfidf.toarray() # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    return weights


def tfidf_class_feat_extractor(label_list, text_list):
    '''
    每个类型的语料为一个文档, 传入顺序一致的标签列表和text列表
    '''
    # 建立词袋模型
    print("开始提取 TF-IDF 特征（每个类型的语料为一个文档")
    print("data len: {}".format(len(label_list)))
    corpus_map = {}
    for i, label in enumerate(label_list):
        if not label in corpus_map:
            corpus_map[label] = ""
        corpus_map[label] = corpus_map[label] + " " + text_list[i]


    label_id_map = {}
    corpus = []

    for i, key in enumerate(corpus_map.keys()):
        label_id_map[key] = i
        corpus.append(corpus_map[key])

    vectorizer = CountVectorizer(max_features=1000, stop_words="english")
    label_matrix = vectorizer.fit_transform(corpus)
    word_list = vectorizer.get_feature_names_out()
    label_matrix = label_matrix.toarray()

    print(label_matrix.shape)
    features = []

    for i, text in enumerate(text_list):
        words = text.split(" ")
        feature = []
        for j, word in enumerate(word_list):
            if word in words:
                feature.append(label_matrix[label_id_map[label_list[i]]][j])
            else:
                feature.append(0)
        features.append(feature)
    
    return np.array(features, dtype=np.float32)


def word2vec_2d_extractor(model_dir, texts, padding=False, max_length=128):
    '''
    model_dir: 预训练好的 model
    texts: 文本 list of list 形式, 需要做分词处理
    padding: 是否为固定长度矩阵，多截断，少补齐
    max_length: 在 padding 为 True 使设置句子的截断长度
    '''
    print("通过word2vec进行向量化, 模型路径: {}".format(model_dir))
    model = Word2Vec.load(model_dir)

    def embedding_text(text):
        feat_matrix = []
        for word in text:
            if word in model.wv:
                feat_matrix.append(model.wv[word])
            else:
                feat_matrix.append(np.zeros((model.vector_size)))

        if padding:
            if len(feat_matrix) > max_length:
                feat_matrix = feat_matrix[0: max_length]
            else:
                for i in range(max_length - len(feat_matrix)):
                    feat_matrix.append(np.zeros((model.vector_size)))
        return feat_matrix

    features = [embedding_text(text) for text in texts]

    return np.array(features)
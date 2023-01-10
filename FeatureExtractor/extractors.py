import sys
sys.path.append("../")
import csv
import numpy as np
import utils.NLPUtils as NLPUtils

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from gensim.models.word2vec import Word2Vec

def _extract_name_feat(label):
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

def bow_feat_extractor(text_list):
    # 建立词袋模型
    print("建立词袋模型")
    cv = CountVectorizer(max_features=1000, binary=False, stop_words="english")
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
    print("开始提取 TF-IDF 特征(每个包的预料为一个文档)")
    tv = TfidfVectorizer(max_features=1000, stop_words="english")
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
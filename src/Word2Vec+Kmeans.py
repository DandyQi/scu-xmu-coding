import numpy as np
import csv
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from sklearn import metrics


embedding_size = 100


# 计算针对每个文档的词嵌入表示的平均值，作为doc的向量表示
def get_input_matrix(filepath, embeddingPath):
    res = []
    wordDict = []
    for row in csv.reader(open(embeddingPath)):
        # print(row, type(row))
        res1 = []
        for x in row:
            if row.index(x) == 0:
                wordDict.append(x)
            else:
                res1.append(float(x))
        res.append(res1)
    # print(type(wordDict), len(wordDict))
    file = open(filepath, 'r', encoding='utf-8')
    line = file.readlines()
    line_new = []
    for line0 in line:
        # print(line0[0])
        line0 = line0.strip().split(' ')
        # print(line0)
        line_new.append(line0)
    # print line_new
    doc_embedding = np.zeros(shape=(len(line_new), embedding_size))
    for idx, i in enumerate(line_new):
        word_list = i
        # print(word_list)
        vector_empression = np.zeros(shape=embedding_size)
        for word in word_list:
            order = wordDict.index(word)
            vector_empression += res[order]
        vector_empression /= len(word_list)
        doc_embedding[idx] = vector_empression
    file.close()
    return doc_embedding


wordFile = 'comment.txt'
embeddings_save_path = '../results/Word2Vec_embeddings(100,2,2).csv'
doc_embedding = get_input_matrix(wordFile, embeddings_save_path)
# print(doc_embedding, np.shape(doc_embedding))


# 聚为33类
Kmeans_every_cluster_path = '../results/Word2Vec+Kmeans(100,2,2,33).txt'
num_clusters = 33
K_model = KMeans(n_clusters=num_clusters, max_iter=500)
s = K_model.fit(doc_embedding)
# print(s)
# print('Predict Label:', K_model.labels_, np.shape(K_model.labels_))


# 把每一类聚类结果写入txt中
def write_everyone_cluster(rfilepath, wfilepath):
    file_in = open(rfilepath, 'r', encoding='utf_8')
    line = file_in.readlines()
    file_out = open(wfilepath, 'w', encoding='utf-8')
    i = 1
    s = ''
    while i <= num_clusters:
        s += '\n\n' + '=========Cluster' + str(i) + ':=========\n'
        label_indexs = np.where(K_model.labels_ == i - 1)[0]
        for j in label_indexs:
            s += line[j]
        i += 1
    file_out.write(s)
    file_out.flush()
    file_out.close()


'''
write_everyone_cluster(wordFile, Kmeans_every_cluster_path)
print(K_model.inertia_)
'''
XX = doc_embedding
YY = K_model.labels_


# 可视化聚类结果
def plot_Kmeans(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    colors = ['#00FFFF', '#7FFFD4', '#000000', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF', '#B22222', '#228B22', '#FF00FF', '#FFD700', '#DAA520', '#808080', '#008000', '#ADFF2F', '#FF69B4', '#CD5C5C', '#4B0082', '#F0E68C', '#7CFC00', '#ADD8E6', '#F08080', '#90EE90', '#FFB6C1', '#FFA07A', '#20B2AA', '#87CEFA', '#778899', '#B0C4DE', '#00FF00', '#32CD32', '#FF00FF', '#800000', '#66CDAA', '#0000CD', '#BA55D3', '#9370DB', '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585', '#191970', '#000080', '#808000', '#6B8E23', '#FFA500', '#FF4500', '#DA70D6', '#98FB98', '#AFEEEE', '#DB7093', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD', '#B0E0E6', '#800080', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072', '#FAA460', '#2E8B57', '#A0522D', '#87CEEB', '#6A5ACD', '#708090', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE', '#FFFF00', '#9ACD32']  # 鲜艳的颜色列表
    colorlist = np.random.choice(colors, num_clusters, replace=False)
    plt.figure(figsize=(12, 12))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(YY[i] + 1),
                 color=colorlist[YY[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    plt.savefig('../results/Word2Vec+Kmeans(100,1,33).eps')  # skypewindow取1


tsne = TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(doc_embedding)
plot_Kmeans(X_tsne)

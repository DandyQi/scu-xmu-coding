import numpy as np
import csv
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import metrics


embedding_size = 200


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
    # print(res, np.shape(res), wordDict, len(wordDict))
    file = open(filepath, 'r', encoding='utf-8')
    line = file.readlines()
    line_new = []
    for line0 in line:
        # print(line0[0])
        line_cluster = int(line0[0])  # 取第一个字符，并从str转数字形
        line0 = line0.strip().split('\t')
        if line_cluster == 4:
            line_new.append(line0[1:])
    # print(line_new)
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


rawFile = '../data/seg_marked_xiaoxiangquan_4classes_new.txt'
embeddings_save_path = '../results/Word2Vec_embeddings(200,2,2).csv'
doc_embedding = get_input_matrix(rawFile, embeddings_save_path)
print(doc_embedding, np.shape(doc_embedding))

'''
# 将矩阵写入csv为了DP-Means
wrtr = csv.writer(open("./DP-Means/input/embedding_xiaoxiangquan.csv", "w", newline=''))
for x in doc_embedding:
    wrtr.writerow(x)
'''


def AMIfindK(filepath):
    file_in = open(filepath, 'r', encoding='utf-8')
    line = file_in.readlines()
    labels = []  # 用来存放raw标签
    trueLabel = []  # 用来存放归一化标签 0，1，2，……
    for eachLine in line:
        numb = eachLine.split('\t')[0]
        if numb not in labels:
            labels.append(numb)
        trueLabel.append(labels.index(numb))
    num_clusters = range(10, 80)
    modelList = []
    AMIList = []
    for k in num_clusters:
        K_model = KMeans(n_clusters=k, max_iter=500)
        K_model.fit(doc_embedding)
        modelList.append(K_model)
        AMI = metrics.adjusted_mutual_info_score(trueLabel, K_model.labels_)
        AMIList.append(AMI)
    maxAMI = max(AMIList)
    bestID = np.where(AMIList == maxAMI)[0][0]
    print(bestID)
    bestK = num_clusters[bestID]
    # bestmodel = modelList[bestID]
    print('BestK:', bestK, '\n', 'BestAMI:', maxAMI)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(num_clusters, AMIList, marker='o')
    ax.set_xlabel("Number of K")
    ax.set_ylabel("AMI Score")
    plt.show()


def RSSfindK():
    RSSList = []
    num_clusters = range(20, 51)
    for k in num_clusters:
        RSSList_i = []
        for i in range(0, 50):
            K_model = KMeans(n_clusters=k, max_iter=500, init='random')
            K_model.fit(doc_embedding)
            RSS = K_model.inertia_
            RSSList_i.append(RSS)
        minRSS = min(RSSList_i)
        # print(RSSList_i, minRSS)
        RSSList.append(minRSS)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(num_clusters, RSSList, marker='o')
    ax.set_xlabel("Number of K")
    ax.set_ylabel("RSS")
    plt.savefig('../temp/Word2Vec200_RSS2.eps')
    plt.show()


input_text_path = '../data/marked_xiaoxiangquan_4classes_new_for_writing.txt'
# AMIfindK(input_text_path)
RSSfindK()

'''
# 原来的code
# Kmeans_text_path = '../results/Word2Vec+Kmeans(100,2,2,29)_heuristic.txt'
Kmeans_text_path = '../results/Word2Vec+Kmeans(100,2,2,33)_3text_small.txt'
num_clusters = 33
K_model = KMeans(n_clusters=num_clusters, max_iter=500)
s = K_model.fit(doc_embedding)
print(s)
print('Predict Label:', K_model.labels_)


# 把聚类结果写入txt中
def write_cluster(filepath):
    file_in = open(filepath, 'r', encoding='utf-8')
    line = file_in.readlines()
    labels = []  # 用来存放raw标签
    trueLabel = []  # 用来存放归一化标签 0，1，2，……
    for eachLine in line:
        numb = eachLine.split('\t')[0]
        if numb not in labels:
            labels.append(numb)
        trueLabel.append(labels.index(numb))
    print(labels, len(labels), trueLabel, len(trueLabel))
    ARI = metrics.adjusted_rand_score(trueLabel, K_model.labels_)
    NMI = metrics.normalized_mutual_info_score(trueLabel, K_model.labels_)
    AMI = metrics.adjusted_mutual_info_score(trueLabel, K_model.labels_)
    V = metrics.v_measure_score(trueLabel, K_model.labels_)
    FM = metrics.fowlkes_mallows_score(trueLabel, K_model.labels_)
    SS = metrics.silhouette_score(doc_embedding, K_model.labels_, metric='euclidean')
    CH = metrics.calinski_harabaz_score(doc_embedding, K_model.labels_)
    print('ARI: ', ARI, '\n', 'NMI: ', NMI, '\n', 'AMI: ', AMI, '\n', 'V-measure: ', V, '\n', 'Fowlkes-Mallows Scores: ', FM, '\n', 'Silhouette Coefficient: ', SS, '\n', 'Calinski-Harabaz Index: ', CH)
    file_out = open(Kmeans_text_path, 'w', encoding='utf-8')
    i = 1
    s = 'ARI: ' + str(ARI) + '\n' + 'NMI: ' + str(NMI) + '\n' + 'AMI: ' + str(AMI) + '\n' + 'V-measure: ' + str(V) + '\n' + 'Fowlkes-Mallows Scores: ' + str(FM) + '\n' + 'Silhouette Coefficient: ' + str(SS) + '\n' + 'Calinski-Harabaz Index: ' + str(CH)
    while i <= num_clusters:
        s += '\n\n' + '==========Cluster' + str(i) + ':==========\n'
        labels_indexs = np.where(K_model.labels_ == i - 1)[0]  # where用来索引
        for j in labels_indexs:
            s += line[j]
        i += 1
    file_out.write(s)
    file_in.close()
    file_out.flush()
    file_out.close()


write_cluster(input_text_path)
print(K_model.inertia_)  # Sum of squared distances of samples to their closest cluster center
'''

'''
# B 是34维 x 10986
XX = doc_embedding
# result 是所聚的label
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
    plt.savefig('../results/Word2Vec+Kmeans/K/Word2Vec(100,1,40).png')


tsne = TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(doc_embedding)
plot_Kmeans(X_tsne)
'''

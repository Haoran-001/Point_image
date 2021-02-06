import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def cal_ap(label, indice, gallery_label):
    precision = []
    recall = []
    hit = 0
    for i in range(len(indice)):
        if gallery_label[int(indice[i])] == label:
            hit += 1
            precision.append(hit / (i+1.))
    if hit == 0:
        return 0.
    return np.mean(precision)


def pr(label, indice, gallery_label):
    precision = []
    recall = []
    recall_i = [10*i for i in range(11)]
    TP_total = np.where(qL==label)[1].shape[0]
    hit = 0
    for i in range(len(indice)):
        if gallery_label[int(indice[i])] == label:
            hit += 1
            if int((hit/TP_total)*100) in recall_i:
                precision.append(hit / (i+1.))
    if hit == 0:
        return 0.
    return precision


query = sio.loadmat('MVCNN_retrieval/query.mat')
qF = query['features']
qL = query['labels']

gallery = sio.loadmat('MVCNN_retrieval/gallery.mat')
gF = gallery['features']
gL = gallery['labels']

print(qF.shape, qL.shape, gF.shape, gL.shape)

n_query = qF.shape[0]
print('The num of retrieval model: ', qF.shape[0])
print('The dimension of model feature: ', qF.shape[1])

# Features from PointImage
# Rank = np.argsort(cdist(qF, qF, 'cosine'))
# print(Rank.shape)
# ap_list = []
# for i in range(n_query):
#     label = qL[0][i]
#     ap = cal_ap(label, Rank[i][1:], qL[0])
#     ap_list.append(ap)
#
# mAP = np.mean(ap_list)
# print('4096 dimension---mAP', mAP)

# Reducing dimension
# PCA
# pca = PCA(n_components=128)
# qF_pca = pca.fit(qF).transform(qF)
#
# ap_pca = []
# Rank_pca = np.argsort(cdist(qF_pca, qF_pca, 'cosine'))
# for i in range(n_query):
#     label = qL[0][i]
#     ap = cal_ap(label, Rank_pca[i][1:], qL[0])
#     ap_pca.append(ap)
#
# mAP_pca = np.mean(ap_pca)
# print('The dimension reduced by PAC: ', qF_pca.shape[1], '\npca_mAP', mAP_pca)

# # PCA+Whiten
# pca = PCA(n_components=32, svd_solver='auto', whiten=False)
# qF_pca = pca.fit(qF).transform(qF)
#
# ap_pca = []
# Rank_pca = np.argsort(cdist(qF_pca, qF_pca, 'cosine'))
# for i in range(n_query):
#     label = qL[0][i]
#     ap = cal_ap(label, Rank_pca[i][1:], qL[0])
#     ap_pca.append(ap)
#
# mAP_pca = np.mean(ap_pca)
# print('The dimension reduced by PAC: ', qF_pca.shape[1], '\npca_mAP', mAP_pca)

### LDA
lda = LinearDiscriminantAnalysis(n_components=39)
qF_lda = lda.fit(gF, gL[0]).transform(qF)
sio.savemat('LDA_39.mat', {'features': qF_lda})
print(qF_lda.shape)

ap_lda = []
Rank_lda = np.argsort(cdist(qF_lda, qF_lda, 'cosine'))

for i in range(n_query):
    label = qL[0][i]
    ap = cal_ap(label, Rank_lda[i][1:], qL[0])
    ap_lda.append(ap)

mAP_lda = np.mean(ap_lda)
print('The dimension of reduced by LDA: ', qF_lda.shape[1], '\nlda_mAP:', mAP_lda)

p_list = np.zeros([n_query, 10])

for i in range(n_query):
    label = qL[0][i]
    p = pr(label, Rank_lda[i], qL[0])
    p_list[i] = p

print(np.mean(p_list, 0))


# SVD
# import tensorflow as tf
# def post_processing(feat, dim=512):
#     s, u, v = tf.svd(feat)
#     feat_svd = tf.transpose(v[:dim, :]) # (b, dim)?
#     return feat_svd
#
#
# dim = 256
# qF_svd = post_processing(qF, dim)
# sess = tf.Session()
# qF_svd = sess.run(qF_svd)
# print(qF_svd.shape)
#
# ap_svd = []
# Rank_svd = np.argsort(cdist(qF_svd, qF_svd, 'cosine'))
#
# for i in range(n_query):
#     label = qL[0][i]
#     ap = cal_ap(label, Rank_svd[i][1:], qL[0])
#     ap_svd.append(ap)
#
# mAP_svd = np.mean(ap_svd)
# print('The dimension of reduced by SVD: ', qF_svd.shape[1], '\nsvd_mAP', mAP_svd)

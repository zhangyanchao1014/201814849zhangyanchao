# coding: utf-8
import sklearn
import json
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

max_df=0.5
min_df=2
max_features=1000
epoch=200

lines=[]
labels=[]
with open('Tweets.txt','r') as f:
    for line in f:
        line=line.rstrip()
        line=line[1:-1]
        # print(line)
        s=line.split('"')
        lines.append(s[3])
        labels.append(int(s[6][2:]))

length=len(lines)
ratio=0.8
train_len=int(length*0.8)
test_len=length-train_len
labels=np.asarray(labels)
class_num=np.unique(labels).shape[0]
print(class_num)

vectorizer=TfidfVectorizer(max_df=max_df,min_df=min_df,max_features=max_features,use_idf=True,stop_words='english') # tf-idf
X=vectorizer.fit_transform(lines)

# print(type(X))
print('n_samples:{} n_features:{}'.format(X.shape[0],X.shape[1]))
# print(X[120])

index=np.random.permutation(length)
X=X[index]
X=X.toarray()
labels=labels[index]

# train_x=X[:train_len]
# train_y=labels[:train_len]
#
# test_x=X[train_len:]
# test_y=labels[train_len:]

# KMeans
km=KMeans(n_clusters=class_num)
km.fit(X)
pred_y=km.labels_

nmi=normalized_mutual_info_score(labels,pred_y)
print('KMeans NMI:{:.4f}'.format(nmi))


# AffinityPropagation
affinity_propagation=AffinityPropagation(damping=0.9,preference=-1)
affinity_propagation.fit(X)
pred_y=affinity_propagation.labels_

nmi=normalized_mutual_info_score(labels,pred_y)
print('AffinityPropagation NMI:{:.4f}'.format(nmi))


# Mean-shift
bandwidth=estimate_bandwidth(X,quantile=0.2)
mean_shift=MeanShift(bandwidth=0.8,bin_seeding=True)
mean_shift.fit(X)
pred_y=mean_shift.labels_

nmi=normalized_mutual_info_score(labels,pred_y)
print('Mean-shift NMI:{:.4f}'.format(nmi))


# Spectral clustering
spectral=SpectralClustering(n_clusters=class_num)
spectral.fit(X)
pred_y=spectral.labels_

nmi=normalized_mutual_info_score(labels,pred_y)
print('Spectral Clustering NMI:{:.4f}'.format(nmi))



connectivity=kneighbors_graph(X,n_neighbors=200,include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)
# Ward hierarchical clustering
ward=AgglomerativeClustering(n_clusters=class_num,linkage='ward',connectivity=connectivity)
ward.fit(X)
pred_y=ward.labels_
nmi=normalized_mutual_info_score(labels,pred_y)
print('Ward hierarchical clustering: {:.4f}'.format(nmi))



# Agglomerative clustering
agglomerative=AgglomerativeClustering(n_clusters=class_num,linkage='average',connectivity=connectivity)
agglomerative.fit(X)
pred_y=agglomerative.labels_
nmi=normalized_mutual_info_score(labels,pred_y)
print('Agglomerative clustering (average): {:.4f}'.format(nmi))



agglomerative=AgglomerativeClustering(n_clusters=class_num,linkage='complete',connectivity=connectivity)
agglomerative.fit(X)
pred_y=agglomerative.labels_
nmi=normalized_mutual_info_score(labels,pred_y)
print('Agglomerative clustering (complete): {:.4f}'.format(nmi))


# # DBSCAN
dbscan=DBSCAN(eps=0.9,min_samples=1,algorithm='auto')
dbscan.fit(X)
pred_y=dbscan.labels_

nmi=normalized_mutual_info_score(labels,pred_y)
print('DBSCAN NMI:{:.4f}'.format(nmi))



# # Gaussian mixtures
gmm=GaussianMixture(n_components=class_num)
gmm.fit(X)
pred_y=gmm.predict(X)

nmi=normalized_mutual_info_score(labels,pred_y)
print('Gaussian Mixture NMI:{:.4f}'.format(nmi))


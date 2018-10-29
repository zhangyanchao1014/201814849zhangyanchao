# coding : utf-8
import numpy as np
import os
path='20news-18828'

record_filter=np.load('record_filter.npz')
words=record_filter['words']
frequency=record_filter['frequency']
df=record_filter['df']
tf=record_filter['tf']

doc_list=[]
doc_path=os.path.join(path,'20news-18828')
for doc_dir in os.listdir(doc_path):
    doc_dir=os.path.join(doc_path,doc_dir)
    for document in os.listdir(doc_dir):
        document_name=os.path.join(doc_dir,document)
        doc_list.append(document_name)

np.save('doc_list.npy',doc_list)

print(words[20])
print(doc_list[17391])

word_size=len(words)
doc_size=len(doc_list)
print(doc_size)

tf_idf_matrix=np.zeros(shape=[word_size,doc_size])
for i in range(word_size):
    word=words[i]
    dfi=df[i]
    tfi=tf[i]

    for j in range(doc_size):
        doc=doc_list[j]
        if doc in tfi.keys():
            tf_word_doc=1+np.log(tfi[doc]) # tf
            # tf_word_doc=tfi[doc]
            idfi=np.log(doc_size/(dfi)) # idf
            tf_idf=tf_word_doc*idfi # calculate tf-idf
        else:
            tf_idf=0.0

        tf_idf_matrix[i,j]=tf_idf

np.save('tf_idf.npy',tf_idf_matrix)

feature_dir='feature'
if not os.path.exists(feature_dir):
    os.mkdir(feature_dir)

# save the vector space model
for i in range(doc_size):
    document_name=doc_list[i]
    s=document_name.split('\\')
    save_dir=os.path.join(feature_dir,s[2])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path=os.path.join(save_dir,s[3]+'.npy')
    np.save(save_path,tf_idf_matrix[:,i])
# coding : utf-8
import numpy as np
from sklearn import *
from sklearn.model_selection import StratifiedShuffleSplit
import os

path = '20news-18828\\20news-18828'
ratio=0.8


def train_test(doc_list, label_list, ratio):
    doc_list = np.asarray(doc_list)
    label_list = np.asarray(label_list)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-ratio, random_state=0)
    for train_index, test_index in sss.split(doc_list, label_list):
        train_doc_list = list(doc_list[train_index])
        train_label_list = list(label_list[train_index])
        test_doc_list = list(doc_list[test_index])
        test_label_list = list(label_list[test_index])

    with open('train.txt', 'w') as f:
        for i in range(len(train_label_list)):
            line = train_doc_list[i] + ' ' + train_label_list[i] + '\n'
            f.write(line)

    with open('test.txt', 'w') as f:
        for i in range(len(test_label_list)):
            line = test_doc_list[i] + ' ' + test_label_list[i] + '\n'
            f.write(line)

def class_no():
    class_id=[]
    doc_class=[]
    i=0
    for doc_dir in os.listdir(path):
        class_id.append(doc_dir)
        doc_dir=os.path.join(path,doc_dir)
        for doc in os.listdir(doc_dir):
            doc=os.path.join(doc_dir,doc)
            doc_class.append((doc,i))
        i+=1

    with open('class_id.txt','w') as f:
        for i in range(len(class_id)):
            f.write(str(i)+' '+class_id[i]+'\n')

    with open('doc_class.txt','w') as f:
        for item in doc_class:
            doc=item[0]
            c=item[1]
            f.write(doc+' '+str(c)+'\n')


if __name__ == '__main__':
    class_no()
    doc_list=[]
    label_list=[]

    with open('doc_class.txt','r') as f:
        for line in f:
            line=line.rstrip()
            s=line.split()
            doc_list.append(s[0])
            label_list.append(s[1])

    train_test(doc_list,label_list,ratio)
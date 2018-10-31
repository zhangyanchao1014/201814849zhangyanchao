# coding : utf-8
import os
import numpy as np
path='20news-18828\\20news-18828'
feature_dir='feature'
k=10

def load_feature(train_doc_list,test_doc_list):
    train_feature=[]
    test_feature=[]
    for doc in train_doc_list:
        feature_file=os.path.join(feature_dir,doc+'.npy')
        feature=np.load(feature_file)
        train_feature.append(feature)

    for doc in test_doc_list:
        feature_file=os.path.join(feature_dir,doc+'.npy')
        feature=np.load(feature_file)
        test_feature.append(feature)

    # train_feature=np.asarray(train_feature)
    # test_feature=np.asarray(test_feature)

    return train_feature,test_feature



def knn(k,train_doc_list,train_label_list,test_doc_list,test_label_list,cos=False):
    '''
    :param k:
    :param train_doc_list:
    :param train_label_list:
    :param test_doc_list:
    :param test_label_list:
    :return:
    '''

    train_feature,test_feature=load_feature(train_doc_list,test_doc_list)
    train_size=len(train_label_list)
    test_size=len(test_label_list)
    right=0

    for i in range(test_size):
        feature = test_feature[i]
        label=test_label_list[i]
        value={}
        for j in range(train_size):
            feature2=train_feature[j]
            if cos==False: # 欧几里得距离
                d=np.linalg.norm(np.subtract(feature,feature2))
            else: # 余弦相似度
                up=np.sum(np.multiply(feature,feature2))
                down=np.linalg.norm(feature)*np.linalg.norm(feature2)
                d=np.divide(up,down)
                # print(up,down)
            # print(d)
            value[j]=d
        if cos == False:
            value=sorted(value.items(),key=lambda d:d[1],reverse=False) # small -> large
        else:
            value=sorted(value.items(),key=lambda d:d[1],reverse=True) # large ->  small

        class_count={}
        value_k=value[:k]
        # print(value_k)

        for item in value_k:
            doc_id=item[0]
            pred=train_label_list[doc_id]
            if pred in class_count.keys():
                class_count[pred]+=1
            else:
                class_count[pred]=1

        class_count=sorted(class_count.items(),key=lambda d:d[1],reverse=True) # large -> small
        # print(class_count)
        pred_label=class_count[0][0]

        if label == pred_label:
            right+=1
        # print(i,label,pred_label,right)


    accuracy=right/test_size
    print('k={} cos={} accuracy={:.4f}'.format(k,cos,accuracy))


if __name__ == '__main__':
    train_doc_list=[]
    train_label_list=[]
    test_doc_list=[]
    test_label_list=[]

    with open('train.txt','r') as f:
        for line in f:
            line=line.rstrip()
            s=line.split()
            ss=s[0].split('\\')
            train_doc_list.append(ss[2]+'\\'+ss[3])
            train_label_list.append(int(s[1]))

    with open('test.txt','r') as f:
        for line in f:
            line=line.rstrip()
            s=line.split()
            ss=s[0].split('\\')
            test_doc_list.append(ss[2]+'\\'+ss[3])
            test_label_list.append(int(s[1]))

    for i in range(1,k+1):
        knn(i,train_doc_list,train_label_list,test_doc_list,test_label_list,True)


        '''
        k=1 cos=False accuracy=0.6126
        k=2 cos=False accuracy=0.4753
        k=3 cos=False accuracy=0.4748
        k=4 cos=False accuracy=0.4849
        k=5 cos=False accuracy=0.4902
        k=6 cos=False accuracy=0.4904
        k=7 cos=False accuracy=0.4769
        k=8 cos=False accuracy=0.4387
        k=9 cos=False accuracy=0.4241
        k=10 cos=False accuracy=0.4211
        
        
        k=1 cos=True accuracy=0.8893
        k=2 cos=True accuracy=0.8500
        k=3 cos=True accuracy=0.8678
        k=4 cos=True accuracy=0.8731
        k=5 cos=True accuracy=0.8757
        k=6 cos=True accuracy=0.8760
        k=7 cos=True accuracy=0.8752
        k=8 cos=True accuracy=0.8717
        k=9 cos=True accuracy=0.8733
        k=10 cos=True accuracy=0.8728
        
        '''

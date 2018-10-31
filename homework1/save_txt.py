import numpy as np

words=np.load('words.npy')
with open('words.txt','w',encoding='utf-8') as f:
    for word in words:
        f.write(word+'\n')

frequency=np.load('frequency.npy')
print(frequency.shape[0])
with open('frequency.txt','w',encoding='utf-8') as f:
    for i in range(frequency.shape[0]):
        f.write(str(frequency[i])+'\n')

doc_list=np.load('doc_list.npy')
with open('doc_list.txt','w',encoding='utf-8') as f:
    for doc in doc_list:
        f.write(doc+'\n')

df=np.load('df.npy')
with open('df.txt','w',encoding='utf-8') as f:
    for i in range(df.shape[0]):
        f.write(str(df[i])+'\n')

tf=np.load('tf.npy')
print(tf.shape)
with open('tf.txt','w',encoding='utf-8') as f:
    for i in range(tf.shape[0]):
        line="{"
        for item in tf[i].items():
            item0=item[0]
            item1=item[1]
            line+='{}:{} '.format(item0,item1)
        line+='}\n'
        f.write(line)
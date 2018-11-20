# coding: utf-8
import os
import numpy as np
from textblob import *

path = '20news-18828'


def preprocess():
    word = np.load('words.npy')
    doc_dir = os.path.join(path, '20news-18828_preprocess')
    save_dir = os.path.join(path, '20new_18828_preprocess_filter')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for dir in os.listdir(doc_dir):
        save_path = os.path.join(save_dir, dir)
        dir_path = os.path.join(doc_dir, dir)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for document_path in os.listdir(dir_path):
            save_doc_path = os.path.join(save_path, document_path)
            document_path = os.path.join(dir_path, document_path)
            filter_word = []
            with open(document_path, 'r', encoding='latin2') as f:
                for line in f:
                    line = line.rstrip()
                    items = line.split()
                    for item in items:
                        if item in word:
                            filter_word.append(item)

            content = ' '.join(filter_word) + '\n'
            with open(save_doc_path, 'w', encoding='latin2') as f:
                f.write(content)


def load_data():
    train_class_doc = {}
    with open('train.txt', 'r') as f:
        for line in f:
            line = line.rstrip()
            s = line.split()
            label = int(s[1])
            doc = s[0]
            s = doc.split('\\')
            doc = '20news-18828\\20new_18828_preprocess_filter\\' + s[2] + '\\' + s[3]
            if label in train_class_doc.keys():
                train_class_doc[label].append(doc)
            else:
                train_class_doc[label] = [doc]
    test_doc_class = {}
    with open('test.txt', 'r') as f:
        for line in f:
            line = line.rstrip()
            s = line.split()
            label = int(s[1])
            doc = s[0]
            s = doc.split('\\')
            doc = '20news-18828\\20new_18828_preprocess_filter\\' + s[2] + '\\' + s[3]
            test_doc_class[doc] = label
    return train_class_doc, test_doc_class


def pre_calculate(train_class_doc):
    dir = '20news-18828/class_word_info'
    if not os.path.exists(dir):
        os.mkdir(dir)

    for label, doc_list in train_class_doc.items():
        print(label)
        class_word_df = {}
        class_word_frequency = {}
        # i=0
        for doc in doc_list:
            with open(doc, 'r', encoding='latin2') as f:
                content = f.read()
                b = TextBlob(content)
                words = set(b.words)
                for word in words:
                    count = b.word_counts[word]
                    if word not in class_word_frequency.keys():
                        class_word_frequency[word] = count
                    else:
                        class_word_frequency[word] += count

                    if word not in class_word_df.keys():
                        class_word_df[word] = 1
                    else:
                        class_word_df[word] += 1
            # i+=1
            # print(i)
            # print(class_word_df)
        with open(os.path.join(dir, '{}.txt'.format(label)), 'w') as f:
            for word in class_word_df.keys():
                line = word + ' ' + str(class_word_df[word]) + ' ' + str(class_word_frequency[word]) + '\n'
                f.write(line)


def load_info():
    infos = {}
    dir = '20news-18828/class_word_info'
    for info_file in os.listdir(dir):
        info_file_path = os.path.join(dir, info_file)
        print(info_file_path)
        class_info = {}
        class_id = int(info_file.split('.')[0])
        with open(info_file_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                s = line.split()
                class_info[s[0]] = (int(s[1]), int(s[2]))
        infos[class_id] = class_info
    print(len(infos))

    class_df_freq = {}
    for info in infos.items():
        class_id = info[0]
        items = info[1]
        df = 0
        freq = 0
        for item in items.items():
            df += int(item[1][0])
            freq += int(item[1][1])
        class_df_freq[class_id] = (df, freq)

    return infos, class_df_freq


def cal_class_prob(train_class_doc):
    class_prob = []
    sum = 15062
    for class_id, doc_list in train_class_doc.items():
        doc_list_len = len(doc_list)
        prob = doc_list_len / sum
        class_prob.append(prob)
    class_prob = np.asarray(class_prob)
    # print(np.sum(class_prob))
    return class_prob


def bernoullli(test_doc_class, infos, class_df_freq, class_prob):  # 伯努利模型
    class_num = len(class_df_freq)
    test_len = len(test_doc_class)

    corr = 0
    for doc, label in test_doc_class.items():
        with open(doc, 'r', encoding='latin2') as f:
            line = f.read()
        line = line.rstrip()
        items = line.split()
        items = set(items)  # no repeat word
        sentence_len = len(items)
        prob_matrix = np.zeros(shape=[sentence_len, class_num], dtype=np.float64)
        i = 0
        for word in items:
            for j in range(class_num):
                sum_info = class_df_freq[j]
                sum_df = sum_info[0]
                info = infos[j]  # {word:(df,freq)}
                if word in info.keys():
                    df = info[word][0]
                else:
                    df = 0
                prob = np.log((df + 1) / (sum_df + 2))
                # prob = (df + 1) / (sum_df + 2)
                prob_matrix[i, j] = prob
            i += 1
            # print(i)

        prob_matrix=np.sum(prob_matrix,axis=0)
        # print(prob_matrix)
        # print(prob_matrix*class_prob)
        pred=np.argmax(prob_matrix)
        pred1=np.argmax(prob_matrix+np.log(class_prob))
        # print(label,pred,pred1)

        # pred_class = np.zeros(shape=class_num)
        # for j in range(class_num):
        #     pred_prob = prob_matrix[:, j]
        #     prob = 1
        #     for i in range(sentence_len):
        #         prob *= pred_prob[i]
        #     prob *= class_prob[j]
        #     pred_class[j] = prob
        # pred = np.argmax(pred_class)
        # # print(label,pred)

        if pred == label:
            corr += 1
    print(corr)

    print('accuracy:{:.4f}'.format(corr / test_len))


def binomial(test_doc_class, infos, class_df_freq, class_prob):  # 多项式模型
    class_num = len(class_df_freq)
    test_len = len(test_doc_class)

    corr = 0
    for doc, label in test_doc_class.items():
        with open(doc, 'r', encoding='latin2') as f:
            line = f.read()
        line = line.rstrip()
        items = line.split()
        items = (items)  # repeat word
        sentence_len = len(items)
        prob_matrix = np.zeros(shape=[sentence_len, class_num], dtype=np.float64)
        i = 0
        for word in items:
            for j in range(class_num):
                sum_info = class_df_freq[j]
                sum_freq = sum_info[1]
                info = infos[j]  # {word:(df,freq)}
                word_size = len(info)
                if word in info.keys():
                    freq = info[word][1]
                else:
                    freq = 0
                prob = np.log((freq + 1) / (sum_freq + word_size))
                # prob = (freq + 1) / (sum_freq + word_size)
                prob_matrix[i, j] = prob
            i += 1
            # print(i)

        prob_matrix=np.sum(prob_matrix,axis=0)
        # print(prob_matrix)
        # print(prob_matrix*class_prob)
        pred=np.argmax(prob_matrix)
        pred1=np.argmax(prob_matrix+np.log(class_prob))
        # print(label,pred,pred1)

        # pred_class = np.zeros(shape=class_num)
        # for j in range(class_num):
        #     pred_prob = prob_matrix[:, j]
        #     prob = 1
        #     for i in range(sentence_len):
        #         prob *= pred_prob[i]
        #     prob *= class_prob[j]
        #     pred_class[j] = prob
        # print(pred_class)
        # pred = np.argmax(pred_class)
        # # print(label,pred)

        if pred == label:
            corr += 1
    print(corr)

    print('accuracy:{:.4f}'.format(corr / test_len))


def fusion_model(test_doc_class, infos, class_df_freq, class_prob):  # 混合模型
    class_num = len(class_df_freq)
    test_len = len(test_doc_class)

    corr = 0
    for doc, label in test_doc_class.items():
        with open(doc, 'r', encoding='latin2') as f:
            line = f.read()
        line = line.rstrip()
        items = line.split()
        items = set(items)  # not repeat word
        sentence_len = len(items)
        prob_matrix = np.zeros(shape=[sentence_len, class_num], dtype=np.float64)
        i = 0
        for word in items:
            for j in range(class_num):
                sum_info = class_df_freq[j]
                sum_freq = sum_info[1]
                info = infos[j]  # {word:(df,freq)}
                word_size = len(info)
                if word in info.keys():
                    freq = info[word][1]
                else:
                    freq = 0
                prob = np.log((freq + 1) / (sum_freq + word_size))
                # prob = (freq + 1) / (sum_freq + word_size)
                prob_matrix[i, j] = prob
            i += 1
            # print(i)

        prob_matrix = np.sum(prob_matrix, axis=0)
        # print(prob_matrix)
        # print(prob_matrix*class_prob)
        pred = np.argmax(prob_matrix)
        pred1 = np.argmax(prob_matrix + np.log(class_prob))
        # print(label, pred, pred1)

        # pred_class = np.zeros(shape=class_num)
        # for j in range(class_num):
        #     pred_prob = prob_matrix[:, j]
        #     prob = 1
        #     for i in range(sentence_len):
        #         prob *= pred_prob[i]
        #     prob *= class_prob[j]
        #     pred_class[j] = prob
        # pred = np.argmax(pred_class)
        # # print(label,pred)

        if pred == label:
            corr += 1
    print(corr)

    print('accuracy:{:.4f}'.format(corr / test_len))


if __name__ == '__main__':
    # preprocess()
    train_class_doc, test_doc_class = load_data()
    # pre_calculate(train_class_doc)
    class_prob = cal_class_prob(train_class_doc)
    print(class_prob)
    infos, class_df_freq = load_info()  # infos {class_id:[]} [[word,df,freq]]  class_df_freq {class_id:()} (sum_df,sum_freq)
    bernoullli(test_doc_class, infos, class_df_freq, class_prob)
    binomial(test_doc_class, infos, class_df_freq, class_prob)
    fusion_model(test_doc_class, infos, class_df_freq, class_prob)

# -*- coding:utf-8 -*-
import os
import numpy as np
from textblob import *
import time
import string
import re

# BlobText Word Sentence
path = '20news-18828'

word_frequency = {}  # dict{word:word_frequency}
word_df = {}  # dict{word:df}
word_tf = {}  # dict{word:(doc_name,tf)}


def preprocessing(document_name, save_name, stop_words):
    global word_frequency
    global word_df
    global word_tf

    with open(document_name, 'rb') as f:
        content = f.read()
    content = str(content, encoding='latin2')
    # print(content)

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    content = regex.sub('', content)
    s = TextBlob(content)
    # s = TextBlob.correct(s)  # check spelling
    s = TextBlob.lower(s)  # transform upper letters to lower letter

    word_list = []
    word_count = {}
    for word in s.words:
        word = Word.singularize(word)  # pluralize to singularize
        word = Word.lemmatize(word)  # lemma
        if word in stop_words:
            continue

        word_list.append(word)
        if word not in word_count.keys():
            word_count[word] = 1
        else:
            word_count[word] += 1

    for item in word_count.items():
        word = item[0]
        count = item[1]
        # calculate word frequency
        if word not in word_frequency.keys():
            word_frequency[word] = count
        else:
            word_frequency[word] += count

        # calculate word df
        if word not in word_df.keys():
            word_df[word] = 1
        else:
            word_df[word] += 1

        # calculate word tf
        if word not in word_tf.keys():
            word_tf[word] = {document_name: count}
        else:
            word_tf[word][document_name] = count

    result = ' '.join(word_list)

    # save the clean document
    with open(save_name, 'w', encoding='latin2') as f:
        f.write(result)

    # print(word_frequency)


def filter():
    stop_words = []
    with open('stop_words.txt') as f:
        for line in f:
            line = line.rstrip()
            stop_words.append(line)

    doc_path = os.path.join(path, '20news-18828')
    save_path = os.path.join(path, '20news-18828_preprocess')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for doc_dir in os.listdir(doc_path):
        save_dir = os.path.join(save_path, doc_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        doc_dir = os.path.join(doc_path, doc_dir)
        for document in os.listdir(doc_dir):
            save_file = os.path.join(save_dir, document)
            document_name = os.path.join(doc_dir, document)
            # print(doc_dir, document)

            preprocessing(document_name, save_file, stop_words)


def create_word_table(threshold):
    '''
    create word table
    :param threshold:
    :return:
    '''
    record = np.load('record.npz')
    w_f = record['word_frequency'].item()  # dict
    w_df = record['word_df'].item()
    w_tf = record['word_tf'].item()

    w_f_list = sorted(w_f.items(), key=lambda d: d[1], reverse=True)
    words = []
    for item in w_f_list:
        word = item[0]
        count = item[1]
        if count >= threshold:
            words.append(word)

    frequency = []
    dfs = []
    tfs = []
    for word in words:
        frequency.append(w_f[word])
        dfs.append(w_df[word])
        tfs.append(w_tf[word])

    np.savez('record_filter.npz', words=words, frequency=frequency, df=dfs, tf=tfs)


if __name__ == '__main__':
    global word_frequency
    global word_df
    global word_tf

    filter()
    np.savez('record.npz', word_frequency=word_frequency, word_df=word_df, word_tf=word_tf)
    create_word_table(15)

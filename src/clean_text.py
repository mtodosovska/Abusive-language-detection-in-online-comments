import re
from nltk.corpus import stopwords
from nltk import sent_tokenize
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import pickle


def clean_re(comment):
    # print('New lines')
    com = re.sub(r'NEWLINE_TOKEN', ' ', comment)
    comm = re.sub(r'\d+', '.', com)
    return comm


def clean_whitespaces(comment):
    # print('Whitespaces')
    w = re.split(r'\W+', comment)
    return w


# clean stop words and case
def clean_stopwords(comment):
    # print('Stopwords')
    stop_words = set(stopwords.words('english'))
    all_words = []
    for w in comment:
        if w not in stop_words:
            all_words.append((w.lower()))
    return all_words


def split(comment):
    sentences = sent_tokenize(comment)
    # print(sentences)


def clean_comments():
    comments = pd.read_csv('../data/4054689/attack_annotated_comments.tsv', delimiter='\t')
    comments = comments.drop(['year', 'sample', 'split'], axis=1)
    j = 0
    print('Cleaning')
    comments_clean = pd.DataFrame()
    for rev_id, comment in zip(comments.iloc[:, 0], comments.iloc[:, 1]):
        if j % 100 == 0:
            print('Comment:', j, 'of total:', comments.shape[0])
        j += 1
        comm = clean_re(comment)
        words = clean_whitespaces(comm)
        all_words = clean_stopwords(words)
        # stemmed = stem(all_words)
        words_list = pd.DataFrame([])
        words_list = words_list.append([rev_id, all_words]).transpose()
        # words_list
        comments_clean = comments_clean.append(words_list)

    comments_clean.to_csv('../data/comments_clean.csv')


def pickle_comments():
    comments = pd.read_csv('../data/comments_clean.csv', encoding='latin1').drop(['Unnamed: 0'], axis=1)
    comments_cleaned = pd.DataFrame()
    for index, row in comments.iterrows():
        print(index)
        # comment = row.split(' ')
        ls = row[1].split()
        lss = []
        for s in ls:
            lss.append(s.replace('\'', '').replace(',', ''))
        comment = pd.DataFrame([row[0], lss[1:len(lss) - 1]])
        comments_cleaned = comments_cleaned.append(comment.transpose())

    print(comments_cleaned)
    with open("../data/comments_clean.txt", "wb") as fp:  # Pickling
        pickle.dump(comments_cleaned, fp)


pickle_comments()


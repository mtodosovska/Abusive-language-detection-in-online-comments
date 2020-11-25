from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import pickle
from data_manager import DataManager


def get_sentences(comments):
    print('Getting sentences...')
    sentences = []
    i = 0
    for comment in comments:
        print('Sentences:', i)
        i += 1
        sentences.append(" ".join(str(x) for x in comment))

    sent = pd.DataFrame(sentences)
    sent.to_csv('../data/sentences.csv')
    return sentences


def train_model(comments):
    print('Training model...')
    min_word_count = 1  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    print("Training Word2Vec model...")
    model = Word2Vec(comments, workers=num_workers, min_count=min_word_count)
    print(model)

    words = list(model.wv.vocab)
    print(words)

    model.save('model.bin')

    return model


def load_model():
    model = Word2Vec.load('../models/model.bin')
    return model


def get_avg_vec(model, comment, num_features):
    print('Getting average vectors...')
    index2word = set(model.wv.index2word)
    num = 0

    featureVec = np.zeros((num_features,), dtype="float32")
    for word in comment:
        if word in index2word:
            num = num + 1
            featureVec += np.array(model[word])

    featureVec /= num
    return featureVec


def get_vectors(path, comments, num_features, train):
    print('Getting features...')

    if train:
        model = train_model(comments)
    else:
        model = load_model()

    counter = 0
    data = pd.DataFrame()

    for index, row in comments.iterrows():
        if counter % 100 == 0:
            print("Review %d of %d" % (counter, comments.iloc[:, 0].shape[0]))
        counter += 1

        vector = get_avg_vec(model, row.iloc[1], num_features)

        ls = np.zeros(len(vector) + 1)
        ls[0] = row[0]
        ls[1:] = vector
        ft = pd.DataFrame(ls)
        data = data.append(ft.transpose())

    data.to_csv(path)


def get_data(path):
    with open(path, "rb") as fp:  # Unpickling
        data = pickle.load(fp)
    return data


def get_embeddings():
    num_features = 150
    data = DataManager.get_basic_comments()
    get_vectors('../features/embeddings150.csv', data.iloc[:, 3:5], num_features, False)

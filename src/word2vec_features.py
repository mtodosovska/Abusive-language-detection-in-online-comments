from gensim.models import Word2Vec
import logging
import numpy as np
from nltk import sent_tokenize
import pandas as pd
import pickle


def get_sentences(comments):
    print('Getting sentences...')
    # print(comment)
    sentences = []
    i = 0
    for comment in comments:
        print('Sentences:', i)
        i += 1
        # sentences.append(sent_tokenize(comment))
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
    # model = Word2Vec.load("word2vec_model_second.txt")
    model = Word2Vec.load('model.bin')
    return model


def get_avg_vec(model, comment, num_features):
    # print('Getting average vectors...')
    index2word = set(model.wv.index2word)
    num = 0

    featureVec = np.zeros((num_features,), dtype="float32")
    for word in comment:
        if word in index2word:
            num = num + 1
            featureVec += np.array(model[word])

    featureVec /= num
    return featureVec


def get_vectors(comments, num_features):
    print('Getting features...')

    # model = train_model(comments)
    model = load_model()

    # feature_vector = []
    counter = 0
    data = pd.DataFrame()

    for index, row in  comments.iterrows():
        if counter % 100 == 0:
            print("Review %d of %d" % (counter, comments.iloc[:, 0].shape[0]))
        counter += 1
        # feature_vector.append(comment.iloc[0])

        vector = get_avg_vec(model, row.iloc[2], num_features)
        print(vector)

        # TODO - embeddings shouldn't be a list, each embedding should be a column of its own
        row = row.append(pd.DataFrame(['m', vector]))
        comment = pd.DataFrame(row)
        comment = comment.transpose().drop(0, axis=1)
        comment.columns = ['rev_id', 'label', 'words', 'embedding']

        data = data.append(comment)

    with open("data_features.txt", "wb") as fp:  # Pickling
        pickle.dump(data, fp)

def get_data():
    # data = pd.read_csv('../data/data.csv', index_col=0, encoding='latin1')
    with open("../data/data.txt", "rb") as fp:  # Unpickling
        data = pickle.load(fp)
    return data


num_features = 100
data = get_data()
# train_model(data.iloc[:, 2])
get_vectors(data, num_features)

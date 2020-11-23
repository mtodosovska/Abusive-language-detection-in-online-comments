import pandas as pd
import numpy as np
import pickle
import re
import math

from data_manager import DataManager


def delete_empty_rows(data):
    i = 0
    new_data = pd.DataFrame()
    for index, row in data.iterrows():
        print('To delete or not to delete the question is now:', i)
        if len(list(row.iloc[2])) != 0:
            new_data = new_data.append(row)
        i += 1

    print(new_data)
    with open("../data/data_features_clean.txt", "wb") as fp:  # Pickling
        pickle.dump(new_data, fp)
    return new_data


def clean_sentences():
    sentences = DataManager.get_sentences()
    i = 0
    new_data = pd.DataFrame()
    for index, row in sentences.iterrows():
        if i % 1000 == 0:
            print('To delete or not to delete the question is now:', i)
        if isinstance(row[0], float) and math.isnan(row[0]):
            print('Deleting...', i)
        else:
            new_data = new_data.append(row)
        i += 1

    new_data.to_csv('../data/sentences_clean.csv')
    print(new_data)


def get_embeddings():
    data = DataManager.get_data_features_clean()

    data = data[['rev_id', 'label', 'words', 'embedding']]
    data.set_index('rev_id')
    embeddings = data['embedding']
    with open("../features/embeddings.txt", "wb") as fp:  # Pickling
        pickle.dump(embeddings, fp)


def flat_data(data):
    embeddings = data['embedding'].tolist()
    data = data.drop(['embedding'], axis=1)
    data = data.reset_index(drop=True)
    names = list(data)
    embeddings_names = [str('embedding: ' + str(x + 1)) for x in range(0, 100)]
    # names.append(embeddings_names)
    names[len(names):len(names) + 1] = embeddings_names
    print(names)
    new_data = pd.DataFrame()

    for index, row in data.iterrows():
        print(index)
        row = pd.DataFrame(np.append(row.values, embeddings[index])).transpose()
        new_data = new_data.append(row)


    new_data = new_data.reset_index(drop=True)
    new_data.columns = names

    print(new_data)
    new_data.to_csv('../features/data_features_clean_flat.csv', header=False, mode='a')

def flat_data_calling():
    for i in range (0, 115):
        print('-----------', i)
        data = DataManager.get_data_features_clean()
        dt = data.iloc[1000*i:1000*(i+1)]
        del data
        flat_data(dt)

    data = DataManager.get_data_features_clean()
    dt = data.iloc[1000*i:]
    del data
    flat_data(dt)


def separate_features():
    data = DataManager.get_data_features()
    print(data)
    basic = data.iloc[:, 1:3]
    data = data.drop(3, axis=1)
    # n_grams = data.iloc[:, 1:525]
    for i in range(4, 527):
        data = data.drop(i, axis=1)
    data = data.drop(1, axis=1)

    embeddings = data
    print('basic', basic)
    # print('n_grams', n_grams)
    print('embeddings', embeddings)
    # n_grams.to_csv('../features/n_grams.csv')
    embeddings.to_csv('../features/embeddings.csv')
    with open("../features/basic.txt", "wb") as fp:  # Pickling
        pickle.dump(basic, fp)


def sentences_clean():
    comments = DataManager.get_comments()
    sentences = DataManager.get_sentences_clean()
    sent = comments.merge(sentences, how='left', left_index=True, right_index=True).drop('words', axis=1)
    sent.columns = ['rev_id', 'sentence']
    sent.to_csv('../features/sentences_clean.csv', index=False)


def relabel():
    labels = DataManager.get_labels()
    class_0 = labels[labels['label'] == 0].iloc[0:, :]
    class_2 = labels[labels['label'] == 2].append(labels[labels['label'] == 1]).iloc[0:, :]
    class_2.iloc[:, 2] = 0
    class_3 = labels[labels['label'] == 3].append(labels[labels.label == 4]).append(labels[labels.label == 5]).iloc[0:,:]
    class_3.iloc[:, 2] = 1
    labels = class_0.append(class_2).append(class_3)

    labels = labels.sample(frac=1)
    labels = labels.reset_index().drop('level_0', axis=1).drop('index', axis=1)
    labels.to_csv('../features/labels_2_classes.csv', index=False)
    print('-------------Labels done!++++++++++++++')

def clean_whitespaces(comment):
    # print('Whitespaces')
    comment = re.sub(r'[^\w\s]{2,}', '', comment)
    comment = re.sub(r' [^\w\s] ', ' ', comment)
    comment = re.sub(r' {2,}', ' ', comment)

    return comment


def group_data(labels):
    grouped = labels.groupby('rev_id')
    i = 0
    partial = pd.DataFrame()
    print('Total:', len(grouped))
    for name, group in grouped:
        print(i)
        temp = pd.DataFrame(np.zeros(group.shape[0] + 1))
        temp.iloc[0] = group.iloc[0, 0]
        j = 0
        for index, row in group.iterrows():
            temp.iloc[j + 1] = row[1]
            j += 1
        partial = partial.append(temp.transpose())
        i += 1

    print(partial)
    partial.to_csv('../data/partial.csv')


def get_average_scores():
    partial = pd.read_csv('../data/partial.csv').drop(['Unnamed: 0'], axis=1)
    avg_partial = pd.DataFrame()

    length = partial.iloc[0, :].shape[0]
    c = 0
    for index, row in partial.iterrows():
        print(index)
        scores = row.iloc[1: length]
        # avg = int(round(np.mean(scores)))
        avg = np.mean(scores)
        av = 0

        if avg < 0.05:
            av = 0
            c += 1
        elif avg < 0.1:
            av = 1
        elif avg < 0.3:
            av = 2
        elif avg < 0.5:
            av = 3
        elif avg < 0.7:
            av = 4
        else:
            av = 5

        avg_partial = avg_partial.append([np.append(int(row.iloc[0]), av)])
    print(c/115863)

    # avg_partial.to_csv('../data/partial_avg.csv')

    avg_partial.to_csv('../features/labels.csv')

    print(avg_partial)

    return avg_partial


def read_data():
    comments = DataManager.get_comments_clean()
    labels = DataManager.get_original_comments()

    labels = labels.drop(['worker_id', 'quoting_attack', 'recipient_attack', 'third_party_attack', 'other_attack'], axis=1)
    return labels, comments


def merge_data(group, comments):
    data = group.merge(comments, left_on='0', right_on=0, how='inner')
    data.columns = ['rev_id', 'label', '0', 'words']
    data = data.drop(data.columns[2], axis=1)

    with open("../data/data.txt", "wb") as fp:  # Pickling
        pickle.dump(data, fp)
    return data


def merge_comments_words():
    basic = DataManager.get_basic()
    comments = DataManager.get_original_comments()
    comments = comments.drop(['year', 'sample', 'split'], axis=1)

    basic_comments = basic.merge(comments, left_on=2, right_on='rev_id')
    labels = basic_comments.iloc[:, 0:2]
    basic_comments = basic_comments.drop(2, axis=1)
    basic_comments = basic_comments.drop(1, axis=1)
    cl = basic_comments.columns.tolist()
    cl[0] = 'comment'
    with open("../features/basic_comments.txt", "wb") as fp:  # Pickling
        pickle.dump(basic_comments, fp)

    labels = labels[[2, 1]]
    cols = ['rev_id', 'label']
    labels.columns = cols
    pd.DataFrame(labels).to_csv('../features/labels.csv', header=cols)


def create_basic_data(path):
    data = DataManager.get_data()
    comments = DataManager.get_original_comments()

    basic = data.merge(comments, on='rev_id')

    basic = basic.drop('label', axis=1)
    basic = basic.drop('sample', axis=1)
    basic = basic.drop('split', axis=1)

    revs = []

    for index, row in basic.iterrows():
        if len(row['words']) == 0:
            revs.append(row['rev_id'])

    bsc = pd.DataFrame()

    for index, row in basic.iterrows():
        print(index)
        if row['rev_id'] not in revs:
            bsc = bsc.append(row)

    basic = bsc

    with open(path, "wb") as fp:  # Pickling
        pickle.dump(basic, fp)


def clean_new_lines(path):
    data = DataManager.get_comments()
    data = data.reset_index(drop=True)
    for index, row in data.iterrows():
        data.iloc[index, 0] = clean_whitespaces(row['comment'])
        if index % 1000 == 0:
            print(index)
            print(row['comment'])
            print(data.iloc[index, 0])

        if index % 10000 == 0:
            with open(path, "wb") as fp:  # Pickling
                pickle.dump(data, fp)

    data = data.reset_index(drop=True)
    with open(path, "wb") as fp:  # Pickling
        pickle.dump(data, fp)


separate_features()
sentences_clean()
relabel()
create_basic_data('../features/basic_comments.txt')
clean_new_lines('../features/basic_comments_clean.txt')

data = DataManager.get_comments()

labels = DataManager.get_labels()

labels.columns = ['rev_id', 'label']
dt = data.merge(labels, how='left', on='rev_id')
dt = dt.drop('comment', axis=1)

dt = dt.reset_index()
dt.to_csv('../features/labels.csv', index=False)

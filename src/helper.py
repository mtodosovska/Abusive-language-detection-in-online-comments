import pandas as pd
import pickle
import math
import numpy as np

def get_data():
    data = pd.read_pickle('../data/data_features_clean.txt')
    # data = data[['rev_id', 'label', 'words', 'embedding']]
    return data


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


def get_sentences():
    sentences = pd.read_csv('../data/sentences.csv', encoding='latin1').drop(['Unnamed: 0'], axis=1)
    return sentences


def clean_sentences():
    sentences = get_sentences()
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
    data = pd.read_pickle('../data/data_features_clean.txt')
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
    # new_data = new_data.drop(0, axis=1)
    # print(new_data)?
    # new_data.rename(columns={1: 'rev_id'}, inplace=True)
    new_data.columns = names

    print(new_data)
    new_data.to_csv('../features/data_features_clean_flat.csv', header=False, mode='a')
    # with open("../features/data_features_clean_flat.txt", "wb") as fp:  # Pickling
    #     pickle.dump(new_data, fp)


def flat_data_calling():
    for i in range (0, 115):
        print('-----------', i)
        data = get_data()
        dt = data.iloc[1000*i:1000*(i+1)]
        del data
        flat_data(dt)

    data = get_data()
    dt = data.iloc[1000*i:]
    del data
    flat_data(dt)


def separate_features():
    data = pd.read_csv('../features/data_features_clean_flat.csv', header=None).drop(0, axis=1)
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
    data = pd.read_pickle('../features/basic_comments_clean.txt')
    comments = data.drop('logged_in', axis=1) \
        .drop('ns', axis=1) \
        .drop('year', axis=1) \
        .drop('comment', axis=1)
    del data
    comments = comments[['rev_id', 'words']]
    sentences = pd.read_csv('../data/sentences_clean.csv', encoding='latin1').drop('Unnamed: 0', axis=1)
    sent = comments.merge(sentences, how='left', left_index=True, right_index=True).drop('words', axis=1)
    sent.columns = ['rev_id', 'sentence']
    print(sent)
    # print(sent.iloc[0])
    sent.to_csv('../features/sentences_clean.csv', index=False)


def relabel():
    labels = pd.read_csv('../features/labels.csv').iloc[:, :]
    class_0 = labels[labels['label'] == 0].iloc[0:, :]
    class_2 = labels[labels['label'] == 2].append(labels[labels['label'] == 1]).iloc[0:, :]
    class_2.iloc[:, 2] = 0
    class_3 = labels[labels['label'] == 3].append(labels[labels.label == 4]).append(labels[labels.label == 5]).iloc[0:,:]
    class_3.iloc[:, 2] = 1
    print(class_0.shape)
    print(class_2.shape)
    # size = int((class_3.shape[0])/2)
    # class_0 = class_0.sample(frac=1).iloc[0:size, :]
    # class_2 = class_2.sample(frac=1).iloc[0:size, :]
    labels = class_0.append(class_2).append(class_3)

    labels = labels.sample(frac=1)
    labels = labels.reset_index().drop('level_0', axis=1).drop('index', axis=1)
    print(labels)
    labels.to_csv('../features/labels_2_classes.csv', index=False)
    print('-------------Labels done!++++++++++++++')

# separate_features()
# sentences_clean()
relabel()
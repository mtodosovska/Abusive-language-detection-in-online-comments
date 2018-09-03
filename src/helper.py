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
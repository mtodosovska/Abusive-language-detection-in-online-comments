import pandas as pd
import numpy as np
from textblob import TextBlob
import re

# length of comment in tokens
# • average length of word
# • number of punctuations (as is after cleaning, and as it was before cleaning)
#   number of one letter tokens
# • number of capitalized letters
# • number of URLS (I mean it shouldn't be too difficult, but then again, would it make that much difference)
# • number of tokens with non-alpha characters in the# middle (VERY IMPORTANT!!!)
# • number of insult and hate blacklist words (this will be covered with the offensiveness score feature)


def clean_re(comment):
    # print('New lines')
    com = re.sub(r'NEWLINE_TOKEN', ' ', comment)
    comm = re.sub(r'\d+', '.', com)
    return comm


def create_structure(data):
    linguistic = pd.DataFrame()
    ls = np.zeros(8)
    for i, row in data.iterrows():
        print(i)
        ls[0] = row['rev_id']
        linguistic = linguistic.append(pd.DataFrame(ls).transpose())

    linguistic = linguistic.reset_index(drop=True)
    return linguistic


def length_tokens(data, linguistic, i):

    for index, row in data.iterrows():
        print('Method:', i, 'Sample:', index)
        blob = TextBlob(row['comment'])
        words = sum([len(x.words) for x in blob.sentences])
        linguistic.iloc[index, i] = words

    i += 1
    return linguistic, i


def avg_len_words(data, linguistic, i):
    for index, row in data.iterrows():
        print('Method:', i, 'Sample:', index)
        blob = TextBlob(row['comment'])
        words = np.mean([len(x.words) for x in blob.sentences])
        linguistic.iloc[index, i] = words

    i += 1
    return linguistic, i


def num_punct(data, linguistic, i):
    for index, row in data.iterrows():
        print('Method:', i, 'Sample:', index)
        comment = re.findall(r'\W+', row['comment'])
        linguistic.iloc[index, i] = len(comment)

    i += 1
    return linguistic, i


def one_letter(data, linguistic, i):
    for index, row in data.iterrows():
        print('Method:', i, 'Sample:', index)
        comment = re.findall(r' [A-Za-z] ', row['comment'])
        linguistic.iloc[index, i] = len(comment)

    i += 1
    return linguistic, i


def num_capitalised(data, linguistic, i):
    for index, row in data.iterrows():
        print('Method:', i, 'Sample:', index)
        comment = re.findall(r'.[A-Z].', row['comment'])
        linguistic.iloc[index, i] = len(comment)

    i += 1
    return linguistic, i


def num_urls(data, linguistic, i):
    for index, row in data.iterrows():
        print('Method:', i, 'Sample:', index)
        comment = re.findall(r'(http|ftp|https)://', row['comment'])
        linguistic.iloc[index, i] = len(comment)

    i += 1
    return linguistic, i


def num_non_alpha(data, linguistic, i):
    for index, row in data.iterrows():
        print('Method:', i, 'Sample:', index)
        comment = re.findall(r'\w[^\w\s]+\w+', clean_re(row['comment']))
        print(comment)
        linguistic.iloc[index, i] = len(comment)

    i += 1
    return linguistic, i

i = 1

data_clean = pd.read_pickle('../features/basic_comments_clean.txt')
data = pd.read_pickle('../features/basic_comments.txt')

data = data.reset_index(drop=True)

linguistic = create_structure(data)
print(data_clean.shape)
print(data.shape)
print(linguistic.shape)

linguistic, i = num_punct(data, linguistic, i)
linguistic, i = length_tokens(data_clean, linguistic, i)
linguistic, i = avg_len_words(data_clean, linguistic, i)
linguistic, i = one_letter(data_clean, linguistic, i)
linguistic, i = num_capitalised(data_clean, linguistic, i)
linguistic, i = num_urls(data, linguistic, i)
linguistic, i = num_non_alpha(data, linguistic, i)

linguistic.to_csv('../features/linguistic.csv', index=False)

print(linguistic)

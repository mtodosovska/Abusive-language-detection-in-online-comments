import spacy
import pandas as pd

from data_manager import DataManager

""" 
find all the offensive words in the sentence
hatebase has an offensivenss level for every word, add that up
the swear words dict doesn't, but they are swear words, so they must be more offensive than
others, I'm not sure about this, but I should add something for it...
step two will follow the original plan, if there are second person pronouns around the
offensive words (in the dependencies) then we add up twice the offensiveness of the original
offensive word.
(the first is according to Kontostathis, the second is according to saving people
([Chen et al., 2012)
"""


def get_dependencies(comments, pronouns, bad_words, intensities, path):
    nlp = spacy.load('en_core_web_sm')
    offensiveness_score = pd.DataFrame()

    for index, row in comments.iterrows():
        print(index)
        doc = nlp(row['words'])
        offensiveness = 0

        for token in doc:
            if token.head.text in bad_words:
                print(row['words'])
                print(token.head.text)
                offensiveness += intensities[token.head.text]
                if token.text in pronouns:
                    print(token.text, token.head.text)
                    offensiveness += intensities[token.head.text]
                    # print(token.text, token.head.text)

            if token.text in bad_words:
                print(row['words'])
                print(token.text)
                offensiveness += intensities[token.text]
                if token.head.text in pronouns:
                    print(token.text, token.head.text)
                    offensiveness += intensities[token.text]
                    # print(token.text, token.head.text)

        ls = [row['rev_id'], offensiveness]
        offensiveness_score = offensiveness_score.append(pd.DataFrame(ls).transpose())

    offensiveness_score.reset_index(drop=True)
    print(offensiveness_score)
    offensiveness_score.to_csv(path, index=False)


def get_offensiveness_score():
    pronouns = ['you', 'your', 'yours', 'yourself', 'yourselves']

    comments = DataManager.get_comments()
    comments['words'] = comments['words'].apply(lambda x: " ".join(y for y in x))

    swear_words = DataManager.get_swear_words()
    hate_words = DataManager.get_hate_words()

    hate_words = hate_words[hate_words.offensiveness != 0]

    hw = hate_words['vocabulary'].tolist()
    sw = swear_words.iloc[:, 0].tolist()

    intensities = {}
    bad_words = set(hw)
    for index, row in hate_words.iterrows():
        intensities[row['vocabulary']] = row['offensiveness']
    for i in sw:
        bad_words.add(i)
        intensities[i] = 1

    get_dependencies(comments, pronouns, bad_words, intensities, '../features/offensiveness_score.csv')


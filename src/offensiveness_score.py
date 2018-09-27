import spacy
import pandas as pd


def get_comments():
    data = pd.read_pickle('../features/basic_comments_clean.txt')
    comments = data.drop('logged_in', axis=1)\
        .drop('ns', axis=1)\
        .drop('year', axis=1)\
        .drop('comment', axis=1)
    comments = comments[['rev_id', 'words']]
    print(comments['words'])
    comments['words'] = comments['words'].apply(lambda x: " ".join(y for y in x))
    print(comments.iloc[0])
    return comments


def get_dependencies(comments, pronouns, bad_words, intensities):
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
    offensiveness_score.to_csv('../features/offensiveness_score.csv', index=False)


comments = get_comments()
pronouns = ['you', 'your', 'yours', 'yourself', 'yourselves']
swear_words = pd.read_csv('../data/bad words/swear_words.csv', delimiter='\n', encoding='latin1')
hate_words = pd.read_csv('../data/bad words/hatebase.csv', encoding='latin1').drop('Unnamed: 0', axis=1)#['vocabulary']
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
print('dict', intensities)
get_dependencies(comments, pronouns, bad_words, intensities)

# find all the offensive words in the sentence
# hatebase has an offensivenss level for every word, add that up
# the swear words dict doesn't, but they are swear words, so they must be more offensive than
# others, I'm not sure about this, but I should add something for it...
# step two will follow the original plan, if there are second person pronouns around the
# offensive words (in the dependencies) then we add up twice the offensiveness of the original
# offensive word.
# (the first is according to Kontostathis, the second is according to saving people
# ([Chen et al., 2012)

import pandas as pd
import pickle

class DataManager:
    def __init__(self):
        pass

    @staticmethod
    def get_data():
        data = pd.read_pickle('../data/data.txt')
        return data

    @staticmethod
    def get_ngrams():
        ngrams = pd.read_csv('../features/ngrams.csv', encoding='latin1').drop('Unnamed: 0', axis=1)
        return ngrams

    @staticmethod
    def get_pos_ngrams():
        ngrams = pd.read_csv('../features/pos_ngrams.csv', encoding='latin1')
        return ngrams

    @staticmethod
    def get_scores():
        scores = pd.read_csv('../features/scores.csv', encoding='latin1').drop('Unnamed: 0', axis=1)
        return scores

    @staticmethod
    def get_original_comments():
        comments = pd.read_csv('../data/4054689/attack_annotated_comments.tsv', sep='\t', index_col=0, delimiter='\t', encoding='latin1')
        return comments

    @staticmethod
    def get_annotations():
        annotations = pd.read_csv('../data/4054689/attack_annotations.tsv',  sep = '\t')
        return annotations

    @staticmethod
    def get_data_features_clean():
        data = pd.read_csv('../features/data_features_clean.csv', header=None)
        return data

    @staticmethod
    def get_data_features():
        data = pd.read_csv('../features/data_features_clean_flat.csv', header=None)
        return data

    @staticmethod
    def get_swear_words():
        swear_words = pd.read_csv('../data/bad words/swear_words.csv', delimiter='\n', encoding='latin1')
        return swear_words

    @staticmethod
    def get_hate_words():
        hate_words = pd.read_csv('../data/bad words/hatebase.csv', encoding='latin1').drop('Unnamed: 0', axis=1)#['vocabulary']
        return hate_words

    @staticmethod
    def get_comments():
        data = pd.read_pickle('../features/basic_comments_clean.txt')
        comments = data.drop('logged_in', axis=1) \
            .drop('ns', axis=1) \
            .drop('year', axis=1) \
            .drop('words', axis=1)
        comments = comments[['rev_id', 'comment']]
        return comments

    @staticmethod
    def get_sentences_clean():
        sentences = pd.read_csv('../features/sentences_clean.csv', encoding='latin1')
        return sentences

    @staticmethod
    def get_comments_clean():
        with open("../data/comments_clean.txt", "rb") as fp:  # Unpickling
            comments = pickle.load(fp)
        return comments

    @staticmethod
    def get_basic():
        with open("../features/basic.txt", "rb") as fp:  # Unpickling
            basic = pickle.load(fp)
        return basic

    @staticmethod
    def get_basic_comments():
        with open("../features/basic_comments.txt", "rb") as fp:  # Unpickling
            basic = pickle.load(fp)
        return basic

    @staticmethod
    def get_labels():
        labels = pd.read_csv('../features/labels.csv').drop('Unnamed: 0', axis=1).drop('Unnamed: 0.1', axis=1)
        return labels

    @staticmethod
    def get_labels_2_classes():
        labels = pd.read_csv('../features/labels_2_classes.csv').drop('Unnamed: 0', axis=1).drop('Unnamed: 0.1', axis=1)
        return labels

    @staticmethod
    def get_general_inquierer():
        inq = pd.read_csv('../data/general inquierer/inquireraugmented.csv', encoding='latin1').iloc[1:]
        return inq

    @staticmethod
    def get_tokenised():
        tokenised = pd.read_pickle('../features/basic_comments_tokenised.txt')
        return tokenised

    @staticmethod
    def get_sentences():
        sentences = pd.read_csv('../data/sentences.csv', encoding='latin1').drop(['Unnamed: 0'], axis=1)
        return sentences

    @staticmethod
    def get_offensiveness_score():
        offensiveness = pd.read_csv('../features/offensiveness_score.csv', encoding='latin1')
        return offensiveness

    @staticmethod
    def get_embeddings():
        embeddings = pd.read_csv('../features/embeddings.csv', encoding='latin1')
        return embeddings

    @staticmethod
    def get_linguistic():
        lin = pd.read_csv('../features/linguistic.csv', encoding='latin1')
        return lin
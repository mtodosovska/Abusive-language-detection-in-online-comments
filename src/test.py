import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin

class SimpleTransformer(TransformerMixin):

    def transform(self, X, **transform_params):
        return X

    def fit(self, X, y=None, **fit_params):
        return self


comments = pd.read_csv('../data/4054689/attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('../data/4054689/attack_annotations.tsv',  sep = '\t')

len(annotations['rev_id'].unique())

# labels a comment as an atack if the majority of annoatators did so
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

# join labels and comments
comments['attack'] = labels

# remove newline and tab tokens
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

comments.query('attack')['comment'].head()

train_comments = comments.query("split=='train'")
test_comments = comments.query("split=='test'")

print('-------------------------------------------------------------------------')
print(len(train_comments['comment']))
print(len(test_comments['comment']))
print('-------------------------------------------------------------------------')


# labels = [int(x) for x in train_comments['attack'].tolist()]
# test_labels = [int(x) for x in test_comments['attack'].tolist()]
#
# vect = CountVectorizer(max_features=1000, ngram_range = (2,2))
# vect.fit(train_comments['comment'])
# vct = vect.transform(train_comments['comment'])
# ngrams = pd.DataFrame(vct.todense(), columns=vect.get_feature_names())
# vect.fit(test_comments['comment'])
# vct_test = vect.transform(test_comments['comment'])
# ngrams_test = pd.DataFrame(vct_test.todense(), columns=vect.get_feature_names())

# clf = Pipeline([
#     ('vect', vect),
#     ('tfidf', tfidf),
#     ('clf', LogisticRegression()),
# ])


# classifier = LogisticRegression(class_weight='balanced')
# classifier.fit(ngrams, labels)
# results = classifier.predict(ngrams_test)
# probabilities = classifier.predict_proba(ngrams_test)


clf = Pipeline([
    ('vect', CountVectorizer(max_features = 10000, ngram_range = (1,2))),
    ('tfidf', TfidfTransformer(norm = 'l2')),
    ('clf', LogisticRegression()),
])

clf = clf.fit(train_comments['comment'], train_comments['attack'])
probabilities = clf.predict_proba(test_comments['comment'])[:, 1]
results = clf.predict(test_comments['comment'])
print(results)


# print(classification_report(test_comments['attack'], results, target_names=['class_0', 'class_1']))
# print(confusion_matrix(test_comments['attack'], results))
#
# auc = roc_auc_score(test_labels, results)
# print('Test ROC AUC: %.3f' %auc)

print(classification_report(test_comments['attack'], results, target_names=['class_0', 'class_1']))
print(confusion_matrix(test_comments['attack'], results))

auc = roc_auc_score(test_labels, probabilities)
print('Test ROC AUC: %.3f' %auc)
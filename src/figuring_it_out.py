import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime

from sqlalchemy.orm.attributes import ScalarAttributeImpl


class SimpleTransformerGeneral(TransformerMixin):

    def transform(self, X, **transform_params):
        # ls = X.drop('comment', axis=1).drop('words', axis=1)
        # print(ls.shape)
        return X


    def fit(self, X, y=None, **fit_params):
        return self


class SimpleTransformerComments(TransformerMixin):

    def transform(self, X, **transform_params):
        # sentences.append(" ".join(str(x) for x in comment))
        # X['comment'] = X['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
        # X['comment'] = X['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
        # ls = X['words']
        X['words'] = X['comment'].apply(lambda x: " ".join(str(y) for y in x))
        print()
        ls = X['words']
        print(ls.shape)
        return ls

    def fit(self, X, y=None, **fit_params):
        return self

print('Fitting pipeline')
clf = Pipeline([
    ('features', SimpleTransformerGeneral()),
    ('LogisticRegression', LogisticRegression(class_weight='balanced'))
    # ('RandomForest', RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=0, class_weight='balanced', bootstrap=True))
])

n = 100000

print('Getting data')
print(datetime.datetime.now())
labels2 = pd.read_csv('../features/labels_2_classes.csv', encoding='latin1')
# partial = pd.read_csv('../data/partial.csv', encoding='latin1').drop(['Unnamed: 0'], axis=1)

# comments = pd.read_pickle('../features/basic_comments_clean.txt').iloc[0:n]
# print(comments.iloc[0])
# ngrams = pd.read_csv('../features/ngrams3.csv', encoding='latin1')
scores = pd.read_csv('../features/scores.csv', encoding='latin1').drop('Unnamed: 0', axis=1).iloc[0:n]
# pos_tags = pd.read_csv('../features/pos_ngrams.csv', encoding='latin1').iloc[0:n]
linguistic = pd.read_csv('../features/linguistic.csv', encoding='latin1').iloc[0:n]
embeddings = pd.read_csv('../features/embeddings.csv', encoding='latin1').iloc[0:n]

print('Merging data')
print(datetime.datetime.now())

embeddings = embeddings.merge(linguistic, on='0')

# ngrams = ngrams.merge(scores, on='rev_id')
# scores = scores.merge(pos_tags, on='rev_id')
# scores = scores.merge(ngrams, on='rev_id')
# scores = scores.merge(comments, on='rev_id')

new_labels = labels2.merge(scores, how='right', left_on='rev_id', right_on='rev_id')
new_labels = new_labels.merge(embeddings, left_on='rev_id', right_on='0').drop('0', axis=1)

# features = scores.merge(linguistic, left_on='rev_id', right_on='0').drop('0', axis=1)

# new_labels = labels2.merge(features, on='rev_id')

features = new_labels.drop('label', axis=1).drop('rev_id', axis=1)


print('Splitting data')
train, test, labels_train, labels_test = train_test_split(
    features, new_labels['label'],
    test_size=0.20, random_state=42)

print('hi')
print('Fitting and classifying')
print(datetime.datetime.now())
clf = clf.fit(train, labels_train)
probabilities = clf.predict_proba(test)
results = clf.predict(test)
print(results)

print(classification_report(labels_test, results, target_names=['class_0', 'class_1']))
print(confusion_matrix(labels_test, results))

auc = roc_auc_score(labels_test, results)
print('Test ROC AUC: %.3f' %auc)
print(datetime.datetime.now())
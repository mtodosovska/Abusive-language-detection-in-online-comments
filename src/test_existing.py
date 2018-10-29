import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

comments = pd.read_csv('../data/4054689/attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('../data/4054689/attack_annotations.tsv',  sep = '\t')
len(annotations['rev_id'].unique())
# labels a comment as an atack if the majority of annoatators did so
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

# join labels and comments# join
comments['attack'] = labels


# remove newline and tab tokens# remov
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

comments.query('attack')['comment'].head()

# train_comments = comments.query("split=='train'")
# test_comments = comments.query("split=='test'")

# train_comments, test_comments, labels_train, labels_test = train_test_split(
#     comments, comments['attack'],
#     test_size=0.20, random_state=42)

# clf = Pipeline([
#     ('vect', CountVectorizer(max_features = 10000, ngram_range = (1,2))),
#     ('tfidf', TfidfTransformer(norm = 'l2')),
#     ('clf', LogisticRegression()),
# ])
# clf = clf.fit(train_comments['comment'], labels_train)

ngrams = pd.read_csv('../features/ngrams.csv', encoding='latin1').drop('Unnamed: 0', axis=1)
new_labels = ngrams.merge(pd.DataFrame(comments['attack']), how='left', left_index=True, right_index=True)
features = ngrams.drop('rev_id', axis=1)

train_comments, test_comments, labels_train, labels_test = train_test_split(
    features, new_labels['attack'],
    test_size=0.20, random_state=42)

clf = LogisticRegression()
clf = clf.fit(train_comments, labels_train)
results = clf.predict(test_comments)
probabilities = clf.predict_proba(test_comments)

auc = roc_auc_score(labels_test, results)
print('Test ROC AUC: %.3f' %auc)
auc = roc_auc_score(labels_test, probabilities[:, 1])
print('Test ROC AUC: %.3f' %auc)
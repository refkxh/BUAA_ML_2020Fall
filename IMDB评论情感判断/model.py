from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC


training_data = pd.read_csv('public_data/train.csv', index_col=0)
raw_test_data = pd.read_csv('public_data/test_data.csv', index_col=0)
test_data = pd.read_csv('public_data/submission.csv', index_col=0)

training_reviews = training_data['review'].tolist()
training_labels = training_data.index.tolist()
training_sentiments = training_data['sentiment'].tolist()

test_reviews = raw_test_data['review'].tolist()
test_labels = raw_test_data.index.tolist()


def clean_text(text):
    punctuations = """.,?!:;(){}[]"""
    text = text.lower().replace('\n','')
    text = text.replace('<br />', ' ')

    for punctuation in punctuations:
        text = text.replace(punctuation, ' {} '.format(punctuation))
    
    text = text.split()
    return text


def labelize_reviews(reviews, labels):
    for i, review in enumerate(reviews):
        yield TaggedDocument(clean_text(review), [labels[i]])


labelized_training_reviews = list(labelize_reviews(training_reviews, training_labels))
labelized_test_reviews = list(labelize_reviews(test_reviews, test_labels))

dm = Doc2Vec(vector_size=100, min_count=2, window=10, alpha=0.05, sample=0, negative=5, hs=0, epochs=20, workers=4)
dbow = Doc2Vec(vector_size=100, min_count=2, sample=0, negative=5, hs=0, epochs=20, workers=4, dm=0)

all_reviews = labelized_training_reviews.copy()
all_reviews.extend(labelized_test_reviews)
dm.build_vocab(all_reviews)
dbow.build_vocab(all_reviews)

for epoch in range(20):
    print('EPOCH {}'.format(epoch + 1))
    random.shuffle(all_reviews)
    dm.train(all_reviews, total_examples=dm.corpus_count, epochs=1)
    dbow.train(all_reviews, total_examples=dbow.corpus_count, epochs=1)

training_vectors_dm = np.zeros((len(training_reviews), 100))
train_vectors_dbow = np.zeros((len(training_reviews), 100))
for i in range(len(training_reviews)):
    training_vectors_dm[i] = dm.docvecs[training_labels[i]]
    train_vectors_dbow[i] = dbow.docvecs[training_labels[i]]
training_vectors = np.hstack((training_vectors_dm, train_vectors_dbow))

test_vectors_dm = np.zeros((len(test_reviews), 100))
test_vectors_dbow = np.zeros((len(test_reviews), 100))
for i in range(len(test_reviews)):
    test_vectors_dm[i] = dm.docvecs[test_labels[i]]
    test_vectors_dbow[i] = dbow.docvecs[test_labels[i]]
test_vectors = np.hstack((test_vectors_dm, test_vectors_dbow))

parameters = {'C': np.logspace(-2, -1, 20)}
model = GridSearchCV(LinearSVC(dual=False), parameters, n_jobs=4)
model.fit(training_vectors, training_sentiments)
print('Best params:', model.best_params_) # 0.055
print('Best score:', model.best_score_) # 0.897

test_data.loc[:, 'sentiment'] = model.predict(test_vectors)
test_data.to_csv('submission.csv')

import json
from datetime import datetime
from collections import Counter
from itertools import islice
import numpy as np
N = 200000
t1 = datetime.now()
# read the data from disk and split into lines
# we use .strip() to remove the final (empty) line
# each line of the file is a separate JSON object
with open("yelp_academic_dataset_review.json") as f:
    reviews = [json.loads(review) for review in islice(f, N)]

print "read file in:" + str(datetime.now() - t1)

# we're interested in the text of each review
# and the stars rating, so we load these into
# separate lists


texts = [review['text'] for review in reviews]
stars = np.array([review['stars'] for review in reviews])

#try to predict good, neutral or bad recommendation
stars[stars < 3] = 1
stars[stars > 3] = 5


def balance_classes(xs, ys):
    """Undersample xs, ys to balance classes."""
    freqs = Counter(ys)

    # the least common class is the maximum number we want for all classes
    max_allowable = freqs.most_common()[-1][1]
    num_added = {clss: 0 for clss in freqs.keys()}
    new_ys = []
    new_xs = []
    for i, y in enumerate(ys):
        if num_added[y] < max_allowable:
            new_ys.append(y)
            new_xs.append(xs[i])
            num_added[y] += 1
    return new_xs, new_ys


print "Count numbers of classes in unbalanced set"
print(Counter(stars))
balanced_x, balanced_y = balance_classes(texts, stars)
print "Count numbers of classes in balanced set"
print(Counter(balanced_y))

from sklearn.feature_extraction.text import TfidfVectorizer

# This vectorizer breaks text into single words and bi-grams
# and then calculates the TF-IDF representation
vectorizer = TfidfVectorizer(ngram_range=(1,2))
print "finished TFid"

# the 'fit' builds up the vocabulary from all the reviews
# while the 'transform' step turns each indivdual text into
# a matrix of numbers.
vectors = vectorizer.fit_transform(balanced_x)
print "finished vectorizer:"
print(datetime.now() - t1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, balanced_y, test_size=0.33, random_state=42)

from sklearn.svm import LinearSVC


def train_predict(model, x_train, y_train, x_test, label="model"):
    """
    train and predict for a given model
    """
    # train the classifier
    print "Start Analysis for " + label
    print ""
    print "start training " + label
    model.fit(x_train, y_train)
    print "end training:" + label
    print(datetime.now() - t1)

    preds = model.predict(x_test)

    return preds


def analysis(y_test, preds):
    """
    analyses the output with classification report and confusion matrix
    """
    from sklearn.metrics import classification_report
    print "classification_report:"
    print(classification_report(y_test, preds))

    from sklearn.metrics import confusion_matrix
    print "confusion matrix:"
    print(confusion_matrix(y_test, preds))

    print "Test examples"
    print("predicted:" + str(list(preds[:10])))
    print("reality:" + str(list(y_test[:10])))


classifier = LinearSVC()
preds_a = train_predict(classifier, X_train, y_train, X_test, label="SVM")
analysis(y_test, preds_a)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
preds_b = train_predict(classifier, X_train, y_train, X_test, label="Logistic Regression")
analysis(y_test, preds_b)



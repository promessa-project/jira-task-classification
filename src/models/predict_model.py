from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
import joblib

def predict_model_cv(clf, count_vect, x_test):
    tfidf_transformer = TfidfTransformer()
    x_test_counts = count_vect.transform(x_test)
    x_test_tfidf = tfidf_transformer.fit_transform(x_test_counts)
    # x_test_tfidf = tfidf_transformer.transform(x_test)
    predicted = clf.predict(x_test_tfidf)
    # print('-----PREDICTION-----')
    # for doc, category in zip(x_test, predicted):
    #    print('%r => %s' % (doc, category))

    return predicted


def predict_task(model_name, x_test):#todo
    clf = joblib.load('models/'+model_name+'.joblib')
    count_vect = joblib.load('models/feature/CountVectorizer.joblib')

    tfidf_transformer = TfidfTransformer()
    x_test_tfidf = tfidf_transformer.fit_transform(count_vect)

    predicted = clf.predict(x_test_tfidf)[0]

    print('predicted value::', predicted)

    return predicted

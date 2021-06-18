from sklearn.feature_extraction.text import TfidfTransformer

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


def predict_task(clf, x_test):#todo
    tfidf_transformer = TfidfTransformer()

    x_test_tfidf = tfidf_transformer.transform([x_test])
    predicted = clf.predict(x_test_tfidf)
    # print('-----PREDICTION-----')
    # for doc, category in zip(x_test, predicted):
    #    print('%r => %s' % (doc, category))

    return predicted

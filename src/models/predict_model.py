from sklearn.feature_extraction.text import TfidfTransformer

def predict(model, count_vect, X_train_tfidf, y_train, X_test):
    clf = model.fit(X_train_tfidf, y_train)
    docs_new = X_test
    tfidf_transformer = TfidfTransformer()
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.fit_transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)
#     print('-----PREDICTION-----')
#     for doc, category in zip(docs_new, predicted):
#        print('%r => %s' % (doc, category))

    return predicted

predict()

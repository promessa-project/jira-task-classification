from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import pandas as pd

def encode_data(df):
    col = ['issue', 'epic', 'summary', 'clean_summary', 'epic_clean_summary', 'project_key']
    df_encode = df[col]
    df_encode = df_encode[pd.notnull(df_encode['summary'])]
    df_encode.columns = ['issue', 'epic', 'summary', 'clean_summary', 'epic_clean_summary', 'project_key']
    df['category_id'] = df['epic'].factorize()[0]
    df_encode['category_id'] = df['category_id']
    category_id_df = df[['epic', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'epic']].values)
    df_encode['clean_summary'] = df_encode['clean_summary'].str.replace('|',' ')
    df_encode['epic_clean_summary'] = df_encode['epic_clean_summary'].str.replace('|',' ')

    return df_encode


def get_count_vectorizer_matrix(proj_train):
    count_vect = CountVectorizer(binary=True)
    X_train_counts = count_vect.fit_transform(proj_train)

    return X_train_counts, count_vect


def get_tfidf_matrix(X_train_counts):
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return X_train_tfidf

from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer():
    # max_df=0.7 ignores terms that appear in more than 70% of documents
    return TfidfVectorizer(stop_words='english', max_df=0.7)

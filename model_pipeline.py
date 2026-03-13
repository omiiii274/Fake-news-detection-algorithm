import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessor import get_tfidf_vectorizer

def run_detection_pipeline(csv_path):
    df = pd.read_csv(csv_path)
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf_vectorizer = get_tfidf_vectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # PassiveAggressive is excellent for large-scale data stream classification
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(X_train_tfidf, y_train)

    y_pred = pac.predict(X_test_tfidf)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_detection_pipeline('news.csv')

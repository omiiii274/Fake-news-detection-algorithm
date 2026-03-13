import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def run_fake_news_detector(data_path='news.csv'):
    # Load and basic cleaning
    df = pd.read_csv(data_path)
    X = df['text']
    y = df['label']

    # Stratified split to maintain balance between Fake and Real
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    
    # Fit and transform
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # PassiveAggressiveClassifier - Ideal for high-dimensional text data
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(X_train_tfidf, y_train)

    # Prediction and Evaluation
    y_pred = pac.predict(X_test_tfidf)
    score = accuracy_score(y_test, y_pred)
    
    print(f'✅ Model Accuracy: {score*100:.2f}%')
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.title('Fake News Detection: Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("📂 Analysis saved: confusion_matrix.png")

if __name__ == "__main__":
    run_fake_news_detector()

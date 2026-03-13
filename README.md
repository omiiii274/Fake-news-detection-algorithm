# 📰 Fake News Detection Algorithm

## 📋 Executive Summary
This project implements a high-performance machine learning pipeline to identify misinformation in news articles. Using a **Passive-Aggressive Classifier**, the model is optimized for the high-velocity, high-volume nature of digital news cycles.

## 🧪 Technical Strategy
* **Feature Extraction:** Utilized **TF-IDF Vectorization** to convert unstructured text into high-signal numerical features, filtering out corpus-specific stop words.
* **Algorithm Choice:** Selected the **Passive-Aggressive Classifier** due to its efficiency in large-scale text classification and its ability to adjust weightings aggressively upon misclassification.
* **Validation:** Employed **Stratified Splitting** to ensure the model was trained and tested on a representative balance of legitimate and fraudulent news.

## 📊 Performance
* **Accuracy:** 93%+ (Subject to dataset version)
* **Metric:** Evaluated via **Precision-Recall** to ensure minimal "False Positives" (labeling real news as fake).

![Fake News Detection Confusion Matrix](confusion_matrix.png)

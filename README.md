# 📚 Book Genre Classification using NLP & Deep Learning  

> Classifying book genres from summaries using deep learning and NLP.

---

## 🔎 Overview  
This project applies **Natural Language Processing (NLP)** and **Deep Learning (DL)** to classify books into genres based on their summaries.  
It combines **statistical text features (TF–IDF, PMI)** with **modern embeddings (Word2Vec, SBERT)** and evaluates multiple classifiers, from classical models like **Naive Bayes & Logistic Regression** to **Neural Networks**.  

The work demonstrates how traditional frequency-based representations and semantic embeddings can be integrated for accurate and interpretable genre classification.  

---

## 🎯 Objectives  
1. Design a preprocessing pipeline (tokenization, stopword removal, lemmatization).  
2. Extract features using TF–IDF, PMI-based collocations, Word2Vec, and SBERT.  
3. Train and evaluate classifiers (Naive Bayes, Logistic Regression, Neural Networks).  
4. Compare sparse vs. dense features and analyze per-genre performance.  
5. Visualize embeddings and confusion matrices for deeper insights.  
6. Provide a **reproducible, extensible NLP pipeline** for future research.  

---

## 📊 Dataset  
- **Size**: 4,657 records  
- **Columns**:  
  - `index` – unique ID  
  - `title` – book title  
  - `genre` – target variable (6 genres: Thriller, Fantasy, Science, Horror, History, Crime)  
  - `summary` – book description / input text  

- **Notes**: Balanced preprocessing steps applied (removal of duplicates, truncation of very long summaries, genre filtering).  

---

## 🛠 Methods & Models  

### ✨ Feature Engineering  
- **TF–IDF** (n-grams, max_features=5000)  
- **PMI-based features** (word–genre associations)  
- **Word2Vec** (CBOW & Skip-gram)  
- **Sentence-BERT (SBERT)** embeddings  

### 🤖 Models  
- **Multinomial Naive Bayes**  
- **Logistic Regression** (on TF–IDF & SBERT)  
- **Neural Networks** (with TF–IDF and PMI features)  

---

## 📈 Results (Highlights)  

| Model / Features            | Test Accuracy | Weighted F1 |
|------------------------------|---------------|-------------|
| Naive Bayes (TF–IDF)        | 67.4%         | 0.66        |
| Logistic Regression (TF–IDF)| 73.9%         | 0.74        |
| Logistic Regression (SBERT) | 69.3%         | 0.69        |
| Word2Vec (CBOW)             | 33.4%         | 0.22        |
| Word2Vec (Skip-gram)        | 63.9%         | 0.63        |
| Neural Net (TF–IDF/PMI)     | ~73–74%       | ~0.74       |

**Key Insights**  
- **TF–IDF + Logistic Regression** outperformed others for this dataset.  
- **Skip-gram Word2Vec** worked significantly better than CBOW.  
- **SBERT embeddings** gave semantic robustness, though not higher accuracy here.  
- PMI features improved interpretability and genre-specific associations.  


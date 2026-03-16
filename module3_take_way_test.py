#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:33:04 2026

@author: mutua
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 2: Text Preprocessing and Linguistic Fundamentals
Module 3: Text Representation and Feature Engineering

Objective:
Implement text preprocessing and convert text into numerical
representations suitable for machine learning models.

Tasks Covered:

MODULE 2
1. Convert text to lowercase
2. Remove numbers and punctuation
3. Remove stopwords
4. Apply stemming or lemmatization
5. Display original vs cleaned text

MODULE 3
1. Load publicly available NLP data
2. Implement Bag-of-Words
3. Implement TF-IDF
4. Generate Word2Vec embeddings
5. Visualize embeddings

Author: mutua
Date: Feb 2026
"""

# ==================================================
# IMPORT REQUIRED LIBRARIES
# ==================================================

import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import gutenberg

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA

from gensim.models import Word2Vec

# download resources if missing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('gutenberg')

# ==================================================
# MODULE 2 — TEXT PREPROCESSING
# ==================================================

# SAMPLE RAW TEXT
raw_text = "The cats are running quickly towards the garden!! NLP is AMAZING in 2026."

print("\n==============================")
print("STAGE 0: ORIGINAL TEXT")
print("==============================")
print(raw_text)

# --------------------------------------------------
# STAGE 1: LOWERCASE
# --------------------------------------------------

print("\nSTAGE 1: LOWERCASE")

text_lower = raw_text.lower()
print(text_lower)

# --------------------------------------------------
# STAGE 2: REMOVE NUMBERS AND PUNCTUATION
# --------------------------------------------------

print("\nSTAGE 2: REMOVE NUMBERS AND PUNCTUATION")

text_clean = re.sub(r'[^a-z\s]', '', text_lower)
text_clean = re.sub(r'\s+', ' ', text_clean).strip()

print(text_clean)

# --------------------------------------------------
# STAGE 3: TOKENIZATION
# --------------------------------------------------

print("\nSTAGE 3: TOKENIZATION")

tokens = word_tokenize(text_clean)
print(tokens)

# --------------------------------------------------
# STAGE 4: STOPWORD REMOVAL
# --------------------------------------------------

print("\nSTAGE 4: STOPWORD REMOVAL")

stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w not in stop_words]

print(filtered_tokens)

# --------------------------------------------------
# STAGE 5A: STEMMING
# --------------------------------------------------

print("\nSTAGE 5A: STEMMING")

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

print(stemmed_tokens)

# --------------------------------------------------
# STAGE 5B: LEMMATIZATION
# --------------------------------------------------

print("\nSTAGE 5B: LEMMATIZATION")

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print(lemmatized_tokens)

# --------------------------------------------------
# FINAL COMPARISON
# --------------------------------------------------

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

cleaned_text_stemmed = " ".join(stemmed_tokens)
cleaned_text_lemmatized = " ".join(lemmatized_tokens)

print("Original:", raw_text)
print("Stemmed:", cleaned_text_stemmed)
print("Lemmatized:", cleaned_text_lemmatized)

# ==================================================
# MODULE 3 — TEXT REPRESENTATION
# ==================================================

print("\n\n========================================")
print("MODULE 3: TEXT REPRESENTATION")
print("========================================")

# --------------------------------------------------
# LOAD PUBLIC NLP DATA
# --------------------------------------------------

print("\nLoading public NLP dataset (NLTK Gutenberg Corpus)...")

sentences = gutenberg.sents('austen-emma.txt')

# convert first 500 sentences into text form
documents = [" ".join(sentence) for sentence in sentences[:500]]

print("Total documents loaded:", len(documents))

# --------------------------------------------------
# BAG OF WORDS
# --------------------------------------------------

print("\nSTAGE 1: BAG OF WORDS")

vectorizer = CountVectorizer(stop_words='english', max_features=20)
bow_matrix = vectorizer.fit_transform(documents)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("\nBoW Feature Matrix (first 5 rows)")
print(bow_df.head())

# --------------------------------------------------
# TF-IDF REPRESENTATION
# --------------------------------------------------

print("\nSTAGE 2: TF-IDF")

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\nTF-IDF Feature Matrix (first 5 rows)")
print(tfidf_df.head())

# --------------------------------------------------
# WORD2VEC EMBEDDINGS
# --------------------------------------------------

print("\nSTAGE 3: WORD2VEC EMBEDDINGS")

# tokenize sentences
tokenized_sentences = [
    [word.lower() for word in sentence]
    for sentence in sentences[:500]
]

# train model
w2v_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

print("\nVector for word 'emma':")
print(w2v_model.wv['emma'][:10])

# --------------------------------------------------
# SIMILAR WORDS
# --------------------------------------------------

print("\nWords similar to 'emma':")

print(w2v_model.wv.most_similar('emma', topn=5))

# --------------------------------------------------
# VISUALIZATION USING PCA
# --------------------------------------------------

print("\nSTAGE 4: VISUALIZATION OF WORD EMBEDDINGS")

words = list(w2v_model.wv.index_to_key[:50])
vectors = [w2v_model.wv[word] for word in words]

pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

plt.figure(figsize=(10,7))

for i, word in enumerate(words):
    x, y = coords[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word, fontsize=9)

plt.title("Word2Vec Embedding Visualization (PCA)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# --------------------------------------------------
# FEATURE COMPARISON SUMMARY
# --------------------------------------------------

print("\n================================")
print("FEATURE REPRESENTATION SUMMARY")
print("================================")

print("""
Bag-of-Words:
Counts word occurrences in documents.
Simple but ignores word order and meaning.

TF-IDF:
Improves BoW by weighting rare but important words higher.

Word Embeddings (Word2Vec):
Dense vector representations capturing semantic similarity.
Example: 'king' - 'man' + 'woman' ≈ 'queen'

Applications:
• Text Classification
• Sentiment Analysis
• Recommendation Systems
• Chatbots
""")

print("\nNLP feature engineering pipeline completed successfully.")
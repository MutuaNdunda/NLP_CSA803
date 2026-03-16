#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 2: Text Preprocessing and Linguistic Fundamentals

Objective:
Implement basic text preprocessing in Python.

Tasks Covered:
1. Convert text to lowercase
2. Remove numbers and punctuation
3. Remove stopwords
4. Apply stemming or lemmatization
5. Display original vs cleaned text

Author: mutua
Date: Feb 2026
"""

# ================================
# IMPORT REQUIRED LIBRARIES
# ================================

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ================================
# SAMPLE RAW TEXT
# ================================

raw_text = "The cats are running quickly towards the garden!! NLP is AMAZING in 2026."

print("\n==============================")
print("STAGE 0: ORIGINAL TEXT")
print("==============================")
print(raw_text)

# ==================================================
# STAGE 1: TEXT NORMALIZATION - LOWERCASE
# ==================================================
print("\nSTAGE 1: CONVERT TO LOWERCASE")
print("Description: Converting all characters to lowercase ensures consistency.")
print("This prevents the model from treating 'NLP' and 'nlp' as different words.")

text_lower = raw_text.lower()
print("Output:", text_lower)
#Output: the cats are running quickly towards the garden!! nlp is amazing in 2026.

# ==================================================
# STAGE 2: REMOVE NUMBERS AND PUNCTUATION
# ==================================================
print("\nSTAGE 2: REMOVE NUMBERS AND PUNCTUATION")
print("Description: Removing numbers and punctuation reduces noise.")
print("This ensures the model focuses only on meaningful words.")

text_clean = re.sub(r'[^a-z\s]', '', text_lower)
text_clean = re.sub(r'\s+', ' ', text_clean).strip()

print("Output:", text_clean)

#STAGE 2: REMOVE NUMBERS AND PUNCTUATION
#Description: Removing numbers and punctuation reduces noise.
#This ensures the model focuses only on meaningful words.
#Output: the cats are running quickly towards the garden nlp is amazing in

# ==================================================
# STAGE 3: TOKENIZATION
# ==================================================
print("\nSTAGE 3: TOKENIZATION")
print("Description: Tokenization splits the cleaned text into individual words.")
print("These tokens become the basic units for NLP processing.")

tokens = word_tokenize(text_clean)
print("Output:", tokens)

#STAGE 3: TOKENIZATION
#Description: Tokenization splits the cleaned text into individual words.
#These tokens become the basic units for NLP processing.
#Output: ['the', 'cats', 'are', 'running', 'quickly', 'towards', 'the', 'garden', 'nlp', 'is', 'amazing', 'in']

# ==================================================
# STAGE 4: STOPWORD REMOVAL
# ==================================================
print("\nSTAGE 4: STOPWORD REMOVAL")
print("Description: Stopwords are common words (e.g., 'the', 'is', 'are') that carry minimal meaning.")
print("Removing them reduces dimensionality and improves model efficiency.")

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

print("Output:", filtered_tokens)

#STAGE 4: STOPWORD REMOVAL
#Description: Stopwords are common words (e.g., 'the', 'is', 'are') that carry minimal meaning.
#Removing them reduces dimensionality and improves model efficiency.
#Output: ['cats', 'running', 'quickly', 'towards', 'garden', 'nlp', 'amazing']

# ==================================================
# STAGE 5A: STEMMING
# ==================================================
print("\nSTAGE 5A: STEMMING (PORTER STEMMER)")
print("Description: Stemming reduces words to their root form by removing suffixes.")
print("Note: The resulting word may not always be a valid dictionary word.")

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

print("Output:", stemmed_tokens)

#STAGE 5A: STEMMING (PORTER STEMMER)
#Description: Stemming reduces words to their root form by removing suffixes.
#Note: The resulting word may not always be a valid dictionary word.
#Output: ['cat', 'run', 'quickli', 'toward', 'garden', 'nlp', 'amaz']

# ==================================================
# STAGE 5B: LEMMATIZATION
# ==================================================
print("\nSTAGE 5B: LEMMATIZATION")
print("Description: Lemmatization reduces words to their dictionary base form.")
print("It is more accurate than stemming but computationally more expensive.")

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print("Output:", lemmatized_tokens)

#STAGE 5B: LEMMATIZATION
#Description: Lemmatization reduces words to their dictionary base form.
#It is more accurate than stemming but computationally more expensive.
#Output: ['cat', 'running', 'quickly', 'towards', 'garden', 'nlp', 'amazing']

# ==================================================
# FINAL OUTPUT COMPARISON
# ==================================================
print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

cleaned_text_stemmed = " ".join(stemmed_tokens)
cleaned_text_lemmatized = " ".join(lemmatized_tokens)

print("Original Text:   ", raw_text)
print("Stemmed Version: ", cleaned_text_stemmed)
print("Lemmatized Version:", cleaned_text_lemmatized)

print("\nText preprocessing completed successfully.")

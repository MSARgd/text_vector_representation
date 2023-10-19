#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:34:50 2023

@author: msa
"""
import pandas as pd
import numpy as np
import string
import Helper
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('stopwords')
stop_words = set(stopwords.words('french'))
corpus = [
    'Le chat dort sur le tapis',
    'Les Oiseaux Chantent Le Matin',
    'Le chien court dans le jardin',
]
df = pd.DataFrame({'corpus' : corpus})
print(df)

print(string.punctuation)
def code(texte) : 
    text_s_pon = ''.join( [c for c in texte if c not in string.punctuation])
    return text_s_pon

t_s_p =  string.ascii_letters + string.punctuation
t_s_p= Helper.spaced_text(t_s_p)
corpus.append(code(t_s_p))
for d in corpus:
    print(d)
print("==========Tokenization==============")
tokenizer = WhitespaceTokenizer()
t_s_p_tokenized = tokenizer.tokenize(corpus[3])
print(t_s_p_tokenized)



for i in range(len(corpus)) :
    corpus[i] =  tokenizer.tokenize(corpus[i])
    

print("==============Removing Stop Words =====================")
def delete_stop_words(corpus):
    def crop_sw(words):
        return [word for word in words if word.lower() not in stop_words]

    corpus_without_stopwords = [crop_sw(d) for d in corpus]
    return corpus_without_stopwords
corpus = delete_stop_words(corpus)
print(corpus)


print("============Stemming & lemmatization ======================")
stemmer = SnowballStemmer('french')
lemmatizer = WordNetLemmatizer()

stemmed_corpus = [[stemmer.stem(word) for word in d] for d in corpus]
lemmatized_corpus = [[lemmatizer.lemmatize(word) for word in d] for d in corpus]

print("Stemming Corpus :")
for d in stemmed_corpus:
    print(d)

print("lemmatization Corpus")
for d in lemmatized_corpus:
    print(d)
    
corpus_as_strings = [' '.join(d) for d in corpus]
print("============ CountVectorizer ======================")
vectorizer = CountVectorizer()

vectorizer.fit(corpus_as_strings)
print(vectorizer.vocabulary_)

vector = vectorizer.transform(corpus_as_strings)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())

print("============ TfidfVectorizer ======================")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_as_strings)

# Afficher la matrice TF-IDF
print("Matrice TF-IDF :")
print(tfidf_matrix.toarray())

print(tfidf_matrix[0].toarray())
print("===========Similarity==============")
# TfidfVectorizer Represenation 
similarity = cosine_similarity(tfidf_matrix[0].toarray(), tfidf_matrix[3].toarray())
print("similaritée entre 'chat' et 'chien' (TfidfVectorizer) ",similarity)

# CountVectorize Representation
chat_index = vectorizer.vocabulary_['chat']
chien_index = vectorizer.vocabulary_['chien']

vector_chat = vector.toarray()[:, chat_index]
vector_chien = vector.toarray()[:, chien_index]

# Calcul de la similarité de cosinus
similarity = cosine_similarity([vector_chat], [vector_chien])

print("similaritée entre 'chat' et 'chien' (CountVectorizer) ",similarity)






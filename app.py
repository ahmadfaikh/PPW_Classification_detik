
import streamlit as st
import pickle
from pickle import dump
from pickle import load
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer
import re
import string
import unicodedata
from string import punctuation


# Import library yang dibutuhkan untuk melakukan data preprocessing

st.title("Klasifikasi Berita Detik")
# Fungsi Cleaning data/Preprocessing


def hapus_kurung(text):
    return re.sub('\[[^]]*\]', '', text)


def delete_url(text):
    return re.sub(r'http/S+', '', text)


def remove_specialchars(text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


nltk.download('stopwords')
stop = set(stopwords.words('indonesian'))
punctuation = list(string.punctuation)
stop.update(punctuation)


def remove_stopwords(text):
    res = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            res.append(i.strip().lower())
    return " ".join(res)


def preprocessing_text(text):
    text = hapus_kurung(text)
    text = delete_url(text)
    text = remove_specialchars(text)
    text = remove_stopwords(text)
    return text


text = st.text_input("Input text")
button_clicked = st.button('Prediksi')
# load model
tfidf_fit = pickle.load(open('tfidf_vect.pkl', 'rb'))
clf = pickle.load(open('svm.pkl', 'rb'))


def prediksi(text):
    text = preprocessing_text(text)
    norm = tfidf_fit.transform([text])
    y_pred = clf.predict(norm)
    return y_pred


if button_clicked:
    prd = prediksi(text)
    st.write("Hasil Prediksi: ")
    st.write(prd)

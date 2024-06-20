import streamlit as st
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower() #lowercase transofomation
    text = nltk.word_tokenize(text) #tokenization

    y = []
    for i in text: #removing special characters
        if i.isalnum():
            y.append(i)

    text = y[:] #removing stopwords
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
            
    return " ".join(y)


#loading the vectorizer and naive model
Tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


#streamlit code
st.title("SMS SPAM CLASSIFIER")

input_sms = st.text_input("Enter text message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    
    vector_input = Tfidf.transform([transformed_sms])
    
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
        
    
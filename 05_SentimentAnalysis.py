# Import key libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import plotly.express as px

import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


def clean_words(text):
    ''' Remove stopwords '''

    sw = stopwords.words('english')
    sw.extend(['from', 'subject', 're', 'use', 'edu', 'will', 'aap', 'day', 'user', 'stock', 'today', 'week', 'year', 'https'])
    result = []

    for token in gensim.utils.simple_preprocess(text):
        if token not in sw and len(token) >= 3:
            result.append(token)

    return result


def clean_text(text):
    ''' Remove punctuation '''

    for char in string.punctuation:
        text = text.replace(char, '')

    return text


if __name__ == '__main__':
    ''' Main instance of the program '''

    verbose = False
    plot_wc = False

    print('Sentiment analysis module...')

    sentiment_file = 'data/stock_sentiment.csv'
    sentiment_data = pd.read_csv(sentiment_file)

    if verbose:
        print(sentiment_data.head())
        print(sentiment_data.info())
        print(sentiment_data.isnull().sum())

    # Check the number of unique elements
    #sns.countplot(sentiment_data['Sentiment'])
    print('N unique elements: ', sentiment_data['Sentiment'].nunique())

    print(sentiment_data['Text'].head())

    # Remove punctuation
    puncts = string.punctuation

    sentiment_data['TextClean'] = sentiment_data['Text'].apply(clean_text)
    #print(sentiment_data['TextClean'].head())
    sentiment_data['TextCleanStop'] = sentiment_data['TextClean'].apply(clean_words)
    print(sentiment_data['TextCleanStop'].head())
    sentiment_data['TextCleanStopJoin'] = sentiment_data['TextCleanStop'].apply(lambda x: " ".join(x)) 
    print(sentiment_data['TextCleanStopJoin'].head())
    #nltk.download('stopwords')
    #clean_words('')

    if plot_wc:
        plt.figure(figsize = (20, 20))

        # Positive sentiment
        #wc = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(sentiment_data[sentiment_data['Sentiment'] == 1]['TextCleanStopJoin']))

        # Negative sentiment
        wc = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(sentiment_data[sentiment_data['Sentiment'] == 0]['TextCleanStopJoin']))
        plt.imshow(wc)
        plt.show()
    
    # How to use the word tokenize function
    #nltk.word_tokenize(sentiment_data['TextCleanStopJoin'][0])
    
    # Find what's the maximum length of a sentence
    sentiment_data['Length'] = sentiment_data['TextCleanStopJoin'].apply(lambda x: len(nltk.word_tokenize(x)))
    n_max = sentiment_data['Length'].max()
    print('MaxNumWord: ', n_max)

    list_of_words = []

    # Append each word to the list of total words
    for row in sentiment_data['TextCleanStop']:
        for word in row:
            list_of_words.append(word)

    # Find the unique words
    n_total = len(list(set(list_of_words)))
    print('Total unique words: ', n_total)
    
    # Prepare the data for test train
    X = sentiment_data['TextCleanStop']
    y = sentiment_data['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    # Tokenize the train and test data
    tokenizer = Tokenizer(num_words = n_total)
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    # Pad the data to make it conform with the input size
    train_padded = pad_sequences(train_sequences, maxlen=n_max)
    test_padded = pad_sequences(test_sequences, maxlen=n_max)

    # Change the shape and make it categorical (with two colums)
    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)
    
    # Build the LSTM model
    model = Sequential()
    model.add(Embedding(n_total, output_dim=512))
    model.add(LSTM(256))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    # Fit the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    '''
    model.fit(train_padded, y_train_cat, batch_size=128, validation_split=0.2, epochs=2)

    # Test the model checking accuracy and confusion matrix
    prediction = model.predict(test_padded)
    
    predictions = []
    for p in prediction:
        predictions.append(np.argmax(p))
 
    realvalues = []
    for r in y_test_cat:
        realvalues.append(np.argmax(r))

    acc = accuracy_score(realvalues, predictions)
    print(acc)

    cm = confusion_matrix(realvalues, predictions)
    print(cm)

    sns.heatmap(cm, annot=True)
    #plt.show()
    '''

    # Compare with pre-trained BERT models
    from transformers import pipeline

    print('Using a pre-trained BERT model for sentiment analysis...')
    nlp = pipeline('sentiment-analysis')

    # Make a prediction 
    print('Predicting sentiment analysis values...')
    predictions = []
    for x in X_test.values:

        if x == []:
            x = [' ']
        
        x = ' '.join(x)

        p = nlp(x)

        #print(p)
        #print(type(p))
        val = p[0]['label']
        translate = {'POSITIVE':1, 'NEGATIVE':0}
        #print(val, translate[val])

        predictions.append(translate[val])

    realvalues = []
    for r in y_test_cat:
        realvalues.append(np.argmax(r))

    acc = accuracy_score(realvalues, predictions)
    print(acc)

    cm = confusion_matrix(realvalues, predictions)
    print(cm)

    print('Done.')

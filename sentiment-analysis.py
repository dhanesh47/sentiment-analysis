import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics

data = pd.read_csv('C:/Users/HP/Desktop/dataset.csv', encoding='latin-1')

def process_text(text):
    
    
    
    
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    
    return clean_words
data.head()
data['SentimentText'].apply(process_text)
x_train, x_test, y_train, y_test = train_test_split(data['SentimentText'],data['Sentiment'],test_size=0.2)
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=process_text)), # converts strings to integer counts
    ('tfidf',TfidfTransformer()), # converts integer counts to weighted TF-IDF scores
    ('classifier',MultinomialNB()) # train on TF-IDF vectors with Naive Bayes classifier
])
pipeline.fit(x_train,y_train)
predictions = pipeline.predict(x_test)
print(metrics.accuracy_score(y_test,predictions))
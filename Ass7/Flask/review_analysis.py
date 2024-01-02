
from flask import Flask, render_template, request  # type: ignore
import joblib
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import re
import string

app = Flask(__name__)

def clean_doc(doc):
    # Split into tokens by white space
    tokens = doc.split()
    # Prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # Remove punctuation from each word
    tokens = [re_punc.sub('',w) for w in tokens]
    # Remove remaining token that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    #filter out short tokens
    tokens = [word for word in tokens if len(word)>1]
    return tokens

def encode_docs(tokenizer,max_length,docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded,maxlen=max_length,padding='post')
    return padded

model = tf.keras.models.load_model('/home/dai/Documents/NLP & CV/Assignments/Assignment 7/flask/movie_review_model.keras')

tokenizer = joblib.load('/home/dai/Documents/NLP & CV/Assignments/Assignment 7/flask/movie_review_tokenizer.bin')

max_length = 1380

def predict_sentiment(review):
  line = clean_doc(review)
  padded = encode_docs(tokenizer, max_length, [line])
  yhat = model.predict(padded, verbose=0)
  percent_pos = yhat[0 ,0]
  if round(percent_pos) == 0:
    return (1-percent_pos), 'NEGATIVE'
  return percent_pos, "POSITIVE"


@app.route('/')
def student():
    return render_template('reviewdetector.html')

@app.route('/review', methods = ['GET','POST'])
def result():
    if request.method == 'POST':
        data = dict(request.form)
        review = str(data['review'])
        data['percent'], data['result'] = predict_sentiment(review)
        return render_template('review_output.html', data=data)
             
if __name__ == "__main__":
    app.run(debug = True)
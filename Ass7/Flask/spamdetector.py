from email import message
from flask import Flask, render_template, request  # type: ignore
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
wnl = WordNetLemmatizer()

app = Flask(__name__)

def clean_text(text):
    tokens1 = word_tokenize(text)
    tokens2 = [x.lower() for x in tokens1 if x.isalpha() or x.isdigit()]
    tokens3 = [x for x in tokens2 if x not in stopwords.words('english')]
    tokens4 = []
    tags = pos_tag(tokens3)
    for word in tags:
        if word[1].startswith('N'):
            tokens4.append(wnl.lemmatize(word[0], pos='n'))
        if word[1].startswith('V'):
            tokens4.append(wnl.lemmatize(word[0], pos='v'))
        if word[1].startswith('R'):
            tokens4.append(wnl.lemmatize(word[0], pos='r'))
        if word[1].startswith('J'):
            tokens4.append(wnl.lemmatize(word[0], pos='a'))
    
    return tokens4


classifier = joblib.load('/home/dai/Documents/NLP & CV/Dec20_NLP/spam_model.bin')
tfidf = joblib.load('/home/dai/Documents/NLP & CV/Dec20_NLP/vectorizer.bin')




@app.route('/')
def student():
    return render_template('spamdetector.html')

@app.route('/spamfinder', methods = ['GET','POST'])
def result():
    if request.method == 'POST':
        data = dict(request.form)
        message = tfidf.transform([data['message']])
        data['result'] = classifier.predict(message)[0]
        return render_template('spamoutput.html', data=data)
             
if __name__ == "__main__":
    app.run(debug = True)
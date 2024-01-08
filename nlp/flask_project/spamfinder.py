import string

import joblib
from flask import Flask, render_template, request
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

app = Flask(__name__)


punctuations = string.punctuation
stopword = stopwords.words('english')
wnl = WordNetLemmatizer()

def clean_text(text):
    tokens1 = word_tokenize(text)
    tokens2 = [x.lower() for x in tokens1 if x.isalpha() or x.isdigit()]
    tokens3 = [x for x in tokens2 if x not in stopword]

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
sent = "When we visited the gorund no on was playing."
clean_text(sent)

#load models
classifier = joblib.load('model.bin')
tfidf = joblib.load('vectorizer.bin')

@app.route('/')
def student():
   return render_template('spamdetector.html')

@app.route('/result',methods = ['POST', 'GET'])
def spamfinder():
   if request.method == 'POST':
    data = dict(request.form)
    message = tfidf.transform(data['message'])
    data['result'] = classifier.predict(message)[0]
    return render_template("results.html",result = data)

   
if __name__ == '__main__':
    app.run(debug = True)
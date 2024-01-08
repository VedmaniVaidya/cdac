import string
import joblib
from flask import Flask, render_template, request
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

def load_models():
    classifier = joblib.load('model.bin')
    tfidf = joblib.load('vectorizer.bin')
    return classifier, tfidf

def predict_spam(message, classifier, tfidf):
    vectorized_message = tfidf.transform([message])
    prediction = classifier.predict(vectorized_message)[0]
    return prediction

@app.route('/')
def student():
   return render_template('spamdetector.html')

@app.route('/spamfinder',methods = ['POST', 'GET'])
def spamfinder():
    classifier, tfidf = load_models()
    if request.method == 'POST':
        data = dict(request.form)
        message = data['message']
        data['result'] = predict_spam(message, classifier, tfidf)
        print(data)
        return render_template("results.html", data=data)

if __name__ == '__main__':
    app.run(debug = True)
    # classifier, tfidf = load_models()
    # print(predict_spam("Lottery winnings please collect", classifier, tfidf))
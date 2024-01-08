from flask import Flask
app = Flask(__name__)

@app.route('/well')
def welcome():
    return('Dogs')

@app.route('/good')
def good():
    return('Cats')

@app.route('/')
def home():
    return('404_Error')

if __name__ == '__main__':
    app.run(debug=True)
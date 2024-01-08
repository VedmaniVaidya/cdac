from flask import Flask
app = Flask(__name__)

@app.route('/square/<num>')
def hello_me(num):
    sq = int(num)**2
    return'Square of %s is <b>%d</b>' %(num,sq)

if __name__ == '__main__':
    app.run(debug=True)
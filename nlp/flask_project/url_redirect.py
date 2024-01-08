from flask import Flask, redirect, url_for
from flask import render_template
app = Flask(__name__)

@app.route('/')
def login():
    return render_template('./simple_html.html')

@app.route('/admin')
def hello_admin():
    return('Hello Admin')

@app.route('/guest/<guest>')
def hello_guest(guest):
    return('Hello <b>%s</b> as guest' %guest)

@app.route('/user/<name>')
def hello_user(name):
    if name == 'admin':
        return redirect(url_for('hello_admin'))
    else:
        return redirect(url_for('hello_guest',guest=name))

if __name__ == '__main__':
    app.run(debug=True)
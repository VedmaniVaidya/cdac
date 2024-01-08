from flask import Flask, redirect, url_for,request,render_template
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('simple_html.html')


@app.route('/success/<name>')
def success(name):
    return 'Welcome %s!' %name

@app.route('/admin')
def hello_admin():
    return('Hello Admin')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success',name=user))

if __name__ == '__main__':
    app.run(debug=True)
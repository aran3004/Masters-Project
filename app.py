from flask import Flask, render_template

app = Flask(__name__)
app.secret_key = 'my_masters_project_code'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/mnist')
def mnist():
    return render_template('mnist.html')

@app.route('/contribute')
def contribute():
    return render_template('contribute.html')

if __name__ == '__main__':
    app.run(debug=True)
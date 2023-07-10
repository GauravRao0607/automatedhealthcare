from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template('sickness.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    description = [request.form['description']]
    description_vectorized = vectorizer.transform(description)
    prediction=model.predict(description_vectorized)
    return render_template('sickness.html',pred='You likely have a case of {}'.format(prediction[0]))

@app.route('/about.html')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)

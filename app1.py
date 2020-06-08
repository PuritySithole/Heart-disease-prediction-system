from flask import Flask,render_template,url_for,request,flash
#import pandas as pd
from sklearn.externals import joblib
import numpy as np
import pickle
import os


app = Flask(__name__, template_folder='templates')
#model = pickle.load(open('model_naive.pkl', 'rb'))
@app.route('/')
def home():
	return render_template('app.html')

def getParameters():
    parameters = []
    #parameters.append(request.form('name'))
    parameters.append(request.form['Age'])
    parameters.append(request.form['Serum Cholesterol'])
    parameters.append(request.form['Trestbps'])
    parameters.append(request.form['Thalach'])
    parameters.append(request.form['Old Peak'])
    
    
    return parameters

@app.route('/predict',methods=['POST'])   
def predict():
    if request.method=='POST':
        parameters= getParameters()
        x = np.array(parameters).reshape(1, -1)
     


        scaler_path = os.path.join('scaler.pkl')
        scaler = None
        with open(scaler_path, 'rb') as f:
          scaler = pickle.load(f)

        inputFeature = scaler.transform(x)

        '''model_path = os.path.join(os.path.dirname(__file__), 'model_naive.sav')
        clf = joblib.load(model_path)'''

        y = clfr.predict(inputFeature.reshape(-1,1))
    return render_template('app.html',prediction = y[1])

if __name__ == '__main__':
    #model = pickle.load(open('C:/Users/Ree/Desktop/prototyping/model_naive.pkl', 'rb'))
    clfr=joblib.load('model_naive.sav')
    app.run(debug=True)





'''
@app.route('/predict',methods=['POST'])
def predict():
    model = pickle.load(open('model_naive_test.pkl', 'rb'))
    #clfr = joblib.load(model)

@app.route('/print',methods=['POST','GET'])
def displayParameters():
    if request.method=='POST':
        parameters= getParameters()
        inputFeature = np.asarray(parameters).reshape(-1,-1)
        my_prediction = clfr.predict(inputFeature)
    return render_template('app.html',my_prediction=my_prediction)  

    if request.method == 'POST':
        parameters = getParameters()
        inputFeature = np.asarray(parameters).reshape(1,-1)
        my_prediction = model.predict(inputFeature)
    return render_template('index.html',prediction = int(my_prediction[0]))

     scaler_path = os.path.join(os.path.dirname(__file__), 'model_naive.pkl')
    scaler = None
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    #x = scaler.transform(x)

    model_path = os.path.join(os.path.dirname(__file__), 'model_naive.sav')
    clf = joblib.load(model_path)
'''

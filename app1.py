from flask import Flask,render_template,url_for,request,flash
#import pandas as pd
import joblib
import numpy as np
import pickle
import os



app = Flask(__name__, template_folder='templates')
app.secret_key = c'_100#y2L"remembrancez\n\xec]/'
#model = pickle.load(open('model_naive.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

def getUserInput():
    user_input = []
    #parameters.append(request.form('name'))
    user_input.append(request.form['Age'])
    user_input.append(request.form['Serum Cholesterol'])
    user_input.append(request.form['Trestbps'])
    user_input.append(request.form['Thalach'])
    user_input.append(request.form['Old Peak'])
    
    
    return user_input

@app.route('/predict',methods=['POST'])   
def predict():
    if request.method=='POST':
        user_input= getUserInput()
        x = np.array(user_input).reshape(1, -1)
     


        scaler_path = os.path.join('scaler.pkl')
        scaler = None
        with open(scaler_path, 'rb') as f:
          scaler = pickle.load(f)

        inputFeature = scaler.transform(x)

        '''model_path = os.path.join(os.path.dirname(__file__), 'model_naive.sav')
        clf = joblib.load(model_path)'''

        prediction = clfr.predict(inputFeature.reshape(-1,1))
        predict = np.sort(prediction)[::-1]
        if predict[0]==1:
            flash("You have been diagnosed of a heart disease","warning")
        else:
           flash("You haven't been diagnosed of a heart disease","success")
   
              
          
    return render_template('index.html')

if __name__ == '__main__':
    clfr=joblib.load('model_naive.sav')
    app.run(debug=True)






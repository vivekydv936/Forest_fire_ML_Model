from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
# import ridge regressor and scaler pickle
ridge_model = pickle.load(open('D:/Me/DataScience/JupyterFile/KRISH/ForestFire_Project/models/ridg.pkl', 'rb'))
scaler_model = pickle.load(open('D:/Me/DataScience/JupyterFile/KRISH/ForestFire_Project/models/scaler.pkl', 'rb'))



application =Flask(__name__)
app=application

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_scaled_data=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_scaled_data)
        return render_template("home.html",results=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
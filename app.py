from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.mlops_project.pipeline.prediction import ModelInference
from src.mlops_project.config.configuration import ConfigurationManager
from src.mlops_project.pipeline.prediction import ModelInferencePipeline

prediction_pipeline = ModelInferencePipeline()

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in web UI
def index():
    if request.method == 'POST':
        try:
            sex = float(request.form['sex'])
            age = float(request.form['age'])
            education = float(request.form['education']) 
            smokingStatus = float(request.form['smokingStatus']) 
            cigsPerDay = float(request.form['cigsPerDay']) 
            BPMeds = float(request.form['BPMeds']) 
            prevalentStroke = float(request.form['prevalentStroke']) 
            prevalentHyp = float(request.form['prevalentHyp']) 
            diabetes = float(request.form['diabetes'])
            totChol = float(request.form['totChol']) 
            sysBP = float(request.form['sysBP']) 
            diaBP = float(request.form['diaBP']) 
            BMI = float(request.form['BMI']) 
            heartRate = float(request.form['heartRate']) 
            glucose = float(request.form['glucose']) 
       
         
            data = [sex, age, education, smokingStatus,
                    cigsPerDay, BPMeds, prevalentStroke, 
                    prevalentHyp, diabetes, totChol, sysBP,
                    diaBP, BMI, heartRate, glucose]
            data = np.array(data).reshape(1, 15)
            
            predict = prediction_pipeline.predict(data)
            predict = predict.item()
            if predict == 1:
                p_str = "The model predicts that this sample is at risk for chronic heart disease."
            else:
                p_str = "The model predicts that this sample is NOT at risk for chronic heart disease."

            return render_template('results.html', prediction = str(p_str))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)
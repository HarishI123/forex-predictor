from flask import Flask, render_template,request
import requests
import pandas
import utils
from utils import preprocessdata

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict/', methods=['GET', 'POST'])

def predict():  
    if request.method == 'POST': 
        fromC = request.form.get('fromC')  
        toC = request.form.get('to')  
        days = request.form.get('days')  


        prediction = utils.preprocessdata(fromC,toC,days)
        print(prediction)

    return render_template('predict.html', prediction=prediction) 


if __name__ == '__main__':
    app.run()
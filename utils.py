import numpy as np 
import requests
import pandas
from keras.models import load_model
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras import Sequential
from sklearn.model_selection import train_test_split
from flask import render_template


# model = load_model('humanAI.h5') 
def preprocessdata(fromC,toC,days):
    url = "https://alpha-vantage.p.rapidapi.com/query"
    fromC= fromC.upper()
    toC = toC.upper()

    querystring = {"from_symbol":fromC,"function":"FX_DAILY","to_symbol":toC,"outputsize":"compact","datatype":"json"}

    headers = {
        "X-RapidAPI-Key": "8e92971510msh36af92bf67f8c91p196a61jsn41a617231b1a",
        "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
    }


    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status() 
        data = response.json()
        data = pandas.DataFrame(data['Time Series FX (Daily)'])
        data= data.transpose()
        print(data)
        # data = data.reset_index()
        # data = data.drop(["index"],1)
        data = data.iloc[:int(days),:]
        data = data.astype("float32")
        x = data.iloc[:,:-1]
        y = data.iloc[:,-1:]
        X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.5)
        model = Sequential()
        model.add(Dense(10,activation="relu"))
        model.add(Dense(100,activation="relu"))
        model.add(Dense(200,activation="relu"))
        model.add(Dense(100,activation="relu"))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam',metrics="mse")
        model.fit(X_train,y_train,epochs=100)
        y_p = model.predict(X_test)
        print(y_p[0][0]) 
        return y_p[0][0]
    except requests.exceptions.RequestException as e:
        print(e)
        return f'An error occurred: {str(e)}'



# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

import matplotlib as plt

import pandas as pd
import visuals as vs

in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

outcome=full_data['Survived']
data=full_data.drop('Survived',axis=1)


def accuracy_score(truth,pred):
    if(len(truth)==len(pred)):
        return "Prediction = {:.2f}%".format((truth==pred).mean()*100)
    else:
        return "Number of Prediction Does Not Match"
    
prediction=pd.Series(np.ones(5,dtype=int))
print(accuracy_score(outcome[:5],prediction))


def prediction_0(data):
    predictions=[]
    for _,passenger in data.iterrows():
        predictions.append(1)
    return pd.Series(predictions)

predictions=prediction_0(data)

print(accuracy_score(outcome[:5],predictions[:5]))

vs.survival_stats(data, outcome, 'Sex')

def predictions_1(data):
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex']=="male" and passenger['Pclass']==3:
            predictions.append(0)
        elif  passenger['Sex']=="female":
            if passenger['Pclass'] == 3 and passenger['Fare'] <60 and passenger['Age'] < 50 :
               predictions.append(0)
            else:
               predictions.append(1)
        else:
            if passenger['Age']<10:
               predictions.append(1)
            elif passenger['Fare'] > 275:
                predictions.append(1)
            else:
                predictions.append(0)
    
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)

print(accuracy_score(outcome,predictions))

vs.survival_stats(data, outcome, 'Pclass', ["Sex == 'male'","SibSp > 0"])



vs.survival_stats(data, outcome, 'Pclass',["Sex == 'female'","Pclass == 3"])

vs.survival_stats(data, outcome, 'Sex')

vs.survival_stats(data, outcome, 'Pclass',["Sex == 'male'","Age > 15","Pclass == 3","Fare > 55"])

vs.survival_stats(data, outcome, 'Pclass',["Sex == 'male'","Age > 10"])

vs.survival_stats(data, outcome, 'Pclass',["Sex == 'male'"])
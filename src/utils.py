import os 
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import pickle

import dill 
from sklearn.metrics import r2_score   
from sklearn.model_selection import GridSearchCV 

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models,params):
    """
    This function evaluates the performance of different regression models.
    It returns a dictionary with model names as keys and their R2 scores as values.
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            params_for_model = params[list(models.keys())[i]]

            gs = GridSearchCV(model,params_for_model,cv =3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score  
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    This function loads a saved object from a file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train, X_test,y_test,models,param):
    try:
        report = {}

        # for i in range(len(list(models))):
        #     model = list(models.values())[i]
        #     para = param[list(models.keys())[i]]
        for model_name, model in models.items():
            # Use the correct model name to access parameters
            para = param[model_name]

            gs = GridSearchCV(model,para, cv=3,refit=True)
            gs.fit(X_train,y_train)
            
            # model.fit(X_train, y_train) # Train model
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred= model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)

            test_model_score = r2_score(y_test,y_test_pred)

            # report[list(models.keys())[i]] = test_model_score
            report[model_name] = test_model_score
            


        return report
    
    except Exception as e:
        raise CustomException(e,sys)
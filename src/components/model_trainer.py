import numpy as np 
import pandas as pd


#Modeling 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor

import os 
import sys 

from src.exception import CustomException
from dataclasses import dataclass
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('split training and test input data')
            x_train , y_train , x_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "KNeighbors Regressor":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest Regressor":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor(),
            }

            model_report:dict= evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,
                                              y_test=y_test,models=models )
            

            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("Best model not found")
            
            logging.info("Best model found on both training and testing dataset")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test,predicted)

            return r2_square
            

            
        except Exception as e:

            raise CustomException(e,sys)

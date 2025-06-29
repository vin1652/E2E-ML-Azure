import os 
import sys
import pandas as pd
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor
                              , GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model   

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()    

    def initiate_model_trainer(self, train_array, test_array):  
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )

            logging.info("Training the model")
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'XGBRegressor': XGBRegressor(eval_metric='logloss'),
                'CatBoostRegressor': CatBoostRegressor(verbose=False)
            }

            params = { 'LinearRegression': {},
                'Lasso': {'alpha': [0.1, 0.5, 1.0]},
                'Ridge': {'alpha': [0.1, 0.5, 1.0]},
                'KNeighborsRegressor': {'n_neighbors': [3, 5, 7, 9]},
                'DecisionTreeRegressor': {'max_depth': [None, 5, 10, 15]},
                'RandomForestRegressor': {'n_estimators': [50, 100, 200]},
                'GradientBoostingRegressor': {'n_estimators': [50, 100, 200]},
                'AdaBoostRegressor': {'n_estimators': [50, 100, 200]},
                'XGBRegressor': {'n_estimators': [50, 100, 200]},
                'CatBoostRegressor': {'iterations': [50, 100], 'depth': [6, 8]}
            }

            model_report : dict = evaluate_model(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test, 
                models=models,
                params=params
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)   
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)  
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score} on both train and test data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square 

           
        except Exception as e:
            raise CustomException(e, sys)

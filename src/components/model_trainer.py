import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging


from src.utils import save_object,evaluate_models


#in every file we have to make config class where we can define all the input requires


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]


            )
            models={
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "Gradient Boosting":GradientBoostingRegressor()

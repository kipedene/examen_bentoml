
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class TrainModel:
    def __init__(self, data_path="data/processed"):
        self.DATA_PATH = data_path
    
    def load_data(self):
        X_train_scaled = pd.read_csv(f"{self.DATA_PATH}/X_train_scaled.csv")
        X_test_scaled = pd.read_csv(f"{self.DATA_PATH}/X_test_scaled.csv")
        y_train = pd.read_csv(f"{self.DATA_PATH}/y_train.csv")
        y_test = pd.read_csv(f"{self.DATA_PATH}/y_test.csv")

        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def training(self, X_train, y_train):
        model = LinearRegression()
        param_grid = {'fit_intercept': [True, False]}
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=5)
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_, grid_search.best_estimator_
    
    def evaluation(self, best_model, X_train, X_test, y_train, y_test):
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)

        print("\nÉvaluation du modèle :")
        print(f"R² (Entraînement) : {train_r2:.2f}")
        print(f"R² (Test) : {test_r2:.2f}")
        print(f"MSE (Entraînement) : {train_mse:.4f}")
        print(f"MSE (Test) : {test_mse:.4f}")

        return train_r2, test_r2, train_mse, test_mse
    

def main():
    train_model = TrainModel()
    X_train, X_test, y_train, y_test = train_model.load_data()
    best_params, best_model = train_model.training(X_train, y_train)
    train_r2, test_r2, train_mse, test_mse = train_model.evaluation(best_model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()





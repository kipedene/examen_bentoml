import bentoml
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib



class LinearRegressionTrainer:
    def __init__(self, data_path="data/processed"):
        """
        Initialisation de la classe avec le chemin des données et la colonne cible.
        """
        self.DATA_PATH= data_path
        self.model = None
        self.best_params = None
        self.scaler = joblib.load("models/scaler.pkl")
    
    def load_data(self):
        """
        Chargement des données déjà préparées.
        """
        X_train_scaled = pd.read_csv(f"{self.DATA_PATH}/X_train_scaled.csv")
        X_test_scaled = pd.read_csv(f"{self.DATA_PATH}/X_test_scaled.csv")
        y_train = pd.read_csv(f"{self.DATA_PATH}/y_train.csv")
        y_test = pd.read_csv(f"{self.DATA_PATH}/y_test.csv")

        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Entraînement du modèle avec GridSearchCV.
        """
        model = LinearRegression()
        param_grid = {'fit_intercept': [True, False]}
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=5)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        print("Meilleurs paramètres trouvés :", self.best_params)
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """
        Évaluation du modèle et affichage des métriques.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)

        print("\nÉvaluation du modèle :")
        print(f"R² (Entraînement) : {train_r2:.2f}")
        print(f"R² (Test) : {test_r2:.2f}")
        print(f"MSE (Entraînement) : {train_mse:.4f}")
        print(f"MSE (Test) : {test_mse:.4f}")
    
    def save_model(self, model_name):
        """
        Sauvegarde du modèle avec BentoML.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")

        # Enregistrement du modèle avec BentoML
        bentoml_model = bentoml.sklearn.save_model(model_name, self.model)
        print(f"Modèle sauvegardé avec BentoML sous : {bentoml_model}")

    def predict(self, X_input):
        """
        Prédiction avec scaling des données d'entrée.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        
        # Appliquer le scaler sur les données d'entrée
        X_input_scaled = self.scaler.transform(X_input)
        
        # Faire des prédictions
        predictions = self.model.predict(X_input_scaled)
        return predictions


def main():
    trainer = LinearRegressionTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    trainer.train_model(X_train, y_train)
    trainer.evaluate_model( X_train, X_test, y_train, y_test)
    trainer.save_model("admission_lr")

if __name__ == "__main__":
    main()





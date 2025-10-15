import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """Save an object to a file using dill serialization.
    
    Args:
        file_path (str): Path where the object should be saved
        obj: Object to be saved
        
    Raises:
        CustomException: If saving fails
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Load an object from a file using dill deserialization.
    
    Args:
        file_path (str): Path to the file containing the object
        
    Returns:
        The loaded object
        
    Raises:
        CustomException: If loading fails
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """Evaluate multiple models using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        models: Dictionary of models to evaluate
        params: Dictionary of parameters for each model
        
    Returns:
        Dictionary with model names and their R² scores
        
    Raises:
        CustomException: If evaluation fails
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            # Create a gridsearch cross validation object with three fold cross validation
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            # Training the model
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)

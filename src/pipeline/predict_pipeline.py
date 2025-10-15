import sys
import pandas as pd
from typing import Dict, Any, List

from src.exception import CustomException
from src.utils import load_object

class AcademicPerformancePredictor:
    """
    Academic Performance Prediction Pipeline
    
    This class handles the complete prediction pipeline for academic performance
    using trained machine learning models and preprocessing components.
    """
    
    def __init__(self):
        """Initialize the academic performance predictor."""
        pass

    def predict(self, features: pd.DataFrame) -> List[float]:
        """
        Predict academic performance based on input features.
        
        Args:
            features (pd.DataFrame): Input features for prediction
            
        Returns:
            List[float]: Predicted academic performance scores
            
        Raises:
            CustomException: If prediction fails
        """
        try:
            model_path = 'artifacts/academic_performance_model.pkl'
            preprocessor_path = 'artifacts/academic_preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            scaled_data = preprocessor.transform(features)
            predictions = model.predict(scaled_data)
            return predictions
        except Exception as e:
            raise CustomException(e, sys)

class AcademicDataInput:
    """
    Academic Data Input Handler
    
    Responsible for mapping HTML form inputs to structured data format
    for academic performance prediction.
    """
    
    def __init__(self, gender: str, race_ethnicity: str,
                 parental_level_of_education: str, lunch: str,
                 test_preparation_course: str, reading_score: str, writing_score: str):
        """
        Initialize academic data input with student information.
        
        Args:
            gender (str): Student's gender
            race_ethnicity (str): Student's race/ethnicity group
            parental_level_of_education (str): Parent's education level
            lunch (str): Lunch program type
            test_preparation_course (str): Test preparation course status
            reading_score (str): Reading test score
            writing_score (str): Writing test score
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Convert input data to pandas DataFrame format.
        
        Returns:
            pd.DataFrame: Structured data for prediction
            
        Raises:
            CustomException: If data conversion fails
        """
        try:
            academic_data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(academic_data_dict)
        except Exception as e:
            raise CustomException(e, sys)
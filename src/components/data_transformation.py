# Academic Performance Prediction using Machine Learning - Data Transformation Module
# Handles categorical and numerical feature transformation, data cleaning, and feature engineering

import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class AcademicDataTransformationConfig:
    """
    Configuration class for Academic Data Transformation.
    
    Defines file paths for preprocessing objects and transformation artifacts.
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', "academic_preprocessor.pkl")

class AcademicDataTransformation:
    """
    Academic Data Transformation Handler
    
    This class manages the transformation of academic performance data,
    including categorical encoding, numerical scaling, and preprocessing pipeline creation.
    """
    
    def __init__(self):
        """Initialize the academic data transformation with configuration."""
        self.data_transformation_config = AcademicDataTransformationConfig()
    
    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline for academic performance data transformation.
        
        Returns:
            ColumnTransformer: Complete preprocessing pipeline for academic data
            
        Raises:
            CustomException: If transformer creation fails
        """
        try:
            # Define numerical and categorical columns for academic performance prediction
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            # Create numerical pipeline for academic performance features
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Academic performance numerical columns standardization completed")

            # Create categorical pipeline for academic performance features
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Academic performance categorical columns encoding completed")

            # Combine numerical and categorical pipelines
            academic_preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return academic_preprocessor
        except Exception as e:
            logging.error(f"Error creating academic data transformer: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Initiate the complete data transformation process for academic performance prediction.
        
        Args:
            train_path (str): Path to training data file
            test_path (str): Path to testing data file
            
        Returns:
            Tuple[np.ndarray, np.ndarray, str]: Transformed training array, testing array, and preprocessor path
            
        Raises:
            CustomException: If data transformation fails
        """
        try:
            # Load academic performance training and testing datasets
            academic_train_df = pd.read_csv(train_path)
            academic_test_df = pd.read_csv(test_path)

            logging.info("Successfully loaded academic performance training and testing datasets")

            # Create preprocessing object for academic performance data
            logging.info("Creating academic performance preprocessing object")
            academic_preprocessing_obj = self.get_data_transformer_object()

            # Define target variable for academic performance prediction
            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']

            # Separate input features and target variable for training data
            academic_input_feature_train_df = academic_train_df.drop(columns=[target_column_name], axis=1)
            academic_target_feature_train_df = academic_train_df[target_column_name]

            # Separate input features and target variable for testing data
            academic_input_feature_test_df = academic_test_df.drop(columns=[target_column_name], axis=1)
            academic_target_feature_test_df = academic_test_df[target_column_name]

            logging.info("Applying academic performance preprocessing to training and testing datasets")

            # Transform training and testing features
            academic_input_feature_train_array = academic_preprocessing_obj.fit_transform(academic_input_feature_train_df)
            academic_input_feature_test_array = academic_preprocessing_obj.transform(academic_input_feature_test_df)

            # Combine transformed features with target variables
            academic_train_array = np.c_[academic_input_feature_train_array, np.array(academic_target_feature_train_df)]
            academic_test_array = np.c_[academic_input_feature_test_array, np.array(academic_target_feature_test_df)]

            logging.info("Academic performance preprocessing object saved successfully")

            # Save the preprocessing object for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=academic_preprocessing_obj
            )

            return (
                academic_train_array,
                academic_test_array,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error(f"Error during academic data transformation: {str(e)}")
            raise CustomException(e, sys)

# Academic Performance Prediction using Machine Learning - Training Pipeline

import sys
import os
from typing import Tuple
import numpy as np

from src.components.academic_data_ingestion import AcademicDataIngestion
from src.components.academic_data_transformation import AcademicDataTransformation
from src.components.academic_model_trainer import AcademicModelTrainer
from src.exception import CustomException
from src.logger import logging

class AcademicTrainingPipeline:
    """
    Academic Performance Training Pipeline
    
    This class orchestrates the complete training pipeline for academic performance
    prediction, including data ingestion, transformation, and model training.
    """
    
    def __init__(self):
        """Initialize the academic training pipeline."""
        pass

    def initiate_training_pipeline(self) -> float:
        """
        Initiate the complete academic performance training pipeline.
        
        Returns:
            float: R² score of the trained model
            
        Raises:
            CustomException: If training pipeline fails
        """
        try:
            logging.info("Starting Academic Performance Prediction Training Pipeline")
            
            # Step 1: Data Ingestion
            logging.info("Step 1: Initiating academic data ingestion")
            academic_data_ingestion = AcademicDataIngestion()
            train_data_path, test_data_path = academic_data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            logging.info("Step 2: Initiating academic data transformation")
            academic_data_transformation = AcademicDataTransformation()
            train_array, test_array, preprocessor_path = academic_data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            # Step 3: Model Training
            logging.info("Step 3: Initiating academic model training")
            academic_model_trainer = AcademicModelTrainer()
            model_r2_score = academic_model_trainer.initiate_model_trainer(train_array, test_array)
            
            logging.info("Academic Performance Prediction Training Pipeline completed successfully")
            logging.info(f"Final Model R² Score: {model_r2_score:.4f}")
            
            return model_r2_score
            
        except Exception as e:
            logging.error(f"Error in academic training pipeline: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Initialize and run the academic training pipeline
    academic_training_pipeline = AcademicTrainingPipeline()
    final_score = academic_training_pipeline.initiate_training_pipeline()
    print(f"Academic Performance Model Training Complete - R² Score: {final_score:.4f}")

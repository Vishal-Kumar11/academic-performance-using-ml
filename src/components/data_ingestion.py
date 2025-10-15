import os
import sys
import numpy as np
from typing import Tuple
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import AcademicDataTransformation, AcademicDataTransformationConfig
from src.components.model_trainer import AcademicModelTrainerConfig, AcademicModelTrainer

@dataclass
class AcademicDataIngestionConfig:
    """
    Configuration class for Academic Data Ingestion.
    
    Defines file paths for training, testing, and raw data storage.
    """
    train_data_path: str = os.path.join('artifacts', "academic_train.csv")
    test_data_path: str = os.path.join('artifacts', "academic_test.csv")
    raw_data_path: str = os.path.join('artifacts', "academic_raw_data.csv")

class AcademicDataIngestion:
    """
    Academic Data Ingestion Handler
    
    This class manages the ingestion of academic performance data,
    including data loading, preprocessing, and train-test splitting.
    """
    
    def __init__(self):
        """Initialize the academic data ingestion with configuration."""
        self.ingestion_config = AcademicDataIngestionConfig()
    
    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Initiate the academic data ingestion process.
        
        Returns:
            Tuple[str, str]: Paths to training and testing data files
            
        Raises:
            CustomException: If data ingestion fails
        """
        logging.info("Starting academic data ingestion process")
        try:
            # Load academic performance dataset
            academic_dataframe = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info("Successfully loaded academic performance dataset")

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data for reference
            academic_dataframe.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw academic data saved to: {self.ingestion_config.raw_data_path}")
            
            # Perform train-test split for academic performance prediction
            logging.info("Initiating train-test split for academic performance data")
            training_dataset, testing_dataset = train_test_split(
                academic_dataframe, 
                test_size=0.2, 
                random_state=42
            )

            # Save training and testing datasets
            training_dataset.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            testing_dataset.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Academic data ingestion completed successfully")
            logging.info(f"Training data saved to: {self.ingestion_config.train_data_path}")
            logging.info(f"Testing data saved to: {self.ingestion_config.test_data_path}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error during academic data ingestion: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Initialize academic data ingestion
    academic_data_ingestion = AcademicDataIngestion()
    train_data_path, test_data_path = academic_data_ingestion.initiate_data_ingestion()

    # Initialize academic data transformation
    academic_data_transformation = AcademicDataTransformation()
    train_array, test_array, _ = academic_data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )

    # Initialize academic model training
    academic_model_trainer = AcademicModelTrainer()
    print(f"Academic Performance Model R² Score: {academic_model_trainer.initiate_model_trainer(train_array, test_array)}")
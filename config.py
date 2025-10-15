# Academic Performance Prediction using Machine Learning - Configuration

import os
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class AcademicConfig:
    """
    Configuration class for Academic Performance Prediction System.
    
    This class contains all configuration parameters for the academic
    performance prediction system, including file paths, model parameters,
    and system settings.
    """
    
    # Project Information
    PROJECT_NAME: str = "Academic Performance Prediction using Machine Learning"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Your Name"
    
    # File Paths
    DATA_DIR: str = "notebook/data"
    ARTIFACTS_DIR: str = "artifacts"
    LOGS_DIR: str = "logs"
    MODELS_DIR: str = "artifacts"
    VISUALIZATIONS_DIR: str = "artifacts/visualizations"
    
    # Data Files
    RAW_DATA_FILE: str = "StudentsPerformance.csv"
    TRAIN_DATA_FILE: str = "academic_train.csv"
    TEST_DATA_FILE: str = "academic_test.csv"
    RAW_DATA_PATH: str = os.path.join(ARTIFACTS_DIR, "academic_raw_data.csv")
    
    # Model Files
    MODEL_FILE: str = "academic_performance_model.pkl"
    PREPROCESSOR_FILE: str = "academic_preprocessor.pkl"
    MODEL_COMPARISON_FILE: str = "model_comparison_results.csv"
    
    # Model Parameters
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CROSS_VALIDATION_FOLDS: int = 5
    
    # Hyperparameter Ranges
    RANDOM_FOREST_PARAMS: Dict[str, List[Any]] = None
    XGBOOST_PARAMS: Dict[str, List[Any]] = None
    CATBOOST_PARAMS: Dict[str, List[Any]] = None
    LIGHTGBM_PARAMS: Dict[str, List[Any]] = None
    
    # Validation Rules
    MIN_SCORE: int = 0
    MAX_SCORE: int = 100
    MAX_BATCH_SIZE: int = 1000
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # Web Application Settings
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 5000
    FLASK_DEBUG: bool = False
    
    # Feature Names
    NUMERICAL_FEATURES: List[str] = None
    CATEGORICAL_FEATURES: List[str] = None
    TARGET_COLUMN: str = "math_score"
    
    def __post_init__(self):
        """Initialize derived configuration values."""
        # Hyperparameter ranges
        self.RANDOM_FOREST_PARAMS = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        self.XGBOOST_PARAMS = {
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        self.CATBOOST_PARAMS = {
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.1, 0.15],
            'iterations': [100, 200, 300, 500],
            'l2_leaf_reg': [1, 3, 5, 7]
        }
        
        self.LIGHTGBM_PARAMS = {
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'num_leaves': [15, 31, 63, 127]
        }
        
        # Feature definitions
        self.NUMERICAL_FEATURES = ['writing_score', 'reading_score']
        self.CATEGORICAL_FEATURES = [
            'gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course'
        ]
    
    def get_model_file_path(self) -> str:
        """Get the full path to the model file."""
        return os.path.join(self.MODELS_DIR, self.MODEL_FILE)
    
    def get_preprocessor_file_path(self) -> str:
        """Get the full path to the preprocessor file."""
        return os.path.join(self.MODELS_DIR, self.PREPROCESSOR_FILE)
    
    def get_train_data_path(self) -> str:
        """Get the full path to the training data file."""
        return os.path.join(self.ARTIFACTS_DIR, self.TRAIN_DATA_FILE)
    
    def get_test_data_path(self) -> str:
        """Get the full path to the test data file."""
        return os.path.join(self.ARTIFACTS_DIR, self.TEST_DATA_FILE)
    
    def get_raw_data_path(self) -> str:
        """Get the full path to the raw data file."""
        return os.path.join(self.DATA_DIR, self.RAW_DATA_FILE)
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.ARTIFACTS_DIR,
            self.LOGS_DIR,
            self.MODELS_DIR,
            self.VISUALIZATIONS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def validate_configuration(self) -> bool:
        """
        Validate the configuration settings.
        
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Check if required directories exist or can be created
            self.create_directories()
            
            # Validate numeric ranges
            if not (0 < self.TEST_SIZE < 1):
                raise ValueError("TEST_SIZE must be between 0 and 1")
            
            if self.RANDOM_STATE < 0:
                raise ValueError("RANDOM_STATE must be non-negative")
            
            if self.CROSS_VALIDATION_FOLDS < 2:
                raise ValueError("CROSS_VALIDATION_FOLDS must be at least 2")
            
            # Validate score ranges
            if self.MIN_SCORE >= self.MAX_SCORE:
                raise ValueError("MIN_SCORE must be less than MAX_SCORE")
            
            # Validate batch size
            if self.MAX_BATCH_SIZE <= 0:
                raise ValueError("MAX_BATCH_SIZE must be positive")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

# Global configuration instance
config = AcademicConfig()

# Validate configuration on import
if not config.validate_configuration():
    raise RuntimeError("Invalid configuration detected")

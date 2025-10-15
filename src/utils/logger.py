# Academic Performance Prediction using Machine Learning - Logging Configuration

import os
import sys
import logging
from datetime import datetime
from typing import Optional
import traceback

class AcademicLogger:
    """
    Academic Performance Prediction Logging Handler
    
    This class provides comprehensive logging configuration for the academic
    performance prediction system, including file and console logging with
    different log levels and formatting.
    """
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize the academic logger.
        
        Args:
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file (str, optional): Custom log file path
        """
        self.log_level = log_level.upper()
        self.log_file = log_file or self._get_default_log_file()
        self.logger = self._setup_logger()
        
    def _get_default_log_file(self) -> str:
        """
        Get default log file path for academic performance prediction.
        
        Returns:
            str: Default log file path
        """
        # Create logs directory if it doesn't exist
        logs_dir = 'logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate timestamp-based log file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(logs_dir, f'academic_performance_prediction_{timestamp}.log')
    
    def _setup_logger(self) -> logging.Logger:
        """
        Setup the academic performance prediction logger.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logger
        logger = logging.getLogger('AcademicPerformancePrediction')
        logger.setLevel(getattr(logging, self.log_level))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler for detailed logging
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Console handler for user-friendly output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.log_level))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # Add academic performance specific handlers
        self._add_academic_handlers(logger)
        
        return logger
    
    def _add_academic_handlers(self, logger: logging.Logger) -> None:
        """
        Add academic performance specific logging handlers.
        
        Args:
            logger (logging.Logger): Logger instance to configure
        """
        # Create academic performance specific log file
        academic_log_file = os.path.join('logs', 'academic_performance_operations.log')
        academic_handler = logging.FileHandler(academic_log_file, mode='a', encoding='utf-8')
        academic_handler.setLevel(logging.INFO)
        
        academic_formatter = logging.Formatter(
            '%(asctime)s - ACADEMIC - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        academic_handler.setFormatter(academic_formatter)
        
        # Add filter to only log academic performance related messages
        academic_handler.addFilter(self._academic_filter)
        logger.addHandler(academic_handler)
    
    def _academic_filter(self, record: logging.LogRecord) -> bool:
        """
        Filter for academic performance related log messages.
        
        Args:
            record (logging.LogRecord): Log record to filter
            
        Returns:
            bool: True if record should be logged
        """
        academic_keywords = [
            'academic', 'performance', 'prediction', 'model', 'training',
            'data', 'feature', 'score', 'student', 'education'
        ]
        
        message = record.getMessage().lower()
        return any(keyword in message for keyword in academic_keywords)
    
    def log_model_training_start(self, model_name: str, dataset_size: int) -> None:
        """
        Log the start of model training for academic performance prediction.
        
        Args:
            model_name (str): Name of the model being trained
            dataset_size (int): Size of the training dataset
        """
        self.logger.info(f"Starting academic performance model training: {model_name}")
        self.logger.info(f"Training dataset size: {dataset_size} records")
        self.logger.info("=" * 60)
    
    def log_model_training_end(self, model_name: str, r2_score: float, 
                             training_time: float) -> None:
        """
        Log the completion of model training.
        
        Args:
            model_name (str): Name of the trained model
            r2_score (float): R² score of the trained model
            training_time (float): Training time in seconds
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Academic performance model training completed: {model_name}")
        self.logger.info(f"Final R² Score: {r2_score:.4f}")
        self.logger.info(f"Training Time: {training_time:.2f} seconds")
    
    def log_prediction_request(self, input_data: dict, prediction: float) -> None:
        """
        Log a prediction request for academic performance.
        
        Args:
            input_data (dict): Input data for prediction
            prediction (float): Generated prediction
        """
        self.logger.info(f"Academic performance prediction requested")
        self.logger.info(f"Input data: {input_data}")
        self.logger.info(f"Predicted math score: {prediction:.2f}")
    
    def log_data_processing(self, operation: str, record_count: int, 
                         processing_time: float) -> None:
        """
        Log data processing operations.
        
        Args:
            operation (str): Type of data processing operation
            record_count (int): Number of records processed
            processing_time (float): Processing time in seconds
        """
        self.logger.info(f"Data processing completed: {operation}")
        self.logger.info(f"Records processed: {record_count}")
        self.logger.info(f"Processing time: {processing_time:.2f} seconds")
    
    def log_error_with_context(self, error: Exception, context: str = "") -> None:
        """
        Log errors with detailed context and stack trace.
        
        Args:
            error (Exception): The error that occurred
            context (str): Additional context about the error
        """
        self.logger.error(f"Academic performance prediction error: {str(error)}")
        if context:
            self.logger.error(f"Context: {context}")
        
        # Log stack trace for debugging
        self.logger.debug(f"Stack trace: {traceback.format_exc()}")
    
    def log_performance_metrics(self, metrics: dict) -> None:
        """
        Log model performance metrics.
        
        Args:
            metrics (dict): Dictionary of performance metrics
        """
        self.logger.info("Academic performance model metrics:")
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    def log_feature_importance(self, feature_importance: dict) -> None:
        """
        Log feature importance for academic performance prediction.
        
        Args:
            feature_importance (dict): Dictionary of feature importance scores
        """
        self.logger.info("Academic performance feature importance:")
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_features[:10]:  # Top 10 features
            self.logger.info(f"  {feature}: {importance:.4f}")
    
    def log_batch_processing(self, batch_size: int, total_records: int, 
                           current_batch: int) -> None:
        """
        Log batch processing progress.
        
        Args:
            batch_size (int): Size of each batch
            total_records (int): Total number of records
            current_batch (int): Current batch number
        """
        progress_percentage = (current_batch * batch_size / total_records) * 100
        self.logger.info(f"Batch processing progress: {progress_percentage:.1f}% "
                        f"({current_batch * batch_size}/{total_records} records)")
    
    def log_system_info(self) -> None:
        """Log system information for debugging purposes."""
        self.logger.info("Academic Performance Prediction System Information:")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Log level: {self.log_level}")
    
    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.
        
        Returns:
            logging.Logger: Logger instance
        """
        return self.logger
    
    def close_logger(self) -> None:
        """Close all logger handlers."""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

# Global logger instance for academic performance prediction
academic_logger = AcademicLogger()
logging = academic_logger.get_logger()

# Convenience functions for common logging operations
def log_academic_info(message: str) -> None:
    """Log academic performance information message."""
    logging.info(f"ACADEMIC: {message}")

def log_academic_warning(message: str) -> None:
    """Log academic performance warning message."""
    logging.warning(f"ACADEMIC WARNING: {message}")

def log_academic_error(message: str, error: Exception = None) -> None:
    """Log academic performance error message."""
    if error:
        logging.error(f"ACADEMIC ERROR: {message} - {str(error)}")
    else:
        logging.error(f"ACADEMIC ERROR: {message}")

def log_academic_debug(message: str) -> None:
    """Log academic performance debug message."""
    logging.debug(f"ACADEMIC DEBUG: {message}")

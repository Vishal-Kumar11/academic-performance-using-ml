# Academic Performance Prediction using Machine Learning - Batch Prediction Pipeline

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import io
import base64

from src.exception import CustomException
from src.logger import logging
from src.pipeline.academic_prediction_pipeline import AcademicPerformancePredictor
from src.utils.validators import AcademicDataValidator

class AcademicBatchPredictor:
    """
    Academic Performance Batch Prediction Handler
    
    This class handles batch prediction processing for academic performance,
    including CSV file uploads, validation, and result generation.
    """
    
    def __init__(self):
        """Initialize the academic batch predictor."""
        self.predictor = AcademicPerformancePredictor()
        self.validator = AcademicDataValidator()
        
    def validate_batch_data(self, dataframe: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate batch input data for academic performance prediction.
        
        Args:
            dataframe (pd.DataFrame): Input data to validate
            
        Returns:
            Tuple[bool, List[str]]: Validation result and error messages
            
        Raises:
            CustomException: If validation fails
        """
        try:
            errors = []
            
            # Check required columns
            required_columns = [
                'gender', 'race_ethnicity', 'parental_level_of_education',
                'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
            ]
            
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check data types and ranges
            for index, row in dataframe.iterrows():
                row_errors = self.validator.validate_single_record(row)
                if row_errors:
                    errors.extend([f"Row {index + 1}: {error}" for error in row_errors])
            
            # Check for empty dataframe
            if dataframe.empty:
                errors.append("Input data is empty")
            
            # Check for reasonable number of rows
            if len(dataframe) > 1000:
                errors.append("Too many rows. Maximum 1000 rows allowed for batch processing")
            
            is_valid = len(errors) == 0
            logging.info(f"Batch data validation completed. Valid: {is_valid}, Errors: {len(errors)}")
            
            return is_valid, errors
            
        except Exception as e:
            logging.error(f"Error validating batch data: {str(e)}")
            raise CustomException(e, sys)
    
    def process_batch_predictions(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Process batch predictions for academic performance.
        
        Args:
            dataframe (pd.DataFrame): Input data for batch prediction
            
        Returns:
            pd.DataFrame: Results with predictions and confidence scores
            
        Raises:
            CustomException: If batch processing fails
        """
        try:
            logging.info(f"Processing batch predictions for {len(dataframe)} records")
            
            # Validate input data
            is_valid, errors = self.validate_batch_data(dataframe)
            if not is_valid:
                raise CustomException(f"Batch data validation failed: {'; '.join(errors)}", sys)
            
            # Process predictions in batches to manage memory
            batch_size = 100
            all_predictions = []
            
            for i in range(0, len(dataframe), batch_size):
                batch_data = dataframe.iloc[i:i + batch_size]
                batch_predictions = self.predictor.predict(batch_data)
                all_predictions.extend(batch_predictions)
                
                logging.info(f"Processed batch {i//batch_size + 1}/{(len(dataframe)-1)//batch_size + 1}")
            
            # Create results dataframe
            results_df = dataframe.copy()
            results_df['predicted_math_score'] = all_predictions
            
            # Add confidence intervals (simplified approach)
            results_df['confidence_lower'] = results_df['predicted_math_score'] - 5.0
            results_df['confidence_upper'] = results_df['predicted_math_score'] + 5.0
            
            # Add prediction timestamp
            results_df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Add prediction quality indicators
            results_df['prediction_quality'] = results_df.apply(
                lambda row: self._assess_prediction_quality(row), axis=1
            )
            
            logging.info("Batch predictions completed successfully")
            return results_df
            
        except Exception as e:
            logging.error(f"Error processing batch predictions: {str(e)}")
            raise CustomException(e, sys)
    
    def _assess_prediction_quality(self, row: pd.Series) -> str:
        """
        Assess the quality of a single prediction based on input features.
        
        Args:
            row (pd.Series): Input data row
            
        Returns:
            str: Quality assessment ('High', 'Medium', 'Low')
        """
        try:
            quality_score = 0
            
            # Higher reading and writing scores indicate better prediction quality
            if row['reading_score'] >= 80 and row['writing_score'] >= 80:
                quality_score += 2
            elif row['reading_score'] >= 60 and row['writing_score'] >= 60:
                quality_score += 1
            
            # Test preparation course completion improves quality
            if row['test_preparation_course'] == 'completed':
                quality_score += 1
            
            # Parental education level affects quality
            if row['parental_level_of_education'] in ['master\'s degree', 'bachelor\'s degree']:
                quality_score += 1
            
            # Determine quality level
            if quality_score >= 3:
                return 'High'
            elif quality_score >= 2:
                return 'Medium'
            else:
                return 'Low'
                
        except Exception as e:
            logging.warning(f"Error assessing prediction quality: {str(e)}")
            return 'Medium'
    
    def generate_batch_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for batch prediction results.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            
        Returns:
            Dict[str, Any]: Summary statistics
            
        Raises:
            CustomException: If summary generation fails
        """
        try:
            summary = {
                'total_predictions': len(results_df),
                'prediction_statistics': {
                    'mean': float(results_df['predicted_math_score'].mean()),
                    'median': float(results_df['predicted_math_score'].median()),
                    'std': float(results_df['predicted_math_score'].std()),
                    'min': float(results_df['predicted_math_score'].min()),
                    'max': float(results_df['predicted_math_score'].max())
                },
                'quality_distribution': results_df['prediction_quality'].value_counts().to_dict(),
                'gender_distribution': results_df['gender'].value_counts().to_dict(),
                'ethnicity_distribution': results_df['race_ethnicity'].value_counts().to_dict(),
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logging.info("Batch prediction summary generated successfully")
            return summary
            
        except Exception as e:
            logging.error(f"Error generating batch summary: {str(e)}")
            raise CustomException(e, sys)
    
    def save_results_to_csv(self, results_df: pd.DataFrame, filename: str = None) -> str:
        """
        Save batch prediction results to CSV file.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            filename (str, optional): Custom filename
            
        Returns:
            str: Path to saved file
            
        Raises:
            CustomException: If file saving fails
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'academic_performance_predictions_{timestamp}.csv'
            
            # Ensure filename has .csv extension
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Create results directory if it doesn't exist
            results_dir = 'artifacts/batch_results'
            os.makedirs(results_dir, exist_ok=True)
            
            file_path = os.path.join(results_dir, filename)
            results_df.to_csv(file_path, index=False)
            
            logging.info(f"Batch prediction results saved to: {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving results to CSV: {str(e)}")
            raise CustomException(e, sys)
    
    def create_download_link(self, results_df: pd.DataFrame) -> str:
        """
        Create a downloadable CSV link for batch results.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            
        Returns:
            str: Base64 encoded CSV data for download
            
        Raises:
            CustomException: If download link creation fails
        """
        try:
            # Convert dataframe to CSV string
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Encode to base64
            csv_bytes = csv_string.encode('utf-8')
            csv_b64 = base64.b64encode(csv_bytes).decode('utf-8')
            
            logging.info("Download link created successfully")
            return csv_b64
            
        except Exception as e:
            logging.error(f"Error creating download link: {str(e)}")
            raise CustomException(e, sys)
    
    def process_uploaded_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process uploaded CSV file for batch prediction.
        
        Args:
            file_content (bytes): File content
            filename (str): Original filename
            
        Returns:
            Dict[str, Any]: Processing results and download link
            
        Raises:
            CustomException: If file processing fails
        """
        try:
            # Read CSV file
            dataframe = pd.read_csv(io.BytesIO(file_content))
            
            # Process batch predictions
            results_df = self.process_batch_predictions(dataframe)
            
            # Generate summary
            summary = self.generate_batch_summary(results_df)
            
            # Create download link
            download_link = self.create_download_link(results_df)
            
            # Save results
            saved_path = self.save_results_to_csv(results_df)
            
            result = {
                'success': True,
                'summary': summary,
                'download_link': download_link,
                'saved_path': saved_path,
                'filename': filename
            }
            
            logging.info(f"File {filename} processed successfully")
            return result
            
        except Exception as e:
            logging.error(f"Error processing uploaded file {filename}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }

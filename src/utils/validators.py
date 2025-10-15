# Academic Performance Prediction using Machine Learning - Input Validation Utilities

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import re

from src.exception import CustomException
from src.logger import logging

class AcademicDataValidator:
    """
    Academic Performance Data Validation Handler
    
    This class provides comprehensive input validation for academic performance
    prediction, including data type checking, range validation, and edge case handling.
    """
    
    def __init__(self):
        """Initialize the academic data validator with validation rules."""
        self.validation_rules = self._initialize_validation_rules()
        
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize validation rules for academic performance data.
        
        Returns:
            Dict[str, Dict[str, Any]]: Validation rules dictionary
        """
        return {
            'gender': {
                'type': str,
                'allowed_values': ['male', 'female'],
                'required': True,
                'case_sensitive': False
            },
            'race_ethnicity': {
                'type': str,
                'allowed_values': ['group A', 'group B', 'group C', 'group D', 'group E'],
                'required': True,
                'case_sensitive': False
            },
            'parental_level_of_education': {
                'type': str,
                'allowed_values': [
                    'associate\'s degree', 'bachelor\'s degree', 'high school',
                    'master\'s degree', 'some college', 'some high school'
                ],
                'required': True,
                'case_sensitive': False
            },
            'lunch': {
                'type': str,
                'allowed_values': ['free/reduced', 'standard'],
                'required': True,
                'case_sensitive': False
            },
            'test_preparation_course': {
                'type': str,
                'allowed_values': ['none', 'completed'],
                'required': True,
                'case_sensitive': False
            },
            'reading_score': {
                'type': Union[int, float],
                'min_value': 0,
                'max_value': 100,
                'required': True
            },
            'writing_score': {
                'type': Union[int, float],
                'min_value': 0,
                'max_value': 100,
                'required': True
            }
        }
    
    def validate_single_record(self, record: pd.Series) -> List[str]:
        """
        Validate a single academic performance record.
        
        Args:
            record (pd.Series): Single record to validate
            
        Returns:
            List[str]: List of validation error messages
            
        Raises:
            CustomException: If validation fails
        """
        try:
            errors = []
            
            for field_name, rules in self.validation_rules.items():
                field_value = record.get(field_name)
                
                # Check if field is required
                if rules.get('required', False) and (field_value is None or pd.isna(field_value)):
                    errors.append(f"{field_name} is required")
                    continue
                
                # Skip validation if field is not required and empty
                if not rules.get('required', False) and (field_value is None or pd.isna(field_value)):
                    continue
                
                # Validate field value
                field_errors = self._validate_field(field_name, field_value, rules)
                errors.extend(field_errors)
            
            return errors
            
        except Exception as e:
            logging.error(f"Error validating single record: {str(e)}")
            raise CustomException(e, sys)
    
    def _validate_field(self, field_name: str, field_value: Any, rules: Dict[str, Any]) -> List[str]:
        """
        Validate a single field according to its rules.
        
        Args:
            field_name (str): Name of the field
            field_value (Any): Value to validate
            rules (Dict[str, Any]): Validation rules for the field
            
        Returns:
            List[str]: List of validation error messages
        """
        errors = []
        
        try:
            # Type validation
            expected_type = rules.get('type')
            if expected_type and not isinstance(field_value, expected_type):
                # Try to convert if possible
                try:
                    if expected_type == Union[int, float]:
                        field_value = float(field_value)
                    else:
                        field_value = expected_type(field_value)
                except (ValueError, TypeError):
                    errors.append(f"{field_name} must be of type {expected_type.__name__}")
                    return errors
            
            # String-specific validations
            if isinstance(field_value, str):
                field_value = field_value.strip()
                
                # Case sensitivity handling
                if not rules.get('case_sensitive', True):
                    field_value = field_value.lower()
                    allowed_values = [val.lower() for val in rules.get('allowed_values', [])]
                else:
                    allowed_values = rules.get('allowed_values', [])
                
                # Allowed values validation
                if allowed_values and field_value not in allowed_values:
                    errors.append(f"{field_name} must be one of: {', '.join(allowed_values)}")
            
            # Numeric validations
            elif isinstance(field_value, (int, float)):
                min_value = rules.get('min_value')
                max_value = rules.get('max_value')
                
                if min_value is not None and field_value < min_value:
                    errors.append(f"{field_name} must be >= {min_value}")
                
                if max_value is not None and field_value > max_value:
                    errors.append(f"{field_name} must be <= {max_value}")
            
            return errors
            
        except Exception as e:
            logging.error(f"Error validating field {field_name}: {str(e)}")
            return [f"Error validating {field_name}: {str(e)}"]
    
    def validate_dataframe(self, dataframe: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate an entire dataframe for academic performance prediction.
        
        Args:
            dataframe (pd.DataFrame): Dataframe to validate
            
        Returns:
            Tuple[bool, List[str]]: Validation result and error messages
            
        Raises:
            CustomException: If validation fails
        """
        try:
            all_errors = []
            
            # Check if dataframe is empty
            if dataframe.empty:
                return False, ["Dataframe is empty"]
            
            # Check required columns
            required_columns = [field for field, rules in self.validation_rules.items() 
                              if rules.get('required', False)]
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            
            if missing_columns:
                all_errors.append(f"Missing required columns: {missing_columns}")
            
            # Validate each record
            for index, row in dataframe.iterrows():
                record_errors = self.validate_single_record(row)
                if record_errors:
                    all_errors.extend([f"Row {index + 1}: {error}" for error in record_errors])
            
            # Check for duplicate rows
            duplicate_count = dataframe.duplicated().sum()
            if duplicate_count > 0:
                all_errors.append(f"Found {duplicate_count} duplicate rows")
            
            # Check for reasonable data distribution
            distribution_errors = self._check_data_distribution(dataframe)
            all_errors.extend(distribution_errors)
            
            is_valid = len(all_errors) == 0
            logging.info(f"Dataframe validation completed. Valid: {is_valid}, Errors: {len(all_errors)}")
            
            return is_valid, all_errors
            
        except Exception as e:
            logging.error(f"Error validating dataframe: {str(e)}")
            raise CustomException(e, sys)
    
    def _check_data_distribution(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Check data distribution for potential issues.
        
        Args:
            dataframe (pd.DataFrame): Dataframe to check
            
        Returns:
            List[str]: List of distribution-related error messages
        """
        errors = []
        
        try:
            # Check score distributions
            score_columns = ['reading_score', 'writing_score']
            for col in score_columns:
                if col in dataframe.columns:
                    scores = dataframe[col].dropna()
                    if len(scores) > 0:
                        # Check for extreme values
                        if scores.min() < 0 or scores.max() > 100:
                            errors.append(f"{col} contains values outside valid range [0, 100]")
                        
                        # Check for suspicious patterns
                        if scores.std() < 1:
                            errors.append(f"{col} has very low variance, possible data quality issue")
                        
                        # Check for too many identical values
                        unique_ratio = scores.nunique() / len(scores)
                        if unique_ratio < 0.1:
                            errors.append(f"{col} has very few unique values, possible data quality issue")
            
            # Check categorical distributions
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            for col in categorical_columns:
                if col in dataframe.columns:
                    value_counts = dataframe[col].value_counts()
                    if len(value_counts) > 0:
                        # Check for extreme imbalance
                        max_ratio = value_counts.max() / value_counts.sum()
                        if max_ratio > 0.9:
                            errors.append(f"{col} is extremely imbalanced, possible data quality issue")
            
            return errors
            
        except Exception as e:
            logging.error(f"Error checking data distribution: {str(e)}")
            return [f"Error checking data distribution: {str(e)}"]
    
    def validate_prediction_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input data for single prediction.
        
        Args:
            input_data (Dict[str, Any]): Input data dictionary
            
        Returns:
            Tuple[bool, List[str]]: Validation result and error messages
            
        Raises:
            CustomException: If validation fails
        """
        try:
            # Convert to pandas Series for validation
            record = pd.Series(input_data)
            errors = self.validate_single_record(record)
            
            is_valid = len(errors) == 0
            logging.info(f"Prediction input validation completed. Valid: {is_valid}, Errors: {len(errors)}")
            
            return is_valid, errors
            
        except Exception as e:
            logging.error(f"Error validating prediction input: {str(e)}")
            raise CustomException(e, sys)
    
    def sanitize_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize and normalize input data for academic performance prediction.
        
        Args:
            input_data (Dict[str, Any]): Raw input data
            
        Returns:
            Dict[str, Any]: Sanitized input data
            
        Raises:
            CustomException: If sanitization fails
        """
        try:
            sanitized_data = {}
            
            for field_name, field_value in input_data.items():
                if field_name in self.validation_rules:
                    rules = self.validation_rules[field_name]
                    
                    # Handle string fields
                    if isinstance(field_value, str):
                        field_value = field_value.strip()
                        if not rules.get('case_sensitive', True):
                            field_value = field_value.lower()
                    
                    # Handle numeric fields
                    elif isinstance(field_value, (int, float)):
                        # Ensure numeric values are within bounds
                        min_value = rules.get('min_value')
                        max_value = rules.get('max_value')
                        
                        if min_value is not None:
                            field_value = max(field_value, min_value)
                        if max_value is not None:
                            field_value = min(field_value, max_value)
                    
                    sanitized_data[field_name] = field_value
                else:
                    sanitized_data[field_name] = field_value
            
            logging.info("Input data sanitization completed successfully")
            return sanitized_data
            
        except Exception as e:
            logging.error(f"Error sanitizing input data: {str(e)}")
            raise CustomException(e, sys)
    
    def get_validation_summary(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive validation summary for academic performance data.
        
        Args:
            dataframe (pd.DataFrame): Dataframe to analyze
            
        Returns:
            Dict[str, Any]: Validation summary dictionary
            
        Raises:
            CustomException: If summary generation fails
        """
        try:
            summary = {
                'total_records': len(dataframe),
                'total_columns': len(dataframe.columns),
                'missing_values': dataframe.isnull().sum().to_dict(),
                'data_types': dataframe.dtypes.to_dict(),
                'duplicate_records': dataframe.duplicated().sum(),
                'validation_status': 'Unknown'
            }
            
            # Perform validation
            is_valid, errors = self.validate_dataframe(dataframe)
            summary['validation_status'] = 'Valid' if is_valid else 'Invalid'
            summary['validation_errors'] = errors
            summary['error_count'] = len(errors)
            
            # Add column-specific summaries
            for col in dataframe.columns:
                if col in self.validation_rules:
                    col_summary = {
                        'unique_values': dataframe[col].nunique(),
                        'most_common_value': dataframe[col].mode().iloc[0] if not dataframe[col].mode().empty else None,
                        'missing_percentage': (dataframe[col].isnull().sum() / len(dataframe)) * 100
                    }
                    summary[f'{col}_summary'] = col_summary
            
            logging.info("Validation summary generated successfully")
            return summary
            
        except Exception as e:
            logging.error(f"Error generating validation summary: {str(e)}")
            raise CustomException(e, sys)

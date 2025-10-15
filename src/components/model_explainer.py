# Academic Performance Prediction using Machine Learning - Model Explainability Module

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class AcademicModelExplainer:
    """
    Academic Performance Model Explainability Handler
    
    This class provides comprehensive model explainability using SHAP values,
    feature importance analysis, and prediction explanations for academic performance prediction.
    """
    
    def __init__(self):
        """Initialize the academic model explainer."""
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.shap_explainer = None
        
    def load_model_and_preprocessor(self) -> None:
        """
        Load the trained academic performance model and preprocessor.
        
        Raises:
            CustomException: If model or preprocessor loading fails
        """
        try:
            model_path = 'artifacts/academic_performance_model.pkl'
            preprocessor_path = 'artifacts/academic_preprocessor.pkl'
            
            self.model = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)
            
            # Define feature names for academic performance prediction
            self.feature_names = [
                'writing_score', 'reading_score',
                'gender_female', 'gender_male',
                'race_ethnicity_group_A', 'race_ethnicity_group_B', 
                'race_ethnicity_group_C', 'race_ethnicity_group_D', 'race_ethnicity_group_E',
                'parental_level_of_education_associate_degree', 'parental_level_of_education_bachelor_degree',
                'parental_level_of_education_high_school', 'parental_level_of_education_master_degree',
                'parental_level_of_education_some_college', 'parental_level_of_education_some_high_school',
                'lunch_free_reduced', 'lunch_standard',
                'test_preparation_course_completed', 'test_preparation_course_none'
            ]
            
            logging.info("Academic performance model and preprocessor loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading academic model and preprocessor: {str(e)}")
            raise CustomException(e, sys)
    
    def get_feature_importance(self, X_train: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance for academic performance prediction.
        
        Args:
            X_train (np.ndarray): Training features
            
        Returns:
            Dict[str, float]: Feature importance scores
            
        Raises:
            CustomException: If feature importance calculation fails
        """
        try:
            if self.model is None:
                self.load_model_and_preprocessor()
            
            # Get feature importance based on model type
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance_scores = np.abs(self.model.coef_)
            else:
                # For models without built-in feature importance
                importance_scores = np.ones(len(self.feature_names)) / len(self.feature_names)
            
            # Create feature importance dictionary
            feature_importance_dict = dict(zip(self.feature_names, importance_scores))
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            logging.info("Academic performance feature importance calculated successfully")
            return sorted_importance
            
        except Exception as e:
            logging.error(f"Error calculating feature importance: {str(e)}")
            raise CustomException(e, sys)
    
    def create_feature_importance_plot(self, X_train: np.ndarray, save_path: str = None) -> go.Figure:
        """
        Create interactive feature importance visualization for academic performance.
        
        Args:
            X_train (np.ndarray): Training features
            save_path (str, optional): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
            
        Raises:
            CustomException: If plot creation fails
        """
        try:
            feature_importance = self.get_feature_importance(X_train)
            
            # Prepare data for plotting
            features = list(feature_importance.keys())[:10]  # Top 10 features
            importance_values = list(feature_importance.values())[:10]
            
            # Create interactive bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=importance_values,
                    y=features,
                    orientation='h',
                    marker_color='rgba(55, 128, 191, 0.8)',
                    text=[f'{val:.4f}' for val in importance_values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Academic Performance Prediction - Top 10 Feature Importance',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=600,
                showlegend=False,
                font=dict(size=12)
            )
            
            if save_path:
                fig.write_html(save_path)
                logging.info(f"Feature importance plot saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating feature importance plot: {str(e)}")
            raise CustomException(e, sys)
    
    def explain_prediction(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            input_data (pd.DataFrame): Input data for prediction
            
        Returns:
            Dict[str, Any]: Prediction explanation with SHAP values
            
        Raises:
            CustomException: If prediction explanation fails
        """
        try:
            if not SHAP_AVAILABLE:
                raise CustomException("SHAP is not available. Please install shap package.", sys)
            
            if self.model is None:
                self.load_model_and_preprocessor()
            
            # Transform input data
            transformed_data = self.preprocessor.transform(input_data)
            
            # Create SHAP explainer
            if self.shap_explainer is None:
                # Use TreeExplainer for tree-based models, otherwise use KernelExplainer
                if hasattr(self.model, 'predict_proba') or 'tree' in str(type(self.model)).lower():
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    self.shap_explainer = shap.KernelExplainer(self.model.predict, transformed_data[:100])
            
            # Calculate SHAP values
            if hasattr(self.shap_explainer, 'shap_values'):
                shap_values = self.shap_explainer.shap_values(transformed_data)
            else:
                shap_values = self.shap_explainer(transformed_data)
            
            # Get prediction
            prediction = self.model.predict(transformed_data)[0]
            
            # Create explanation dictionary
            explanation = {
                'prediction': float(prediction),
                'shap_values': shap_values[0].tolist() if len(shap_values.shape) > 1 else shap_values.tolist(),
                'feature_names': self.feature_names,
                'base_value': float(self.shap_explainer.expected_value) if hasattr(self.shap_explainer, 'expected_value') else 0.0
            }
            
            logging.info("Academic performance prediction explanation generated successfully")
            return explanation
            
        except Exception as e:
            logging.error(f"Error explaining prediction: {str(e)}")
            raise CustomException(e, sys)
    
    def create_shap_summary_plot(self, X_train: np.ndarray, save_path: str = None) -> None:
        """
        Create SHAP summary plot for academic performance model.
        
        Args:
            X_train (np.ndarray): Training features
            save_path (str, optional): Path to save the plot
            
        Raises:
            CustomException: If SHAP plot creation fails
        """
        try:
            if not SHAP_AVAILABLE:
                logging.warning("SHAP not available. Skipping SHAP summary plot.")
                return
            
            if self.model is None:
                self.load_model_and_preprocessor()
            
            # Transform training data
            transformed_data = self.preprocessor.transform(X_train)
            
            # Create SHAP explainer
            if hasattr(self.model, 'predict_proba') or 'tree' in str(type(self.model)).lower():
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(transformed_data)
            else:
                explainer = shap.KernelExplainer(self.model.predict, transformed_data[:100])
                shap_values = explainer(transformed_data)
            
            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, transformed_data, feature_names=self.feature_names, show=False)
            plt.title('Academic Performance Prediction - SHAP Summary Plot')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logging.info(f"SHAP summary plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logging.error(f"Error creating SHAP summary plot: {str(e)}")
            raise CustomException(e, sys)
    
    def generate_model_report(self, X_train: np.ndarray, X_test: np.ndarray, 
                            y_test: np.ndarray) -> Dict[str, Any]:
        """
        Generate comprehensive model explainability report.
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Testing features
            y_test (np.ndarray): Testing targets
            
        Returns:
            Dict[str, Any]: Complete model explainability report
            
        Raises:
            CustomException: If report generation fails
        """
        try:
            if self.model is None:
                self.load_model_and_preprocessor()
            
            # Get predictions
            predictions = self.model.predict(X_test)
            
            # Calculate feature importance
            feature_importance = self.get_feature_importance(X_train)
            
            # Generate report
            report = {
                'model_type': str(type(self.model).__name__),
                'feature_importance': feature_importance,
                'top_features': list(feature_importance.keys())[:5],
                'prediction_range': {
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions)),
                    'mean': float(np.mean(predictions))
                },
                'shap_available': SHAP_AVAILABLE,
                'total_features': len(self.feature_names)
            }
            
            logging.info("Academic performance model explainability report generated successfully")
            return report
            
        except Exception as e:
            logging.error(f"Error generating model report: {str(e)}")
            raise CustomException(e, sys)

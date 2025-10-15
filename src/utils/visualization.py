# Academic Performance Prediction using Machine Learning - Visualization Utilities

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from src.exception import CustomException
from src.logger import logging

class AcademicVisualizationUtils:
    """
    Academic Performance Visualization Utilities
    
    This class provides comprehensive visualization tools for academic performance
    prediction, including model comparison charts, performance dashboards, and
    interactive visualizations.
    """
    
    def __init__(self):
        """Initialize the academic visualization utilities."""
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_model_comparison_chart(self, model_results: Dict[str, float], 
                                    save_path: str = None) -> go.Figure:
        """
        Create interactive model comparison chart for academic performance prediction.
        
        Args:
            model_results (Dict[str, float]): Model performance results
            save_path (str, optional): Path to save the chart
            
        Returns:
            go.Figure: Plotly figure object
            
        Raises:
            CustomException: If chart creation fails
        """
        try:
            # Prepare data
            models = list(model_results.keys())
            scores = list(model_results.values())
            
            # Create horizontal bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=scores,
                    y=models,
                    orientation='h',
                    marker=dict(
                        color=scores,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="R² Score")
                    ),
                    text=[f'{score:.4f}' for score in scores],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Academic Performance Prediction - Model Comparison',
                xaxis_title='R² Score',
                yaxis_title='Models',
                height=500,
                showlegend=False,
                font=dict(size=12),
                margin=dict(l=200, r=50, t=80, b=50)
            )
            
            if save_path:
                fig.write_html(save_path)
                logging.info(f"Model comparison chart saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating model comparison chart: {str(e)}")
            raise CustomException(e, sys)
    
    def create_performance_dashboard(self, train_scores: Dict[str, float], 
                                  test_scores: Dict[str, float],
                                  save_path: str = None) -> go.Figure:
        """
        Create comprehensive performance dashboard for academic models.
        
        Args:
            train_scores (Dict[str, float]): Training scores
            test_scores (Dict[str, float]): Testing scores
            save_path (str, optional): Path to save the dashboard
            
        Returns:
            go.Figure: Plotly figure object
            
        Raises:
            CustomException: If dashboard creation fails
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training vs Testing Scores', 'Model Performance Comparison',
                              'Overfitting Analysis', 'Top 5 Models'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            models = list(train_scores.keys())
            
            # Training vs Testing Scores
            fig.add_trace(
                go.Scatter(x=train_scores.values(), y=test_scores.values(),
                          mode='markers+text', text=models, textposition="top center",
                          marker=dict(size=10, color='blue'), name='Models'),
                row=1, col=1
            )
            
            # Add diagonal line for perfect correlation
            min_score = min(min(train_scores.values()), min(test_scores.values()))
            max_score = max(max(train_scores.values()), max(test_scores.values()))
            fig.add_trace(
                go.Scatter(x=[min_score, max_score], y=[min_score, max_score],
                          mode='lines', line=dict(dash='dash', color='red'),
                          name='Perfect Correlation'),
                row=1, col=1
            )
            
            # Model Performance Comparison
            fig.add_trace(
                go.Bar(x=models, y=list(train_scores.values()), name='Training Score'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=models, y=list(test_scores.values()), name='Testing Score'),
                row=1, col=2
            )
            
            # Overfitting Analysis
            overfitting_scores = [train_scores[model] - test_scores[model] for model in models]
            fig.add_trace(
                go.Bar(x=models, y=overfitting_scores, name='Overfitting Gap'),
                row=2, col=1
            )
            
            # Top 5 Models
            sorted_models = sorted(test_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            top_models = [item[0] for item in sorted_models]
            top_scores = [item[1] for item in sorted_models]
            
            fig.add_trace(
                go.Bar(x=top_models, y=top_scores, name='Top 5 Models'),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Academic Performance Prediction - Performance Dashboard',
                height=800,
                showlegend=True,
                font=dict(size=10)
            )
            
            if save_path:
                fig.write_html(save_path)
                logging.info(f"Performance dashboard saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating performance dashboard: {str(e)}")
            raise CustomException(e, sys)
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, float],
                                     save_path: str = None) -> go.Figure:
        """
        Create interactive feature importance plot for academic performance.
        
        Args:
            feature_importance (Dict[str, float]): Feature importance scores
            save_path (str, optional): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
            
        Raises:
            CustomException: If plot creation fails
        """
        try:
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features = [item[0] for item in sorted_features[:15]]  # Top 15 features
            importance_values = [item[1] for item in sorted_features[:15]]
            
            # Create horizontal bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=importance_values,
                    y=features,
                    orientation='h',
                    marker=dict(
                        color=importance_values,
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Importance Score")
                    ),
                    text=[f'{val:.4f}' for val in importance_values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Academic Performance Prediction - Top 15 Feature Importance',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=600,
                showlegend=False,
                font=dict(size=11),
                margin=dict(l=250, r=50, t=80, b=50)
            )
            
            if save_path:
                fig.write_html(save_path)
                logging.info(f"Feature importance plot saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating feature importance plot: {str(e)}")
            raise CustomException(e, sys)
    
    def create_prediction_distribution_plot(self, predictions: List[float],
                                          actual_values: List[float] = None,
                                          save_path: str = None) -> go.Figure:
        """
        Create prediction distribution plot for academic performance.
        
        Args:
            predictions (List[float]): Predicted values
            actual_values (List[float], optional): Actual values for comparison
            save_path (str, optional): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
            
        Raises:
            CustomException: If plot creation fails
        """
        try:
            fig = go.Figure()
            
            # Add prediction histogram
            fig.add_trace(go.Histogram(
                x=predictions,
                name='Predictions',
                opacity=0.7,
                nbinsx=30
            ))
            
            # Add actual values if provided
            if actual_values is not None:
                fig.add_trace(go.Histogram(
                    x=actual_values,
                    name='Actual Values',
                    opacity=0.7,
                    nbinsx=30
                ))
            
            fig.update_layout(
                title='Academic Performance Prediction - Score Distribution',
                xaxis_title='Math Score',
                yaxis_title='Frequency',
                barmode='overlay',
                height=500,
                showlegend=True
            )
            
            if save_path:
                fig.write_html(save_path)
                logging.info(f"Prediction distribution plot saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating prediction distribution plot: {str(e)}")
            raise CustomException(e, sys)
    
    def create_correlation_heatmap(self, dataframe: pd.DataFrame,
                                 save_path: str = None) -> go.Figure:
        """
        Create correlation heatmap for academic performance features.
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
            save_path (str, optional): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
            
        Raises:
            CustomException: If heatmap creation fails
        """
        try:
            # Calculate correlation matrix
            correlation_matrix = dataframe.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Academic Performance Features - Correlation Heatmap',
                height=600,
                font=dict(size=10)
            )
            
            if save_path:
                fig.write_html(save_path)
                logging.info(f"Correlation heatmap saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating correlation heatmap: {str(e)}")
            raise CustomException(e, sys)
    
    def create_interactive_scatter_plot(self, dataframe: pd.DataFrame,
                                      x_col: str, y_col: str, color_col: str = None,
                                      save_path: str = None) -> go.Figure:
        """
        Create interactive scatter plot for academic performance analysis.
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
            x_col (str): X-axis column name
            y_col (str): Y-axis column name
            color_col (str, optional): Color grouping column
            save_path (str, optional): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
            
        Raises:
            CustomException: If scatter plot creation fails
        """
        try:
            if color_col:
                fig = px.scatter(dataframe, x=x_col, y=y_col, color=color_col,
                              title=f'Academic Performance - {x_col} vs {y_col}',
                              hover_data=dataframe.columns)
            else:
                fig = px.scatter(dataframe, x=x_col, y=y_col,
                              title=f'Academic Performance - {x_col} vs {y_col}',
                              hover_data=dataframe.columns)
            
            fig.update_layout(
                height=500,
                font=dict(size=12)
            )
            
            if save_path:
                fig.write_html(save_path)
                logging.info(f"Interactive scatter plot saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating interactive scatter plot: {str(e)}")
            raise CustomException(e, sys)
    
    def create_model_performance_metrics_chart(self, metrics: Dict[str, Dict[str, float]],
                                             save_path: str = None) -> go.Figure:
        """
        Create comprehensive model performance metrics chart.
        
        Args:
            metrics (Dict[str, Dict[str, float]]): Model metrics dictionary
            save_path (str, optional): Path to save the chart
            
        Returns:
            go.Figure: Plotly figure object
            
        Raises:
            CustomException: If metrics chart creation fails
        """
        try:
            models = list(metrics.keys())
            metric_names = list(metrics[models[0]].keys())
            
            fig = make_subplots(
                rows=1, cols=len(metric_names),
                subplot_titles=metric_names,
                specs=[[{"secondary_y": False}] * len(metric_names)]
            )
            
            for i, metric in enumerate(metric_names):
                values = [metrics[model][metric] for model in models]
                
                fig.add_trace(
                    go.Bar(x=models, y=values, name=metric, showlegend=False),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                title='Academic Performance Models - Performance Metrics Comparison',
                height=400,
                font=dict(size=10)
            )
            
            if save_path:
                fig.write_html(save_path)
                logging.info(f"Model performance metrics chart saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating model performance metrics chart: {str(e)}")
            raise CustomException(e, sys)
    
    def save_all_visualizations(self, output_dir: str = 'artifacts/visualizations') -> None:
        """
        Save all visualization templates to the specified directory.
        
        Args:
            output_dir (str): Output directory for visualizations
            
        Raises:
            CustomException: If saving fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create sample data for demonstration
            sample_model_results = {
                'Linear Regression': 0.85,
                'Random Forest': 0.92,
                'XGBoost': 0.94,
                'CatBoost': 0.93,
                'Neural Network': 0.89
            }
            
            # Create and save sample visualizations
            self.create_model_comparison_chart(
                sample_model_results, 
                os.path.join(output_dir, 'model_comparison.html')
            )
            
            logging.info(f"All visualization templates saved to: {output_dir}")
            
        except Exception as e:
            logging.error(f"Error saving visualization templates: {str(e)}")
            raise CustomException(e, sys)

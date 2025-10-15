# Academic Performance Prediction using Machine Learning - Advanced Model Training Module

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Traditional ML Models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, 
    explained_variance_score, mean_absolute_percentage_error
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Advanced ML Models
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.utils.visualization import AcademicVisualizationUtils

@dataclass
class AcademicModelTrainerConfig:
    """
    Configuration class for Advanced Academic Performance Model Training.
    
    Defines file paths and parameters for model training artifacts and visualizations.
    """
    trained_model_file_path: str = os.path.join("artifacts", "academic_performance_model.pkl")
    model_comparison_results_path: str = os.path.join("artifacts", "model_comparison_results.csv")
    model_comparison_plot_path: str = os.path.join("artifacts", "model_comparison.png")
    neural_network_model_path: str = os.path.join("artifacts", "academic_neural_network.h5")

class AcademicModelTrainer:
    """
    Advanced Academic Performance Model Trainer
    
    This class handles training multiple advanced machine learning models for academic
    performance prediction, including traditional ML, gradient boosting, and deep learning
    models with comprehensive hyperparameter tuning and evaluation.
    """
    
    def __init__(self):
        """Initialize the advanced academic model trainer with configuration."""
        self.model_trainer_config = AcademicModelTrainerConfig()
        self.visualization_utils = AcademicVisualizationUtils()
        self.model_results = {}
        self.cv_scores = {}
        
    def create_neural_network_model(self, input_dim: int) -> Sequential:
        """
        Create a neural network model for academic performance prediction.
        
        Args:
            input_dim (int): Number of input features
            
        Returns:
            Sequential: Compiled neural network model
        """
        try:
            model = Sequential([
                Dense(64, activation='relu', input_dim=input_dim),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(16, activation='relu'),
                Dropout(0.1),
                
                Dense(1, activation='linear')
            ])
            
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            logging.info("Neural network model created successfully")
            return model
            
        except Exception as e:
            logging.error(f"Error creating neural network model: {str(e)}")
            raise CustomException(e, sys)
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Sequential, Dict[str, float]]:
        """
        Train neural network model for academic performance prediction.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_test (np.ndarray): Testing features
            y_test (np.ndarray): Testing targets
            
        Returns:
            Tuple[Sequential, Dict[str, float]]: Trained model and performance metrics
        """
        try:
            logging.info("Starting neural network training for academic performance")
            
            # Scale features for neural network
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create model
            model = self.create_neural_network_model(X_train_scaled.shape[1])
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=0
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=0
            )
            
            # Train the model
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=200,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Make predictions
            train_pred = model.predict(X_train_scaled, verbose=0).flatten()
            test_pred = model.predict(X_test_scaled, verbose=0).flatten()
            
            # Calculate metrics
            metrics = {
                'r2_score': r2_score(y_test, test_pred),
                'mae': mean_absolute_error(y_test, test_pred),
                'mse': mean_squared_error(y_test, test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'mape': mean_absolute_percentage_error(y_test, test_pred),
                'explained_variance': explained_variance_score(y_test, test_pred)
            }
            
            logging.info(f"Neural network training completed. R² Score: {metrics['r2_score']:.4f}")
            return model, metrics
            
        except Exception as e:
            logging.error(f"Error training neural network: {str(e)}")
            raise CustomException(e, sys)
    
    def evaluate_model_with_cv(self, model: Any, X: np.ndarray, y: np.ndarray, 
                             model_name: str, cv_folds: int = 5) -> Dict[str, float]:
        """
        Evaluate model using cross-validation for academic performance prediction.
        
        Args:
            model: Model to evaluate
            X (np.ndarray): Features
            y (np.ndarray): Targets
            model_name (str): Name of the model
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Cross-validation scores
        """
        try:
            logging.info(f"Performing {cv_folds}-fold cross-validation for {model_name}")
            
            # Define scoring metrics
            scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            cv_results = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
                cv_results[metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist()
                }
            
            # Store CV results
            self.cv_scores[model_name] = cv_results
            
            logging.info(f"Cross-validation completed for {model_name}")
            return cv_results
            
        except Exception as e:
            logging.error(f"Error in cross-validation for {model_name}: {str(e)}")
            return {}
    
    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        """
        Initiate the complete advanced model training process for academic performance prediction.
        
        Args:
            train_array (np.ndarray): Training data array
            test_array (np.ndarray): Testing data array
            
        Returns:
            float: R² score of the best performing model
            
        Raises:
            CustomException: If model training fails
        """
        try:
            logging.info("Starting advanced academic performance model training process")
            start_time = time.time()
            
            # Split data
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All features except target
                train_array[:, -1],   # Target variable (math score)
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Comprehensive model dictionary for academic performance prediction
            academic_models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.1),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBoost Regressor": XGBRegressor(), 
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "LightGBM Regressor": lgb.LGBMRegressor(verbose=-1),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Support Vector Regressor": SVR()
            }

            # Advanced hyperparameter tuning configuration
            academic_model_params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'max_depth': [5, 8, 12, 15, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 8]
                },
                "Random Forest Regressor": {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [8, 12, 16, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'n_estimators': [100, 200, 300, 400],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10]
                },
                "Linear Regression": {},
                "Ridge Regression": {
                    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
                },
                "Lasso Regression": {
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                "XGBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'n_estimators': [100, 200, 300, 400],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                "CatBoost Regressor": {
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15],
                    'iterations': [200, 300, 500, 700],
                    'l2_leaf_reg': [1, 3, 5, 7, 9]
                },
                "LightGBM Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [15, 31, 63],
                    'subsample': [0.8, 0.9, 1.0]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.5, 0.8, 1.0, 1.2],
                    'n_estimators': [50, 100, 200, 300]
                },
                "Support Vector Regressor": {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            }

            logging.info("Evaluating academic performance models with advanced hyperparameter tuning")
            
            # Train and evaluate traditional models
            for model_name, model in academic_models.items():
                logging.info(f"Training {model_name} for academic performance prediction")
                model_start_time = time.time()
                
                # Perform hyperparameter tuning
                if model_name in academic_model_params and academic_model_params[model_name]:
                    grid_search = GridSearchCV(
                        model, 
                        academic_model_params[model_name], 
                        cv=5, 
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = best_model.predict(X_train)
                test_pred = best_model.predict(X_test)
                
                # Calculate comprehensive metrics
                metrics = {
                    'r2_score': r2_score(y_test, test_pred),
                    'mae': mean_absolute_error(y_test, test_pred),
                    'mse': mean_squared_error(y_test, test_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                    'mape': mean_absolute_percentage_error(y_test, test_pred),
                    'explained_variance': explained_variance_score(y_test, test_pred),
                    'training_time': time.time() - model_start_time
                }
                
                # Perform cross-validation
                cv_results = self.evaluate_model_with_cv(best_model, X_train, y_train, model_name)
                metrics['cv_r2_mean'] = cv_results.get('r2', {}).get('mean', 0)
                metrics['cv_r2_std'] = cv_results.get('r2', {}).get('std', 0)
                
                self.model_results[model_name] = {
                    'model': best_model,
                    'metrics': metrics,
                    'predictions': test_pred
                }
                
                logging.info(f"{model_name} completed - R²: {metrics['r2_score']:.4f}, "
                           f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            # Train Neural Network
            logging.info("Training Neural Network for academic performance prediction")
            nn_model, nn_metrics = self.train_neural_network(X_train, y_train, X_test, y_test)
            
            # Add neural network to results
            self.model_results["Neural Network"] = {
                'model': nn_model,
                'metrics': nn_metrics,
                'predictions': nn_model.predict(StandardScaler().fit_transform(X_test), verbose=0).flatten()
            }
            
            # Determine the best performing model
            best_model_name = max(self.model_results.keys(), 
                                key=lambda x: self.model_results[x]['metrics']['r2_score'])
            best_model_score = self.model_results[best_model_name]['metrics']['r2_score']
            best_model = self.model_results[best_model_name]['model']
            
            logging.info(f"Best academic performance model identified: {best_model_name}")
            logging.info(f"Best model R² score: {best_model_score:.4f}")
            
            # Save the best performing model
            if best_model_name == "Neural Network":
                best_model.save(self.model_trainer_config.neural_network_model_path)
                logging.info(f"Neural network model saved to: {self.model_trainer_config.neural_network_model_path}")
            else:
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
                logging.info(f"Best model saved to: {self.model_trainer_config.trained_model_file_path}")
            
            # Create comprehensive model comparison
            self._create_model_comparison_report()
            
            # Generate model comparison visualization
            self._create_model_comparison_visualization()
            
            total_time = time.time() - start_time
            logging.info(f"Advanced academic model training completed in {total_time:.2f} seconds")
            
            return best_model_score
            
        except Exception as e:
            logging.error(f"Error in advanced academic model training: {str(e)}")
            raise CustomException(e, sys)
    
    def _create_model_comparison_report(self) -> None:
        """
        Create comprehensive model comparison report for academic performance prediction.
        
        Raises:
            CustomException: If report creation fails
        """
        try:
            logging.info("Creating comprehensive model comparison report")
            
            # Prepare comparison data
            comparison_data = []
            for model_name, results in self.model_results.items():
                metrics = results['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'R² Score': metrics['r2_score'],
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'MAPE': metrics['mape'],
                    'Explained Variance': metrics['explained_variance'],
                    'CV R² Mean': metrics.get('cv_r2_mean', 0),
                    'CV R² Std': metrics.get('cv_r2_std', 0),
                    'Training Time (s)': metrics.get('training_time', 0)
                })
            
            # Create DataFrame and save
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('R² Score', ascending=False)
            comparison_df.to_csv(self.model_trainer_config.model_comparison_results_path, index=False)
            
            logging.info(f"Model comparison report saved to: {self.model_trainer_config.model_comparison_results_path}")
            
            # Log top 5 models
            logging.info("Top 5 Academic Performance Models:")
            for i, row in comparison_df.head().iterrows():
                logging.info(f"{i+1}. {row['Model']}: R²={row['R² Score']:.4f}, "
                           f"MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}")
            
        except Exception as e:
            logging.error(f"Error creating model comparison report: {str(e)}")
            raise CustomException(e, sys)
    
    def _create_model_comparison_visualization(self) -> None:
        """
        Create model comparison visualization for academic performance prediction.
        
        Raises:
            CustomException: If visualization creation fails
        """
        try:
            logging.info("Creating model comparison visualization")
            
            # Prepare data for visualization
            model_names = list(self.model_results.keys())
            r2_scores = [self.model_results[name]['metrics']['r2_score'] for name in model_names]
            
            # Create visualization using our visualization utils
            fig = self.visualization_utils.create_model_comparison_chart(
                dict(zip(model_names, r2_scores)),
                save_path=self.model_trainer_config.model_comparison_plot_path.replace('.png', '.html')
            )
            
            logging.info(f"Model comparison visualization saved to: {self.model_trainer_config.model_comparison_plot_path}")
            
        except Exception as e:
            logging.error(f"Error creating model comparison visualization: {str(e)}")
            raise CustomException(e, sys)
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model performance summary for academic performance prediction.
        
        Returns:
            Dict[str, Any]: Complete performance summary
        """
        try:
            summary = {
                'total_models_trained': len(self.model_results),
                'best_model': max(self.model_results.keys(), 
                                key=lambda x: self.model_results[x]['metrics']['r2_score']),
                'model_rankings': {},
                'performance_metrics': {},
                'cross_validation_results': self.cv_scores
            }
            
            # Create rankings
            sorted_models = sorted(self.model_results.items(), 
                                 key=lambda x: x[1]['metrics']['r2_score'], reverse=True)
            
            for i, (model_name, results) in enumerate(sorted_models):
                summary['model_rankings'][i+1] = {
                    'model': model_name,
                    'r2_score': results['metrics']['r2_score'],
                    'mae': results['metrics']['mae'],
                    'rmse': results['metrics']['rmse']
                }
            
            # Aggregate performance metrics
            all_r2_scores = [results['metrics']['r2_score'] for results in self.model_results.values()]
            summary['performance_metrics'] = {
                'best_r2_score': max(all_r2_scores),
                'worst_r2_score': min(all_r2_scores),
                'average_r2_score': np.mean(all_r2_scores),
                'r2_score_std': np.std(all_r2_scores)
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error getting model performance summary: {str(e)}")
            raise CustomException(e, sys)
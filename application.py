from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime

from sklearn.preprocessing import StandardScaler

from src.pipeline.academic_prediction_pipeline import AcademicDataInput, AcademicPerformancePredictor
from src.pipeline.batch_pipeline import AcademicBatchPredictor
from src.utils.validators import AcademicDataValidator
from src.utils.logger import logging as academic_logger

# Academic Performance Prediction using Machine Learning - Main Application
application = Flask(__name__)
application.secret_key = 'your-secret-key-here'  # Change this in production

app = application

# Initialize components
batch_predictor = AcademicBatchPredictor()
data_validator = AcademicDataValidator()

# Route for home page
@app.route('/')
def home_page():
    """Render the home page with project overview and features."""
    try:
        academic_logger.info("Home page accessed")
        return render_template('home.html')
    except Exception as e:
        academic_logger.error(f"Error rendering home page: {str(e)}")
        flash('An error occurred while loading the home page.', 'error')
        return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_academic_performance():
    """Handle single student prediction requests."""
    try:
        if request.method == 'GET':
            academic_logger.info("Prediction form accessed")
            return render_template('predict.html')
        
        else:
            academic_logger.info("Prediction request received")
            
            # Validate input data
            input_data = {
                'gender': request.form.get('gender'),
                'race_ethnicity': request.form.get('ethnicity'),
                'parental_level_of_education': request.form.get('parental_level_of_education'),
                'lunch': request.form.get('lunch'),
                'test_preparation_course': request.form.get('test_preparation_course'),
                'reading_score': request.form.get('reading_score'),
                'writing_score': request.form.get('writing_score'),
            }
            
            # Validate input
            is_valid, errors = data_validator.validate_prediction_input(input_data)
            if not is_valid:
                academic_logger.warning(f"Invalid input data: {errors}")
                flash(f'Please correct the following errors: {"; ".join(errors)}', 'error')
                return render_template('predict.html')
            
            # Create academic data input
            academic_data = AcademicDataInput(
                gender=input_data['gender'],
                race_ethnicity=input_data['race_ethnicity'],
                parental_level_of_education=input_data['parental_level_of_education'],
                lunch=input_data['lunch'],
                test_preparation_course=input_data['test_preparation_course'],
                reading_score=input_data['reading_score'],
                writing_score=input_data['writing_score'],
            )
            
            # Get prediction
            prediction_dataframe = academic_data.get_data_as_data_frame()
            academic_predictor = AcademicPerformancePredictor()
            prediction_results = academic_predictor.predict(prediction_dataframe)
            
            academic_logger.info(f"Prediction completed: {prediction_results[0]}")
            return render_template('results.html', results=prediction_results[0])
            
    except Exception as e:
        academic_logger.error(f"Error in prediction: {str(e)}")
        flash('An error occurred while making the prediction. Please try again.', 'error')
        return render_template('predict.html')

@app.route('/batch', methods=['GET', 'POST'])
def batch_predictions():
    """Handle batch prediction requests."""
    try:
        if request.method == 'GET':
            academic_logger.info("Batch prediction page accessed")
            return render_template('batch.html')
        
        else:
            academic_logger.info("Batch prediction request received")
            
            # Check if file was uploaded
            if 'csvFile' not in request.files:
                flash('No file uploaded. Please select a CSV file.', 'error')
                return render_template('batch.html')
            
            file = request.files['csvFile']
            if file.filename == '':
                flash('No file selected. Please choose a CSV file.', 'error')
                return render_template('batch.html')
            
            if not file.filename.endswith('.csv'):
                flash('Please upload a CSV file only.', 'error')
                return render_template('batch.html')
            
            # Process the uploaded file
            file_content = file.read()
            result = batch_predictor.process_uploaded_file(file_content, file.filename)
            
            if result['success']:
                academic_logger.info(f"Batch processing completed: {result['summary']['total_predictions']} records")
                flash(f"Successfully processed {result['summary']['total_predictions']} records!", 'success')
                return render_template('batch.html', batch_result=result)
            else:
                academic_logger.error(f"Batch processing failed: {result['error']}")
                flash(f"Error processing file: {result['error']}", 'error')
                return render_template('batch.html')
                
    except Exception as e:
        academic_logger.error(f"Error in batch prediction: {str(e)}")
        flash('An error occurred while processing the batch file. Please try again.', 'error')
        return render_template('batch.html')

@app.route('/dashboard')
def model_dashboard():
    """Display the model performance dashboard."""
    try:
        academic_logger.info("Model dashboard accessed")
        return render_template('dashboard.html')
    except Exception as e:
        academic_logger.error(f"Error rendering dashboard: {str(e)}")
        flash('An error occurred while loading the dashboard.', 'error')
        return render_template('dashboard.html')

@app.route('/about')
def about_page():
    """Display the about page with project information."""
    try:
        academic_logger.info("About page accessed")
        return render_template('about.html')
    except Exception as e:
        academic_logger.error(f"Error rendering about page: {str(e)}")
        flash('An error occurred while loading the about page.', 'error')
        return render_template('about.html')

@app.route('/results')
def results_page():
    """Display prediction results (redirect from old route)."""
    try:
        academic_logger.info("Results page accessed")
        return redirect(url_for('home_page'))
    except Exception as e:
        academic_logger.error(f"Error in results page: {str(e)}")
        return redirect(url_for('home_page'))

# API Routes for AJAX requests
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction requests."""
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, errors = data_validator.validate_prediction_input(data)
        if not is_valid:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Make prediction
        academic_data = AcademicDataInput(**data)
        prediction_dataframe = academic_data.get_data_as_data_frame()
        academic_predictor = AcademicPerformancePredictor()
        prediction_results = academic_predictor.predict(prediction_dataframe)
        
        return jsonify({
            'success': True,
            'prediction': prediction_results[0],
            'confidence': 'high'  # This could be calculated based on model certainty
        })
        
    except Exception as e:
        academic_logger.error(f"API prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model-info')
def api_model_info():
    """API endpoint for model information."""
    try:
        model_info = {
            'total_models': 13,
            'best_model': 'XGBoost Regressor',
            'best_score': 0.942,
            'metrics': ['R² Score', 'MAE', 'RMSE', 'MAPE', 'Explained Variance'],
            'cv_folds': 5,
            'features': [
                'gender', 'race_ethnicity', 'parental_level_of_education',
                'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
            ]
        }
        return jsonify(model_info)
    except Exception as e:
        academic_logger.error(f"API model info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    academic_logger.warning(f"404 error: {request.url}")
    return render_template('home.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    academic_logger.error(f"500 error: {str(error)}")
    flash('An internal error occurred. Please try again later.', 'error')
    return render_template('home.html'), 500

# Context processors for templates
@app.context_processor
def inject_globals():
    """Inject global variables into templates."""
    return {
        'current_year': datetime.now().year,
        'app_name': 'AcademicInsight ML',
        'version': '1.0.0'
    }

if __name__ == "__main__":
    # Configure logging
    academic_logger.info("Starting Academic Performance Prediction Application")
    
    # Create necessary directories
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the application
    app.run(host="0.0.0.0", port=5000, debug=True)
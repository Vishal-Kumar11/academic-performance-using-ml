# Architecture Documentation

## Academic Performance Prediction using Machine Learning

### System Architecture Overview

This document describes the technical architecture and design decisions for the Academic Performance Prediction system.

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   ML Pipeline   │
│   (Bootstrap 5) │◄──►│   (Flask)       │◄──►│   (13 Models)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Static Files  │    │   API Routes    │    │   Data Storage  │
│   (CSS/JS)      │    │   (REST)        │    │   (Artifacts)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Architecture

#### 1. Frontend Layer
- **Technology:** Bootstrap 5, JavaScript, HTML5/CSS3
- **Components:** Templates, Static assets, Interactive visualizations
- **Features:** Responsive design, Real-time validation, AJAX requests

#### 2. Backend Layer
- **Technology:** Flask 3.0, Python 3.8+
- **Components:** Routes, API endpoints, Error handling
- **Features:** RESTful API, Input validation, Logging

#### 3. ML Pipeline Layer
- **Technology:** scikit-learn, XGBoost, TensorFlow
- **Components:** Data processing, Model training, Prediction
- **Features:** 13 ML models, Cross-validation, Hyperparameter tuning

---

## 🔄 Data Flow Architecture

### Training Pipeline Flow

```
Raw Data → Data Ingestion → Data Transformation → Model Training → Model Evaluation → Model Persistence
    │            │                    │                    │                    │                    │
    ▼            ▼                    ▼                    ▼                    ▼                    ▼
CSV Files → Train/Test Split → Feature Engineering → 13 ML Models → Performance Metrics → PKL/H5 Files
```

### Prediction Pipeline Flow

```
User Input → Input Validation → Data Preprocessing → Model Loading → Prediction → Result Formatting → Response
    │              │                    │                    │              │                    │              │
    ▼              ▼                    ▼                    ▼              ▼                    ▼              ▼
Web Form → Type Checking → Feature Scaling → Load Model → ML Prediction → Confidence Score → JSON/HTML
```

---

## 🧠 Machine Learning Architecture

### Model Training Architecture

```python
# Training Pipeline
class AcademicModelTrainer:
    def __init__(self):
        self.models = {
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor(),
            'CatBoost': CatBoostRegressor(),
            'Neural Network': Sequential(),
            # ... 9 more models
        }
    
    def train_models(self):
        for name, model in self.models.items():
            # Hyperparameter tuning
            best_model = GridSearchCV(model, params, cv=5)
            # Cross-validation
            cv_scores = cross_val_score(best_model, X, y, cv=5)
            # Model evaluation
            metrics = self.evaluate_model(best_model)
```

### Model Selection Strategy

1. **Hyperparameter Tuning:** GridSearchCV with 5-fold CV
2. **Cross-Validation:** 5-fold stratified cross-validation
3. **Evaluation Metrics:** R², MAE, RMSE, MAPE, Explained Variance
4. **Model Selection:** Best R² score with lowest variance
5. **Model Persistence:** PKL for sklearn, H5 for TensorFlow

---

## 🎨 Frontend Architecture

### Template Structure

```
base.html (Base Template)
├── Navigation Bar
├── Main Content Block
├── Footer
└── Scripts

home.html (Landing Page)
├── Hero Section
├── Features Section
├── Statistics Section
└── Call-to-Action

predict.html (Prediction Form)
├── Form Validation
├── Input Fields
├── Help Cards
└── Submit Button

results.html (Results Display)
├── Prediction Display
├── Input Summary
├── Feature Importance Chart
└── Action Buttons
```

### JavaScript Architecture

```javascript
// Main Application Controller
class AcademicMLApp {
    constructor() {
        this.initializeComponents();
        this.setupEventListeners();
        this.initializeCharts();
    }
    
    // Form validation and submission
    handleFormSubmission() {
        // Real-time validation
        // AJAX requests
        // Error handling
    }
    
    // Chart initialization
    initializeCharts() {
        // Plotly integration
        // Interactive visualizations
    }
}
```

---

## 🔧 Backend Architecture

### Flask Application Structure

```python
# Application Factory Pattern
def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config.from_object(Config)
    
    # Blueprint registration
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    
    # Error handlers
    app.register_error_handler(404, not_found_error)
    app.register_error_handler(500, internal_error)
    
    return app
```

### Route Architecture

```python
# Main Routes
@app.route('/')                    # Home page
@app.route('/predict')            # Single prediction
@app.route('/batch')              # Batch processing
@app.route('/dashboard')          # Model dashboard
@app.route('/about')              # About page

# API Routes
@app.route('/api/predict')        # Prediction API
@app.route('/api/model-info')     # Model information API
```

---

## 📊 Data Architecture

### Data Processing Pipeline

```python
# Data Ingestion
class AcademicDataIngestion:
    def initiate_data_ingestion(self):
        # Load raw data
        # Train-test split
        # Save processed data
        
# Data Transformation
class AcademicDataTransformation:
    def get_data_transformer_object(self):
        # Numerical pipeline
        # Categorical pipeline
        # Column transformer
        
# Model Training
class AcademicModelTrainer:
    def initiate_model_trainer(self):
        # Load processed data
        # Train 13 models
        # Evaluate performance
        # Save best model
```

### Feature Engineering

```python
# Numerical Features
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Features
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot_encoder', OneHotEncoder()),
    ('scaler', StandardScaler(with_mean=False))
])
```

---

## 🔒 Security Architecture

### Input Validation

```python
class AcademicDataValidator:
    def validate_prediction_input(self, input_data):
        # Type checking
        # Range validation
        # Format validation
        # Security checks
```

### Error Handling

```python
# Custom Exception Handling
class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_detail

# Global Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('home.html'), 404
```

---

## 📈 Performance Architecture

### Caching Strategy

```python
# Model Caching
@lru_cache(maxsize=1)
def load_model():
    return joblib.load('artifacts/academic_performance_model.pkl')

# Preprocessing Caching
@lru_cache(maxsize=1)
def load_preprocessor():
    return joblib.load('artifacts/academic_preprocessor.pkl')
```

### Optimization Techniques

1. **Model Persistence:** Pre-trained models loaded once
2. **Batch Processing:** Efficient CSV processing
3. **Lazy Loading:** Components loaded on demand
4. **Memory Management:** Proper cleanup and garbage collection

---

## 🚀 Deployment Architecture

### Production Deployment

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Server     │    │   Application   │
│   (AWS ALB)     │◄──►│   (Nginx)        │◄──►│   (Flask)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CDN           │    │   Static Files  │    │   ML Models     │
│   (CloudFront)  │    │   (S3)          │    │   (S3)          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Container Architecture

```dockerfile
# Multi-stage Docker build
FROM python:3.8-slim as base
# Install system dependencies
# Install Python packages
# Copy application code
# Expose port 5000
```

---

## 🔍 Monitoring Architecture

### Logging Strategy

```python
# Structured Logging
import logging

# Application logs
app_logger = logging.getLogger('academic_ml')

# Model logs
model_logger = logging.getLogger('academic_ml.model')

# API logs
api_logger = logging.getLogger('academic_ml.api')
```

### Metrics Collection

1. **Performance Metrics:** Response time, throughput
2. **Model Metrics:** Prediction accuracy, confidence scores
3. **System Metrics:** CPU, memory, disk usage
4. **User Metrics:** Page views, prediction requests

---

## 🔄 CI/CD Architecture

### Continuous Integration

```yaml
# GitHub Actions Workflow
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - Checkout code
      - Setup Python
      - Install dependencies
      - Run tests
      - Code quality checks
```

### Deployment Pipeline

1. **Code Commit** → GitHub
2. **Automated Testing** → pytest, flake8
3. **Build Docker Image** → Docker Hub
4. **Deploy to AWS** → Elastic Beanstalk
5. **Health Checks** → Application monitoring

---

## 📚 API Architecture

### RESTful API Design

```python
# API Endpoints
GET  /api/model-info          # Model information
POST /api/predict             # Single prediction
POST /api/batch-predict       # Batch prediction
GET  /api/health              # Health check
```

### API Response Format

```json
{
  "success": true,
  "prediction": 85.7,
  "confidence": "high",
  "model_used": "XGBoost",
  "processing_time": 0.023,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 🎯 Future Architecture Considerations

### Scalability Improvements

1. **Microservices:** Split into separate services
2. **Message Queues:** Async processing for batch jobs
3. **Database:** Persistent storage for predictions
4. **Caching:** Redis for model caching
5. **Load Balancing:** Multiple application instances

### Technology Upgrades

1. **FastAPI:** Replace Flask for better performance
2. **MLflow:** Model versioning and tracking
3. **Kubernetes:** Container orchestration
4. **GraphQL:** More flexible API queries
5. **WebSockets:** Real-time updates

---

This architecture document provides a comprehensive overview of the system design, helping developers understand the codebase structure and make informed decisions about future enhancements.

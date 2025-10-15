# API Documentation

## Academic Performance Prediction API

This document describes the RESTful API endpoints for the Academic Performance Prediction system.

---

## 🌐 Base URL

```
Production: https://your-domain.com/api
Development: http://localhost:5000/api
```

---

## 🔐 Authentication

Currently, the API does not require authentication. In production, consider implementing:
- API key authentication
- JWT tokens
- Rate limiting

---

## 📊 API Endpoints

### 1. Model Information

#### GET `/api/model-info`

Returns information about the trained models and system capabilities.

**Response:**
```json
{
  "total_models": 13,
  "best_model": "XGBoost Regressor",
  "best_score": 0.942,
  "metrics": [
    "R² Score",
    "MAE", 
    "RMSE",
    "MAPE",
    "Explained Variance"
  ],
  "cv_folds": 5,
  "features": [
    "gender",
    "race_ethnicity", 
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "reading_score",
    "writing_score"
  ],
  "version": "1.0.0",
  "last_updated": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

### 2. Single Prediction

#### POST `/api/predict`

Predicts academic performance for a single student.

**Request Body:**
```json
{
  "gender": "female",
  "race_ethnicity": "group C",
  "parental_level_of_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation_course": "completed",
  "reading_score": 85,
  "writing_score": 88
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 87.3,
  "confidence": "high",
  "model_used": "XGBoost Regressor",
  "processing_time": 0.023,
  "timestamp": "2024-01-15T10:30:00Z",
  "feature_importance": {
    "writing_score": 0.85,
    "reading_score": 0.78,
    "test_preparation_course": 0.72,
    "parental_level_of_education": 0.68,
    "lunch": 0.65,
    "gender": 0.58,
    "race_ethnicity": 0.45
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "errors": [
    "reading_score must be between 0 and 100",
    "gender is required"
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200` - Success
- `400` - Bad request (validation errors)
- `500` - Server error

---

### 3. Batch Prediction

#### POST `/api/batch-predict`

Processes multiple student records for batch prediction.

**Request Body:**
```json
{
  "data": [
    {
      "gender": "female",
      "race_ethnicity": "group C",
      "parental_level_of_education": "bachelor's degree",
      "lunch": "standard",
      "test_preparation_course": "completed",
      "reading_score": 85,
      "writing_score": 88
    },
    {
      "gender": "male",
      "race_ethnicity": "group A",
      "parental_level_of_education": "high school",
      "lunch": "free/reduced",
      "test_preparation_course": "none",
      "reading_score": 72,
      "writing_score": 75
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "total_records": 2,
  "processed_records": 2,
  "failed_records": 0,
  "predictions": [
    {
      "index": 0,
      "prediction": 87.3,
      "confidence": "high",
      "quality": "High"
    },
    {
      "index": 1,
      "prediction": 68.7,
      "confidence": "medium",
      "quality": "Medium"
    }
  ],
  "summary": {
    "mean_prediction": 78.0,
    "min_prediction": 68.7,
    "max_prediction": 87.3,
    "std_prediction": 9.3
  },
  "processing_time": 0.156,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200` - Success
- `400` - Bad request (validation errors)
- `413` - Payload too large (max 1000 records)
- `500` - Server error

---

### 4. Health Check

#### GET `/api/health`

Returns the health status of the API and ML models.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "models": {
    "loaded": true,
    "count": 13,
    "best_model": "XGBoost Regressor",
    "last_trained": "2024-01-15T08:00:00Z"
  },
  "system": {
    "cpu_usage": 15.2,
    "memory_usage": 45.8,
    "disk_usage": 23.1
  }
}
```

**Status Codes:**
- `200` - Healthy
- `503` - Service unavailable

---

## 📝 Data Models

### Student Record

```json
{
  "gender": "string",                    // Required: "male" | "female"
  "race_ethnicity": "string",           // Required: "group A" | "group B" | "group C" | "group D" | "group E"
  "parental_level_of_education": "string", // Required: "associate's degree" | "bachelor's degree" | "high school" | "master's degree" | "some college" | "some high school"
  "lunch": "string",                    // Required: "free/reduced" | "standard"
  "test_preparation_course": "string",  // Required: "none" | "completed"
  "reading_score": "number",            // Required: 0-100
  "writing_score": "number"             // Required: 0-100
}
```

### Prediction Response

```json
{
  "success": "boolean",
  "prediction": "number",               // Predicted math score (0-100)
  "confidence": "string",              // "low" | "medium" | "high"
  "model_used": "string",              // Name of the model used
  "processing_time": "number",         // Processing time in seconds
  "timestamp": "string",               // ISO 8601 timestamp
  "feature_importance": "object"       // Optional: Feature importance scores
}
```

---

## 🔍 Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": "string",                   // Error message
  "error_code": "string",              // Error code
  "details": "object",                 // Additional error details
  "timestamp": "string"                // ISO 8601 timestamp
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Input validation failed | 400 |
| `MODEL_NOT_FOUND` | ML model not available | 503 |
| `PROCESSING_ERROR` | Error during prediction | 500 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `PAYLOAD_TOO_LARGE` | Request too large | 413 |

---

## 📊 Rate Limiting

**Current Limits:**
- Single predictions: 100 requests/minute
- Batch predictions: 10 requests/minute
- Model info: 1000 requests/minute

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

---

## 🧪 Testing

### Example Requests

#### cURL Examples

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "race_ethnicity": "group C",
    "parental_level_of_education": "bachelor'\''s degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 85,
    "writing_score": 88
  }'
```

**Model Information:**
```bash
curl -X GET http://localhost:5000/api/model-info
```

**Health Check:**
```bash
curl -X GET http://localhost:5000/api/health
```

#### Python Examples

```python
import requests
import json

# Single prediction
url = "http://localhost:5000/api/predict"
data = {
    "gender": "female",
    "race_ethnicity": "group C",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 85,
    "writing_score": 88
}

response = requests.post(url, json=data)
result = response.json()

if result["success"]:
    print(f"Predicted Math Score: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"Error: {result['errors']}")
```

#### JavaScript Examples

```javascript
// Single prediction
const predictStudent = async (studentData) => {
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(studentData)
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log(`Predicted Score: ${result.prediction}`);
      console.log(`Confidence: ${result.confidence}`);
    } else {
      console.error('Prediction failed:', result.errors);
    }
  } catch (error) {
    console.error('Request failed:', error);
  }
};

// Usage
const studentData = {
  gender: "female",
  race_ethnicity: "group C",
  parental_level_of_education: "bachelor's degree",
  lunch: "standard",
  test_preparation_course: "completed",
  reading_score: 85,
  writing_score: 88
};

predictStudent(studentData);
```

---

## 🔄 Webhooks

### Prediction Completed Webhook

**Endpoint:** `POST /webhooks/prediction-completed`

**Payload:**
```json
{
  "event": "prediction.completed",
  "data": {
    "prediction_id": "uuid",
    "student_id": "uuid",
    "prediction": 87.3,
    "confidence": "high",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

---

## 📈 Monitoring

### Metrics Endpoint

**Endpoint:** `GET /api/metrics`

**Response:**
```json
{
  "predictions_total": 15420,
  "predictions_successful": 15380,
  "predictions_failed": 40,
  "average_processing_time": 0.025,
  "models_performance": {
    "XGBoost": 0.942,
    "LightGBM": 0.938,
    "CatBoost": 0.935
  },
  "system_uptime": "99.9%"
}
```

---

## 🔧 SDKs and Libraries

### Python SDK

```python
from academic_ml_client import AcademicMLClient

client = AcademicMLClient(api_key="your-api-key")

# Single prediction
prediction = client.predict({
    "gender": "female",
    "race_ethnicity": "group C",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 85,
    "writing_score": 88
})

print(f"Predicted score: {prediction.score}")
print(f"Confidence: {prediction.confidence}")
```

### JavaScript SDK

```javascript
import { AcademicMLClient } from 'academic-ml-client';

const client = new AcademicMLClient({
  apiKey: 'your-api-key',
  baseURL: 'https://api.academicml.com'
});

// Single prediction
const prediction = await client.predict({
  gender: "female",
  race_ethnicity: "group C",
  parental_level_of_education: "bachelor's degree",
  lunch: "standard",
  test_preparation_course: "completed",
  reading_score: 85,
  writing_score: 88
});

console.log(`Predicted score: ${prediction.score}`);
console.log(`Confidence: ${prediction.confidence}`);
```

---

## 📞 Support

For API support and questions:
- 📧 Email: api-support@academicml.com
- 📚 Documentation: https://docs.academicml.com
- 🐛 Bug Reports: https://github.com/yourusername/academic-performance-ml/issues
- 💬 Community: https://discord.gg/academicml

---

This API documentation is automatically generated and updated with each release. For the latest version, visit our [API Reference](https://api.academicml.com/docs).

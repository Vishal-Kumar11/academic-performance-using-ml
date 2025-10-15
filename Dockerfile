# Academic Performance Prediction using Machine Learning
# Multi-stage Docker build for production deployment

# Stage 1: Base Python environment
FROM python:3.8-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Development
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    sphinx \
    sphinx-rtd-theme

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p logs artifacts uploads

# Set permissions
RUN chmod +x application.py

# Expose port
EXPOSE 5000

# Development command
CMD ["python", "application.py"]

# Stage 4: Production
FROM dependencies as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p logs artifacts uploads && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set production environment
ENV FLASK_ENV=production \
    FLASK_APP=application.py

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "30", "application:app"]

# Stage 5: Testing
FROM development as testing

# Copy test files
COPY tests/ tests/

# Run tests
RUN pytest tests/ --cov=src --cov-report=html --cov-report=term

# Stage 6: Documentation
FROM development as docs

# Build documentation
RUN sphinx-build -b html docs/ docs/_build/html

# Serve documentation
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "--directory", "docs/_build/html"]

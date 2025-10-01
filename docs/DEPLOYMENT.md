# BrandLens Deployment Guide

## Overview

This guide covers production deployment, configuration, monitoring, and operational considerations for BrandLens. The system is designed for both local development and production environments.

## System Requirements

### Minimum Requirements

- **Python**: 3.11+ (recommended: 3.12)
- **Memory**: 2GB RAM (4GB recommended for large analyses)
- **Storage**: 1GB disk space (additional space for caching)
- **Network**: Reliable internet connection for API calls

### Recommended Production Environment

- **Python**: 3.12 with virtual environment
- **Memory**: 8GB RAM for concurrent operations
- **Storage**: 10GB SSD for caching and logs
- **Network**: Low-latency connection (< 100ms to API endpoints)

## Installation Methods

### Method 1: Package Installation (Recommended)

```bash
# Create virtual environment
python3.12 -m venv brandlens-env
source brandlens-env/bin/activate

# Install using uv (fastest)
pip install uv
uv sync

# Or install using pip
pip install -r requirements.txt
```

### Method 2: Development Installation

```bash
# Clone repository
git clone <repository-url>
cd brandlens

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Method 3: Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY setup.py ./

# Install application
RUN pip install -e .

# Create non-root user
RUN useradd -m -s /bin/bash brandlens
USER brandlens

# Set environment
ENV PYTHONPATH=/app
ENV CACHE_DIR=/app/.cache

# Create cache directory
RUN mkdir -p /app/.cache

ENTRYPOINT ["python", "-m", "src"]
```

## Configuration Management

### Environment Configuration

#### Production .env Template

```bash
# =============================================================================
# PRODUCTION CONFIGURATION - BrandLens
# =============================================================================

# REQUIRED: API Keys (obtain from providers)
GEMINI_API_KEY=your_production_gemini_key
TAVILY_API_KEY=your_production_tavily_key

# Model Configuration (optimized for production)
GEMINI_MODEL=models/gemini-2.5-flash
GEMINI_MAX_TOKENS=8192
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_RETRIES=5

# Search Configuration (production limits)
TAVILY_SEARCH_DEPTH=advanced
TAVILY_INCLUDE_RAW_CONTENT=true
TAVILY_MAX_RESULTS=10

# Performance Configuration
MAX_SEARCHES_PER_QUERY=3
MAX_SOURCES_PER_SEARCH=5
TOKEN_COMPRESSION_TARGET=0.25
ASYNC_TIMEOUT=60

# Caching (production optimized)
CACHE_DIR=/var/cache/brandlens
CACHE_TTL=7200
CACHE_ENABLED=true

# Logging (production settings)
LOG_LEVEL=INFO
LOG_FILE=/var/log/brandlens/brandlens.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Performance Monitoring
ENABLE_PERFORMANCE_TRACKING=true
ENABLE_COST_TRACKING=true
COST_ALERT_THRESHOLD=1.00

# Security
VALIDATE_SSL=true
API_TIMEOUT=30
MAX_CONTENT_LENGTH=2097152

# Rate Limiting (production rates)
GEMINI_RATE_LIMIT=60
TAVILY_RATE_LIMIT=100
```

#### Configuration Validation

```bash
# Validate configuration before deployment
python -m src validate-config

# Check API connectivity
python -m src info
```

### Secrets Management

#### Using Environment Variables

```bash
# Set in shell environment
export GEMINI_API_KEY="your_key_here"
export TAVILY_API_KEY="your_key_here"

# Verify secrets are loaded
python -m src validate-config
```

#### Using Docker Secrets

```yaml
# docker-compose.yml
version: '3.8'
services:
  brandlens:
    build: .
    secrets:
      - gemini_api_key
      - tavily_api_key
    environment:
      GEMINI_API_KEY_FILE: /run/secrets/gemini_api_key
      TAVILY_API_KEY_FILE: /run/secrets/tavily_api_key

secrets:
  gemini_api_key:
    external: true
  tavily_api_key:
    external: true
```

#### Using HashiCorp Vault

```bash
# Retrieve secrets from Vault
export GEMINI_API_KEY=$(vault kv get -field=api_key secret/brandlens/gemini)
export TAVILY_API_KEY=$(vault kv get -field=api_key secret/brandlens/tavily)
```

## Production Deployment Patterns

### Pattern 1: Single-Instance Deployment

**Use Case**: Small-scale deployments, development environments

```bash
# Setup production environment
sudo mkdir -p /opt/brandlens
sudo mkdir -p /var/log/brandlens
sudo mkdir -p /var/cache/brandlens

# Deploy application
sudo cp -r src/ /opt/brandlens/
sudo cp requirements.txt /opt/brandlens/
sudo cp .env /opt/brandlens/

# Create service user
sudo useradd -r -s /bin/false brandlens
sudo chown -R brandlens:brandlens /opt/brandlens
sudo chown -R brandlens:brandlens /var/log/brandlens
sudo chown -R brandlens:brandlens /var/cache/brandlens

# Install dependencies
cd /opt/brandlens
sudo -u brandlens python3.12 -m venv venv
sudo -u brandlens ./venv/bin/pip install -r requirements.txt
```

### Pattern 2: Containerized Deployment

**Use Case**: Scalable deployments, cloud environments

```yaml
# docker-compose.production.yml
version: '3.8'
services:
  brandlens:
    build:
      context: .
      dockerfile: Dockerfile.prod
    environment:
      - CACHE_DIR=/app/cache
      - LOG_LEVEL=INFO
    volumes:
      - cache_data:/app/cache
      - log_data:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-m", "src", "validate-config"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  cache_data:
  log_data:
```

### Pattern 3: Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: brandlens
  labels:
    app: brandlens
spec:
  replicas: 3
  selector:
    matchLabels:
      app: brandlens
  template:
    metadata:
      labels:
        app: brandlens
    spec:
      containers:
      - name: brandlens
        image: brandlens:production
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: brandlens-secrets
              key: gemini-api-key
        - name: TAVILY_API_KEY
          valueFrom:
            secretKeyRef:
              name: brandlens-secrets
              key: tavily-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: cache
          mountPath: /app/cache
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: brandlens-cache
```

## Monitoring and Observability

### Application Monitoring

#### Built-in Metrics

BrandLens provides comprehensive built-in monitoring:

```python
# Performance metrics collected automatically
- Response time per component
- Token usage and costs
- API call success/failure rates
- Cache hit rates
- Extraction accuracy scores
- Budget utilization
```

#### Custom Metrics Integration

```python
# Prometheus metrics example
from prometheus_client import Counter, Histogram, Gauge

# Add to your monitoring setup
REQUEST_COUNT = Counter('brandlens_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('brandlens_request_duration_seconds', 'Request duration')
ACTIVE_ANALYSES = Gauge('brandlens_active_analyses', 'Active analyses')
```

#### Health Check Endpoint

```bash
# Built-in health check
python -m src validate-config
echo $?  # 0 = healthy, non-zero = unhealthy
```

### Logging Configuration

#### Production Logging Setup

```python
# Structured logging configuration
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'production': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "file": "%(pathname)s", "line": %(lineno)d}'
        }
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/brandlens/brandlens.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
            'formatter': 'production'
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/brandlens/error.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'formatter': 'json'
        }
    },
    'loggers': {
        'brandlens': {
            'handlers': ['file', 'error_file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
```

#### Log Aggregation

```yaml
# ELK Stack integration
filebeat.inputs:
- type: log
  paths:
    - /var/log/brandlens/*.log
  fields:
    service: brandlens
    environment: production
  json.keys_under_root: true
  json.add_error_key: true
```

### Performance Monitoring

#### Response Time Monitoring

```bash
# Monitor response times
tail -f /var/log/brandlens/brandlens.log | grep "total_time_ms"
```

#### Cost Monitoring

```bash
# Monitor API costs
tail -f /var/log/brandlens/brandlens.log | grep "total_cost_usd"
```

#### Alert Configuration

```yaml
# Alertmanager rules
groups:
- name: brandlens
  rules:
  - alert: BrandLensHighLatency
    expr: brandlens_request_duration_seconds > 30
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "BrandLens high latency detected"

  - alert: BrandLensHighCost
    expr: brandlens_daily_cost_usd > 10
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "BrandLens daily cost threshold exceeded"
```

## Security Considerations

### API Key Security

#### Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables or secrets management**
3. **Rotate keys regularly**
4. **Monitor API key usage**
5. **Implement key rotation procedures**

#### Key Rotation Procedure

```bash
#!/bin/bash
# Key rotation script
NEW_GEMINI_KEY="new_key_here"
NEW_TAVILY_KEY="new_key_here"

# Update environment
export GEMINI_API_KEY="$NEW_GEMINI_KEY"
export TAVILY_API_KEY="$NEW_TAVILY_KEY"

# Validate new keys
python -m src validate-config

if [ $? -eq 0 ]; then
    echo "Key rotation successful"
    # Update persistent configuration
    sed -i "s/GEMINI_API_KEY=.*/GEMINI_API_KEY=$NEW_GEMINI_KEY/" /opt/brandlens/.env
    sed -i "s/TAVILY_API_KEY=.*/TAVILY_API_KEY=$NEW_TAVILY_KEY/" /opt/brandlens/.env
else
    echo "Key rotation failed"
    exit 1
fi
```

### Network Security

#### SSL/TLS Configuration

```python
# Enforce HTTPS for all API calls
VALIDATE_SSL=true
API_TIMEOUT=30
```

#### Firewall Configuration

```bash
# Only allow necessary outbound connections
# Gemini API (generativelanguage.googleapis.com)
# Tavily API (api.tavily.com)

sudo ufw allow out 443/tcp
sudo ufw allow out 80/tcp  # For redirects only
sudo ufw deny out to any port 22  # Block SSH outbound
```

### Input Validation

#### Content Size Limits

```python
# Prevent resource exhaustion
MAX_CONTENT_LENGTH=2097152  # 2MB
MAX_QUERY_LENGTH=1000      # 1000 characters
MAX_BRAND_NAME_LENGTH=100  # 100 characters
```

#### Input Sanitization

```python
# Automatic sanitization in place
- URL validation and normalization
- Brand name validation
- Query content filtering
- Response size limiting
```

## Backup and Recovery

### Data Backup

#### Cache Backup

```bash
#!/bin/bash
# Cache backup script
BACKUP_DIR="/backup/brandlens/$(date +%Y%m%d)"
CACHE_DIR="/var/cache/brandlens"

mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_DIR/cache.tar.gz" -C "$CACHE_DIR" .
```

#### Configuration Backup

```bash
# Configuration backup
cp /opt/brandlens/.env /backup/brandlens/config/env.$(date +%Y%m%d)
cp /opt/brandlens/requirements.txt /backup/brandlens/config/requirements.$(date +%Y%m%d)
```

### Disaster Recovery

#### Recovery Procedure

1. **Restore configuration files**
2. **Validate API keys**
3. **Restore cache data (optional)**
4. **Validate system functionality**
5. **Resume operations**

```bash
#!/bin/bash
# Disaster recovery script
RESTORE_DATE="20250129"
BACKUP_DIR="/backup/brandlens/$RESTORE_DATE"

# Restore configuration
cp "$BACKUP_DIR/env.$RESTORE_DATE" /opt/brandlens/.env

# Restore cache (optional)
tar -xzf "$BACKUP_DIR/cache.tar.gz" -C /var/cache/brandlens/

# Validate system
cd /opt/brandlens
./venv/bin/python -m src validate-config

if [ $? -eq 0 ]; then
    echo "Recovery successful"
else
    echo "Recovery validation failed"
    exit 1
fi
```

## Performance Optimization

### Production Optimizations

#### Python Optimizations

```bash
# Use optimized Python installation
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1

# Increase Python performance
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
```

#### Memory Optimization

```python
# Production memory settings
- Use memory-efficient data structures
- Enable garbage collection optimization
- Configure appropriate worker processes
- Monitor memory usage patterns
```

#### Cache Optimization

```bash
# Cache configuration for production
CACHE_TTL=7200          # 2 hours
CACHE_MAX_SIZE=1000     # 1000 entries
CACHE_CLEANUP_INTERVAL=3600  # 1 hour
```

### Scaling Considerations

#### Horizontal Scaling

- **Stateless design** allows for multiple instances
- **Shared cache** using Redis or similar
- **Load balancing** for request distribution
- **Circuit breakers** for external API protection

#### Vertical Scaling

- **Increase memory** for larger analyses
- **Add CPU cores** for parallel processing
- **Use SSD storage** for faster cache access
- **Optimize network** for API latency

## Troubleshooting

### Common Issues

#### API Key Problems

```bash
# Symptom: Configuration validation fails
# Solution: Check API key validity
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
     https://generativelanguage.googleapis.com/v1/models
```

#### Cache Issues

```bash
# Symptom: Slow performance despite caching
# Solution: Clear and rebuild cache
rm -rf /var/cache/brandlens/*
python -m src info  # Rebuilds cache
```

#### Memory Issues

```bash
# Symptom: Out of memory errors
# Solution: Monitor and optimize memory usage
free -h
ps aux | grep python
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m src analyze "Apple" "iPhone features" --log-level DEBUG
```

### Performance Profiling

```bash
# Profile performance
python -m cProfile -o profile.stats -m src analyze "Apple" "iPhone features"
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## Maintenance

### Regular Maintenance Tasks

#### Daily
- Monitor system health and performance
- Check error logs for issues
- Verify API quota usage
- Review cost metrics

#### Weekly
- Rotate log files
- Clean up old cache entries
- Review performance metrics
- Update dependencies (security patches)

#### Monthly
- Full system backup
- Capacity planning review
- Security audit
- Performance optimization review

### Update Procedures

#### Application Updates

```bash
#!/bin/bash
# Update procedure
cd /opt/brandlens

# Backup current version
tar -czf "backup-$(date +%Y%m%d).tar.gz" .

# Update code
git pull origin main

# Update dependencies
./venv/bin/pip install -r requirements.txt

# Validate configuration
./venv/bin/python -m src validate-config

# Test functionality
./venv/bin/python -m src info
```

#### Dependency Updates

```bash
# Check for security updates
pip-audit

# Update dependencies
pip install --upgrade -r requirements.txt

# Test system
python -m src validate-config
```

---

**Deployment Guide Version**: 1.0
**Last Updated**: 2025-01-29
**Supported Environments**: Linux, macOS, Windows (WSL)
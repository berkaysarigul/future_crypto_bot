# Multi-stage build for production-ready Docker image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r tradingbot && useradd -r -g tradingbot tradingbot

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY advanced_futures_bot/ ./advanced_futures_bot/
COPY config.yaml .
COPY README.md .

# Create necessary directories
RUN mkdir -p logs checkpoints data models && \
    chown -R tradingbot:tradingbot /app

# Switch to non-root user
USER tradingbot

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "advanced_futures_bot.main", "--mode", "trading"]

# Development stage
FROM base as development

# Install development dependencies
USER root
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit

USER tradingbot

# Production stage
FROM base as production

# Additional production optimizations
ENV PYTHONOPTIMIZE=1

# Copy production-specific config
COPY --chown=tradingbot:tradingbot config.prod.yaml ./config.yaml

# Security: Remove unnecessary files
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -delete

# Expose port for monitoring (if needed)
EXPOSE 8080

# Training stage
FROM base as training

# Install additional training dependencies
USER root
RUN pip install --no-cache-dir \
    ray[rllib] \
    tensorboard \
    wandb

USER tradingbot

# Copy training scripts
COPY --chown=tradingbot:tradingbot scripts/train.py ./scripts/
COPY --chown=tradingbot:tradingbot scripts/hyperopt.py ./scripts/

# Default command for training
CMD ["python", "scripts/train.py"]

# Testing stage
FROM base as testing

# Install testing dependencies
USER root
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    pytest-benchmark

USER tradingbot

# Copy tests
COPY --chown=tradingbot:tradingbot tests/ ./tests/

# Default command for testing
CMD ["pytest", "tests/", "-v", "--cov=advanced_futures_bot"]

# Documentation
LABEL maintainer="Advanced Futures Bot Team" \
      version="1.0.0" \
      description="Production-ready crypto futures trading bot with RL and sentiment analysis" \
      org.opencontainers.image.source="https://github.com/your-repo/advanced-futures-bot" 
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY api/requirements-api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data/processed data/vector_db

# Download and prepare data
RUN python setup_data.py && python build_vector_db.py

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Run both API and Streamlit
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
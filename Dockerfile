# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port and run the Streamlit app
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

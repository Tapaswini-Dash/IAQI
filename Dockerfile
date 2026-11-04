FROM python:3.10-slim

WORKDIR /app

# Copy project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the default port
EXPOSE 7860

# Run Uvicorn pointing to your app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
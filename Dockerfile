# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

RUN pip install --upgrade pip

# Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt


# Copy the rest of the app
COPY . .

# Expose port (optional, e.g., if using Flask/FastAPI)
EXPOSE 5000

# Run the app
CMD ["python", "main.py"]

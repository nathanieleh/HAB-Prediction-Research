FROM mcr.microsoft.com/playwright/python:v1.58.0-jammy

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install chromium

# Copy project files
COPY . .

# Persistent output directory
RUN mkdir -p /app/output

WORKDIR /app/Code/Scripts

CMD ["python", "forecast.py", "configs/nathan_config.yaml"]

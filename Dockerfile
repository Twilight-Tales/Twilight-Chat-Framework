# Use an appropriate base image, e.g., python:3.10-slim
FROM python:3.11.2-slim-bullseye

# Update package lists and install git
RUN apt-get update && \
    apt-get install -y git && \
    # Clean up the cache to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONDONTWRITEBYTECODE 1
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1

# Install dependencies
WORKDIR /app
COPY requirements.txt /app/requirements.txt
# '-m pip' instead of 'pip3 'to avoid this bug https://github.com/pypa/pip/issues/5599
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/requirements.txt

# Copy your application code into the container
COPY . /app/

EXPOSE 1680

CMD ["python", "-m", "chainlit", "run", "/app/app.py", "-h", "--port", "1680"]
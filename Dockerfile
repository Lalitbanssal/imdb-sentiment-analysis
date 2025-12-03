# 1. Base Image: Use a lightweight Python version (Slim is faster and smaller)
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy all files from your current folder to the container's /app folder
COPY . .

# 4. Install dependencies from the clean requirements.txt we just made
RUN pip install --no-cache-dir -r requirements.txt

# 5. Download NLTK data explicitly during the build
# This prevents "Resource not found" errors when the app tries to run
RUN python -m nltk.downloader punkt stopwords

# 6. Expose the default Streamlit port
EXPOSE 8501

# 7. Command to run the app
# --server.address=0.0.0.0 is CRITICAL for Docker to accept external connections
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
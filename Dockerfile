FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8080

# Create .streamlit directory and config
RUN mkdir -p /app/.streamlit

# Create Streamlit config file
RUN echo '\
[server]\n\
port = 8080\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
maxUploadSize = 200\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /app/.streamlit/config.toml

# Command to run the Streamlit app
CMD ["streamlit", "run", "spotify_dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"] 
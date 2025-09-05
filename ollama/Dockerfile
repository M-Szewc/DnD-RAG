# Base image for Ollama
FROM ollama/ollama:0.3.6

# Set working directory
WORKDIR /app

# Copy the start script into the container
COPY entrypoint.sh /usr/local/bin

# Start the LLM
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
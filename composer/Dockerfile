# Use a base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y apt-utils && apt-get install -y build-essential bash curl unzip wget git libgl1-mesa-glx libglib2.0-0

# Run pip dependencies
COPY /composer/requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt && pip3 install ipywidgets jupyterlab==4.0.0 jupyter-archive==3.4.0 jupyter

# Copy notebooks
# WORKDIR /app/notebooks
# COPY /notebooks .

# Expose port 888 for JupyterLab
EXPOSE 8888

# WORKDIR /app
# Start JupyterLab when the container runs
COPY composer/entrypoint.sh /usr/local/bin
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
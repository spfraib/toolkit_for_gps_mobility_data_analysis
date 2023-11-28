# Use the jupyter/pyspark-notebook as a base image
FROM jupyter/pyspark-notebook

# Copy your package into the container
COPY ../wbgps /home/jovyan/WorldBank/wbgps

# Set the working directory
WORKDIR /home/jovyan/WorldBank/wbgps

# Install your package
RUN pip install .


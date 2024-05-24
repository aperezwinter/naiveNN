# Use a Linux base image
FROM ubuntu:latest

# Create the working directory
RUN mkdir /app

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . .

# Install any necessary dependencies
RUN apt-get update && apt-get install -y python python-pip

# Install project dependencies from requirements.txt
RUN pip install -r requirements.txt

# Specify the command to run your project
CMD [ "python", "naivenn/run/noreg_varhl.py" ]
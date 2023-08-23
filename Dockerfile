# Use the official TensorFlow GPU image as base
FROM tensorflow/tensorflow:2.8.0-gpu

# Set the working directory inside the container
WORKDIR /emotion-model

# Copy only the requirements file to the container
COPY requirements.txt /emotion-model

# Install dependencies (including Jupyter) if requirements change
RUN pip install -r requirements.txt

# Copy the rest of your local directory to the container
COPY . /emotion-model

# Expose the desired port for Jupyter
EXPOSE 8888

ENV CURLOPT_SSL_VERIFYHOST=0
ENV CURLOPT_SSL_VERIFYPEER=0

# Start Jupyter Notebook when the container runs
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
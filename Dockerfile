#base image with CUDA and cuDNN support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

#install basic utilities and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

#upgrade pip and install TensorFlow (GPU)
RUN pip install --upgrade pip
RUN pip install tensorflow==2.15.0

#verify CUDA is visible to TensorFlow
RUN python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"

#set working directory
WORKDIR /workspace

#default command
CMD [ "python" ]

# Details of the base image are here: hub.docker.com/r/jupyter/scipy-notebook
# Tag [2ce7c06a61a1] is latest image as of August 12, 2019 
# It runs Python 3.7.3

FROM jupyter/scipy-notebook:2ce7c06a61a1 

MAINTAINER Jon Krohn <jon@untapt.com>

USER $NB_USER

# Downgrade NumPy due to version 1.17.0 tensorflow warning:
RUN pip install numpy==1.16.4 

# Install TensorFlow: 
RUN pip install tensorflow==2.0.0-beta1

# Install PyTorch libraries:
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl 
RUN pip install torchvision==0.4.0
RUN pip install torchsummary==1.5.1

# Install TFlearn for its Oxford Flowers 17 dataset: 
RUN pip install tflearn==0.3.2 

# Details of the base image are here: hub.docker.com/r/jupyter/scipy-notebook
# Tag [2ce7c06a61a1] is latest image as of August 12, 2019 
# It runs Python 3.7.3

FROM jupyter/scipy-notebook:2ce7c06a61a1 

MAINTAINER Jon Krohn <jon@untapt.com>

USER $NB_USER

# Install TensorFlow: 
RUN pip install tensorflow==2.0.0

# Install PyTorch libraries:
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl 
RUN pip install torchvision==0.4.0
RUN pip install torchsummary==1.5.1

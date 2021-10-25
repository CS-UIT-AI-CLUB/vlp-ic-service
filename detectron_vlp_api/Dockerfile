FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    apt install --no-install-recommends -y build-essential software-properties-common git python3-pip libssl-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.7 python3.7-dev python3.7-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Register the version in alternatives (and set higher priority to 3.7)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
RUN python3 -m pip install --upgrade pip
RUN pip3 install setuptools

WORKDIR /workingspace

# Download weights from onedrive
# COPY download_weights.sh /workingspace/

# RUN bash download_weights.sh 

# Copy detectron folder 
COPY detectron/ /workingspace/detectron/

# Install Python dependencies
RUN pip3 install -r detectron/requirements.txt 
COPY requirements.txt /workingspace
RUN pip3 install -r requirements.txt
# Make file  
RUN cd detectron && make

# Copy detectron-vlp folder
COPY . /workingspace/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Set lib path 
ENV PYTHONPATH "${PYTHONPATH}:/workingspace/lib"

EXPOSE 5055
CMD ["bash", "run.sh"]

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV debian_frontend noninteractive

RUN apt-get update && \
    apt install --no-install-recommends -y build-essential software-properties-common wget git python3-pip libssl-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.7 python3.7-dev python3.7-distutils libsm6 libxext6 ffmpeg && \
    apt clean && rm -rf /var/lib/apt/lists/*

#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

WORKDIR /workingspace

COPY detectron/ /workingspace/detectron/

RUN pip3 install --upgrade pip && pip3 install \
    aniso8601==9.0.1 \
    asn1crypto==0.24.0 \
    asttokens==2.0.5\
    cached-property==1.5.2\
    click==8.0.3\
    colorama==0.4.4\
    cryptography==2.1.4\
    cycler==0.10.0\
    Cython==0.29.24\
    easydict==1.9\
    executing==0.8.2\
    Flask==2.0.2\
    Flask-RESTful==0.3.9\
    future==0.18.2\
    h5py==3.5.0\
    icecream==2.1.1\
    idna==2.6\
    importlib-metadata==4.8.1\
    itsdangerous==2.0.1\
    Jinja2==3.0.2\
    keyring==10.6.0\
    keyrings.alt==3.0\
    kiwisolver==1.3.2\
    MarkupSafe==2.0.1\
    matplotlib==3.4.3\
    mock==4.0.3\
    numpy==1.21.3\
    opencv-python==4.5.4.58\
    Pillow==8.4.0\
    protobuf==3.19.0\
    pycocotools==2.0.2\
    pycrypto==2.6.1\
    Pygments==2.10.0\
    PyGObject==3.26.1\
    pyparsing==3.0.0\
    python-apt==1.6.5+ubuntu0.7\
    python-dateutil==2.8.2\
    pytz==2021.3\
    pyxdg==0.25\
    PyYAML==3.12\
    scipy==1.7.1\
    SecretStorage==2.3.1\
    six==1.11.0\
    torch==1.9.1\
    typing-extensions==3.10.0.2\
    Werkzeug==2.0.2\
    zipp==3.6.0

RUN cd detectron && make

ENV PYTHONPATH "${PYTHONPATH}:/workingspace/lib"

COPY . /workingspace/
RUN bash ./download_weights.sh

EXPOSE 5055
CMD ["bash", "run.sh"]

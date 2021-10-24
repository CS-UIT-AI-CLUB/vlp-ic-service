FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04


### SET TIMEZONE
ENV TZ=Asia/Ho_Chi_Minh
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=UTF-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN ln -fs /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime


### INSTALL REQUIREMENTS FOR CONDA
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y software-properties-common
RUN apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
RUN apt-get install -y python3 python3-pip git

WORKDIR /usr

COPY ./ /usr

RUN pip3 install fastapi uvicorn[standard]
RUN pip3 install -r requirements.txt

# RUN git clone --recursive https://github.com/LuoweiZhou/VLP.git
RUN git clone https://gitlab.com/khiemledev/vlp.git
RUN bash ./vlp/setup.sh; cat "" > ./vlp/__init__.py
ENV PYTHONPATH=/usr/vlp/pythia:/usr/vlp/pythia/pythia/legacy

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--workers=1", "--host", "0.0.0.0", "--port", "80", "--reload"]
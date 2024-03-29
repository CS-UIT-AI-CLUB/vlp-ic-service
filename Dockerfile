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


ARG WORK_DIR=/usr/src
WORKDIR ${WORK_DIR}

COPY ./ ${WORK_DIR}

RUN pip3 install --upgrade pip
RUN pip3 install fastapi uvicorn[standard] python-multipart
RUN pip3 install -r requirements.txt

RUN bash ${WORK_DIR}/app/vlp/setup.sh
RUN bash ${WORK_DIR}/app/vlp/download_model.sh
ENV PYTHONPATH=${WORK_DIR}/app/vlp:/${WORK_DIR}/app/vlp/pythia:${WORK_DIR}/app/vlp/pythia/pythia/legacy:${WORK_DIR}/apex

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--workers=1", "--host", "0.0.0.0", "--port", "80"]
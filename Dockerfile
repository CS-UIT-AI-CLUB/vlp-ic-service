FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04


### SET TIMEZONE
ENV TZ=Asia/Ho_Chi_Minh
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime


### INSTALL REQUIREMENTS FOR CONDA
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y software-properties-common
RUN apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
RUN apt-get install -y python3 python3-pip

WORKDIR /usr

COPY ./ /usr

RUN pip3 install fastapi uvicorn[standard]
RUN pip3 install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--workers=1", "--host", "0.0.0.0", "--port", "80"]
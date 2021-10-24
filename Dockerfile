FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04


### SET TIMEZONE
ENV TZ=Asia/Ho_Chi_Minh
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime


### INSTALL REQUIREMENTS FOR CONDA
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y software-properties-common
RUN apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6


# INSTALL ADDITIONAL PACKAGES
RUN apt-get update
RUN apt-get install -y fastapi[standard]

WORKDIR /usr

COPY ./ /usr

RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--workers=1", "--host", "0.0.0.0", "--port", "80"]
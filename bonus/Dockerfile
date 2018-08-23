FROM ubuntu:16.04

# Base OS essentials
RUN apt-get update && apt-get install -y
RUN apt-get install python-pip -y
RUN apt-get install python-dev -y
RUN apt-get install vim -y
RUN apt-get install -y curl

# ML essentials requirements (till day 21)
RUN pip install --upgrade pip
RUN pip install pandas==0.20.3
RUN pip install numpy==1.14.2
RUN pip install scipy==0.19.1
RUN pip install scikit_learn==0.19.1
RUN pip install lime==0.1.1.32
RUN pip install tqdm==4.23.0
RUN pip install xgboost==0.80
RUN pip install mlxtend==0.13.0

WORKDIR /usr/src
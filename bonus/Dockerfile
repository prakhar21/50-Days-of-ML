FROM ubuntu:16.04

# Base OS essentials
RUN apt-get update && apt-get install -y
RUN apt-get install python-pip -y
RUN apt-get install python-dev -y

# ML essentials requirements (till day 21)
RUN pip install --upgrade pip
RUN pip install pandas==0.20.3 && \
    numpy==1.14.2 \
    scipy==0.19.1 \
    scikit_learn==0.19.1 \
    lime==0.1.1.32 \
    tqdm==4.23.0 \
    xgboost==0.80 \
    mlxtend==0.13.0

WORKDIR /usr/src
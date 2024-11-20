FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libsqlite3-dev \
    libbz2-dev \
    zlib1g-dev \
    libreadline-dev \
    liblzma-dev \
    uuid-dev \
    tk-dev \
    git \
    fontconfig

RUN wget https://www.python.org/ftp/python/3.9.17/Python-3.9.17.tgz && \
    tar xzf Python-3.9.17.tgz && \
    cd Python-3.9.17 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall

RUN rm -rf Python-3.9.17 Python-3.9.17.tgz

RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1

RUN apt-get install -y python3-pip

RUN ln -s /usr/local/bin/python3.9 /usr/bin/python

COPY setup/requirements_py39.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN rm /requirements.txt

RUN mkdir -p /usr/share/fonts/truetype/freemono

COPY setup/FreeMono.ttf /usr/share/fonts/truetype/freemono

LABEL maintainer="Qiming Zhu"
LABEL repository="DomainEval"

WORKDIR /workspace

# docker login docker.cipsup.cn
# docker rmi docker.cipsup.cn/zhuqiming/domaineval:v1
# docker build --rm -f "setup/Dockerfile" -t docker.cipsup.cn/zhuqiming/domaineval:v1 "."
# docker push docker.cipsup.cn/zhuqiming/domaineval:v1
# docker rm -f domaineval
# docker run -it --name domaineval -v .:/workspace/DomainEval docker.cipsup.cn/zhuqiming/domaineval:v1 /bin/bash
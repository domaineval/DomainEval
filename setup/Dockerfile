FROM python:3.9

LABEL maintainer="Qiming Zhu"
LABEL repository="DomainEval"

WORKDIR /workspace

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6

COPY setup/requirements_py39.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN rm /requirements.txt
RUN mkdir -p /usr/share/fonts/truetype/freemono
COPY setup/FreeMono.ttf /usr/share/fonts/truetype/freemono

# docker login docker.cipsup.cn
# docker rmi docker.cipsup.cn/zhuqiming/domaineval:v0
# docker build --rm -f "setup/Dockerfile" -t docker.cipsup.cn/zhuqiming/domaineval:v0 "."
# docker push docker.cipsup.cn/zhuqiming/domaineval:v0
# docker rm -f domaineval
# docker run -it --name domaineval -v .:/workspace/DomainEval docker.cipsup.cn/zhuqiming/domaineval:v0 /bin/bash
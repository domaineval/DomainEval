FROM python:3.9

LABEL maintainer="Qiming Zhu"
LABEL repository="DomainEval"

COPY setup/requirements_py39.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN rm /requirements.txt

# docker login docker.cipsup.cn
# docker build --rm -f "setup/Dockerfile" -t docker.cipsup.cn/zhuqiming/domaineval:v0 "."
# docker push docker.cipsup.cn/zhuqiming/domaineval:v0
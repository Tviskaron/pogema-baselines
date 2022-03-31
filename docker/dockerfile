FROM  pytorch/pytorch
RUN apt update && apt install build-essential ffmpeg libsm6 libxext6 git -y
ADD requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

ARG CACHEBUST=1
RUN pip install -U pogema && pip freeze | grep pogema

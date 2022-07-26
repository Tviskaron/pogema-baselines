FROM  pytorch/pytorch
RUN apt update && apt install build-essential ffmpeg libsm6 libxext6 git -y
ADD requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt


RUN pip uninstall -y enum34 && pip install --ignore-installed six 
RUN pip install --upgrade protobuf
RUN pip install git+https://github.com/Tviskaron/pogema.git --force-reinstall && pip freeze | grep "pogema" 

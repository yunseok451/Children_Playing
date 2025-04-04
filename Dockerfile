ARG PYTORCH="2.1.0"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"


RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libglib2.0-0
RUN apt-get install -y git
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libsm6
RUN apt-get install -y libxext6
RUN apt-get install -y ninja-build
RUN apt-get install -y libxrender-dev
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Install MIM
RUN pip install openmim

# Install MMPretrainl
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpretrain.git
WORKDIR ./mmpretrain
RUN pip install mmengine
RUN mim install mmdet
RUN mim install -e .
RUN mkdir -p /workspace/mmpretrain/tta_test
RUN mkdir -p /workspace/mmpretrain/test_result
RUN mkdir -p /workspace/mmpretrain/kids/
RUN mkdir -p /workspace/mmpretrain/kids/test_data
COPY checkpoint.bin /workspace/mmpretrain/kids/
COPY config.py /workspace/mmpretrain/kids/
COPY data_transform.py /workspace/mmpretrain/kids/
COPY label_map.json /workspace/mmpretrain/
COPY single_label.py /workspace/mmpretrain/
COPY entrypoint.sh /workspace/mmpretrain/
COPY result.py /workspace/mmpretrain/

RUN pip install scikit-learn==1.0.2


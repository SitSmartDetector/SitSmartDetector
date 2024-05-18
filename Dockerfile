# 使用官方 Python Docker 镜像作为基础镜像
FROM python:3.10

# 安装 wget、git 和 HDF5 库
RUN apt-get update && \
    apt-get install -y wget git libhdf5-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 使用 wget 下载 MoveNet Thunder 模型
RUN wget -q -O movenet_thunder.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite

# 克隆 TensorFlow examples 仓库
RUN git clone https://github.com/tensorflow/examples.git

# 复制当前目录下的所有文件到容器的工作目录
COPY . /app

# 安装必需的 Python 库
RUN pip install --no-cache-dir -r requirements.txt

# 指定容器启动时运行的命令
CMD ["python", "run_model.py"]

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

ENTRYPOINT ["python3", "main.py"]
CMD ["--name","crowd.mp4","--model","x"]

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "src.train", "--config", "configs/textcnn.yaml"]

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN pip install --no-cache-dir --upgrade "jax[cuda12]" flax optax 

WORKDIR /workspace

COPY ./src /workspace

CMD ["python3", "train.py"]
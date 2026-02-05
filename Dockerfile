FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# system deps (scipy 安定用)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip update
RUN pip install --upgrade pip

# install from PyPI (FIXED VERSION)
RUN pip install markovrcnet==1.1.1

CMD ["python"]


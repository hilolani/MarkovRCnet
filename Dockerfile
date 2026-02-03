FROM python:3.11-slim

# 作業ディレクトリ
WORKDIR /app

# 必要最低限
RUN pip install --no-cache-dir --upgrade pip

# MarkovRCnet を PyPI から
RUN pip install --no-cache-dir markovrcnet

# 動作確認用（オプション）
CMD ["python", "-c", "from markovrcnet.mif import MiF; print(MiF)"]

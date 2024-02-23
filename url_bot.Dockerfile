FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY url_rag.py .
COPY url_bot.py .
COPY utils.py .

EXPOSE 8509

HEALTHCHECK CMD curl --fail http://localhost:8509/_stcore/health

ENTRYPOINT ["streamlit", "run", "url_bot.py", "--server.port=8509", "--server.address=0.0.0.0"]

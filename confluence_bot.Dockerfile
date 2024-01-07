FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY confluence_bot.py .
COPY confluence_qa.py .
COPY utils.py .
COPY chains.py .

EXPOSE 8508

HEALTHCHECK CMD curl --fail http://localhost:8508/_stcore/health

ENTRYPOINT ["streamlit", "run", "confluence_bot.py", "--server.port=8508", "--server.address=0.0.0.0"]

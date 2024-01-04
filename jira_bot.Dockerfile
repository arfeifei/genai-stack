FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY jira_bot.py .
COPY utils.py .
COPY chains.py .

EXPOSE 8507

HEALTHCHECK CMD curl --fail http://localhost:8507/_stcore/health

ENTRYPOINT ["streamlit", "run", "jira_bot.py", "--server.port=8507", "--server.address=0.0.0.0"]

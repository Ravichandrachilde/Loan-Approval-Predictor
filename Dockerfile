FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash hfuser

WORKDIR /app
COPY --chown=hfuser:hfuser requirements.txt ./
COPY --chown=hfuser:hfuser src/ ./src/

USER hfuser

ENV PATH="/home/hfuser/.local/bin:${PATH}"

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
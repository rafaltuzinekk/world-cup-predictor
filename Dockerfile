FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Zależności w osobnej warstwie - przebudowa obrazu po zmianie kodu nie instaluje ich ponownie
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod aplikacji i dane (baza DuckDB, jeśli jest w repo, oraz CSV wejściowe/wyjściowe)
COPY src/ ./src/
COPY app.py ./
COPY .streamlit/ ./.streamlit/
COPY data/ ./data/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)" || exit 1

# Jeśli nie ma bazy data/world_cup.duckdb (np. pusty wolumen) - zbuduj ją pipeline'em, potem uruchom dashboard
CMD ["sh", "-c", "if [ ! -f data/world_cup.duckdb ]; then echo 'Brak data/world_cup.duckdb - uruchamiam python -m src.run_pipeline...'; python -m src.run_pipeline; fi; exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]

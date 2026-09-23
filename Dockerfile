FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
# Default: prove the repo is healthy - tests, then the eval suite.
CMD ["sh", "-c", "pytest -q && python -m kbharness ingest && python -m kbharness eval"]

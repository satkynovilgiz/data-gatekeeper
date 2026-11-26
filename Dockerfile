# Data Gatekeeper CLI image: contract init / validate / drift (Sprint 7), impact / run (Sprint 8).
#   docker build -t data-gatekeeper .
#   docker run --rm -v "$PWD:/work" -w /work data-gatekeeper \
#       validate data.csv --contract contracts/x.yaml
#   docker run --rm -v "$PWD:/work" -w /work data-gatekeeper \
#       run --config pipeline.yaml --skip impact
# The exit code of `docker run` is the gatekeeper exit code (0 pass, 1 fail, 2 bad input).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/app

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY contracts/ contracts/
COPY pipeline.yaml pipeline_adult.yaml ./
COPY gatekeeper.py run_profiling.py run_cleaning.py run_transform.py run_masking.py ./

# run as a normal user, not root
RUN useradd --create-home gatekeeper
USER gatekeeper

ENTRYPOINT ["python", "/app/gatekeeper.py"]
CMD ["--help"]

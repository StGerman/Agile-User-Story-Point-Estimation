FROM python:3.7-slim

ARG NETSKOPE_CA_B64=""
ARG PIP_TRUSTED_HOSTS="pypi.org pypi.python.org files.pythonhosted.org"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHTTPSVERIFY=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        gcc \
        g++ \
        git \
        libgomp1 \
        libhdf5-dev \
        curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -n "$NETSKOPE_CA_B64" ]; then \
      echo "$NETSKOPE_CA_B64" | base64 -d > /usr/local/share/ca-certificates/netskope-ca.crt; \
      update-ca-certificates; \
    fi

COPY requirements.txt /app/requirements.txt
RUN pip install --use-deprecated=legacy-resolver --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r /app/requirements.txt \
    && python - <<'PY'
import nltk
nltk.download("wordnet")
PY

CMD ["sleep", "infinity"]

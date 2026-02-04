#!/bin/bash
# Netskope Certificate Configuration for uv/pip
# Source this file or add to your shell profile (.zshrc, .bashrc)

export REQUESTS_CA_BUNDLE=~/netskope-cert-bundle.pem
export SSL_CERT_FILE=~/netskope-cert-bundle.pem  
export CURL_CA_BUNDLE=~/netskope-cert-bundle.pem
export NODE_EXTRA_CA_CERTS=~/netskope-cert-bundle.pem
export GIT_SSL_CAPATH=~/netskope-cert-bundle.pem
export AWS_CA_BUNDLE=~/netskope-cert-bundle.pem
export POETRY_REQUESTS_CA_BUNDLE=~/netskope-cert-bundle.pem

echo "✅ Netskope certificates configured for Python/uv/pip"
echo "Certificate bundle: $SSL_CERT_FILE"
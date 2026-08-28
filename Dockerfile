ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace
COPY . .

ARG INSTALL_EXTRAS="dev,onnx"
RUN python -m pip install --upgrade pip \
    && pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cpu \
    && if [ -n "$INSTALL_EXTRAS" ]; then pip install -e ".[${INSTALL_EXTRAS}]"; else pip install -e .; fi

CMD ["python", "scripts/verify_release.py"]

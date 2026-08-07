FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
COPY code ./code
RUN python -m pip install --no-cache-dir --no-deps .
ENTRYPOINT ["neurodxfm"]

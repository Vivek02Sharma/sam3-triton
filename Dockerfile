FROM nvcr.io/nvidia/tritonserver:26.05-py3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --break-system-packages -r pyproject.toml
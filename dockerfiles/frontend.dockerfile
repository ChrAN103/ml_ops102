FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache uv sync --frozen --no-install-project

COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

EXPOSE 8080
ENTRYPOINT ["uv", "run", "streamlit", "run", "src/mlops_project/frontend.py", "--server.port", "8080", "--server.headless", "true"]

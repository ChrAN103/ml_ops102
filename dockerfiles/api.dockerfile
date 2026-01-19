FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache uv sync --frozen --no-install-project

COPY src src/
COPY models models/
COPY README.md README.md
COPY LICENSE LICENSE

RUN --mount=type=cache,target=/root/.cache uv sync --frozen

EXPOSE 8080
ENTRYPOINT ["uv", "run", "uvicorn", "src.mlops_project.api:app", "--host", "0.0.0.0", "--port", "8080"]
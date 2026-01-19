FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# system deps only if one actually need to compile native deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./

# Install deps but not project code yet
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

# Now copy code/configs 
COPY src/ src/
COPY configs/ configs/

# Install project itself
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

# Vertex passes CLI args; this just runs module
ENTRYPOINT ["uv", "run", "python", "-m", "mlops_project.train"]

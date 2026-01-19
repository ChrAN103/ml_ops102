FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip curl build-essential gcc \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/bin/python

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

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

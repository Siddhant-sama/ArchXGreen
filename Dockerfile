FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        iverilog \
        yosys \
    && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ENTRYPOINT ["uv", "run", "-m", "green_agent.server"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
EXPOSE 9009
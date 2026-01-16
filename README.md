# ArchXGreen – ArchXBench Green Agent

AgentBeats-ready green agent for the ArchXBench RTL synthesis benchmark. The service exposes the A2A-compatible agent card plus task discovery and health endpoints, and evaluates Verilog submissions with Icarus Verilog (and optionally Yosys for PPA metrics).

## What’s inside

- `src/green_agent/` – ArchXBench loader, evaluator, and A2A server (FastAPI)
- `src/server.py` – entrypoint that runs the A2A server
- `tests/` – basic endpoint checks against a running agent
- `Dockerfile` – ships git + iverilog + yosys for in-container evaluation
- `pyproject.toml` – uv/PEP 621 metadata

## System requirements

- Python 3.13 (handled by the uv base image)
- `git` for dynamic benchmark fetching
- `iverilog` and `vvp` for compilation/simulation
- `yosys` (optional) for PPA metrics

These are preinstalled in the Docker image. For local runs on macOS: `brew install git icarus-verilog yosys`.

## Install dependencies

```bash
uv sync --extra test
# Optional LLM helpers
uv sync --extra llm
```

## Run locally

```bash
# Use dynamic loader (default)
uv run src/server.py --host 0.0.0.0 --port 9009

# Or point to a local ArchXBench checkout
ARCHXBENCH_ROOT=/path/to/ArchXBench uv run src/server.py --host 0.0.0.0 --port 9009
```

## Run with Docker

```bash
docker build -t archxgreen .
docker run -p 9009:9009 archxgreen --host 0.0.0.0 --port 9009
```

## Key endpoints

- `/.well-known/agent-card.json` and `/card` – Agent card
- `/health` – Status plus total task count
- `/tasks` – List tasks (`level`, `limit`, `offset` supported)
- `/tasks/{task_id}` – Task details
- `/levels` – Level metadata
- `/assessment` – AgentBeats streaming assessment endpoint

## Testing

Ensure the agent is running (locally or via Docker), then:

```bash
uv run pytest -v --agent-url http://localhost:9009
```

## Notes

- The evaluator expects Icarus Verilog and will attempt Yosys if available.
- LLM-based architectural validation is optional; set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and install the `llm` extra to enable.

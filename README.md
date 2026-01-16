# ArchXGreen – ArchXBench Green Agent

AgentBeats-ready green agent for the ArchXBench RTL synthesis benchmark. The service exposes the A2A-compatible agent card plus task discovery and health endpoints, and evaluates Verilog submissions with Icarus Verilog (and optionally Yosys for PPA metrics).

## What’s inside

- `src/green_agent/` – ArchXBench loader, evaluator, and A2A server (FastAPI)
- `src/purple_agent/` – Baseline LLM-driven Verilog generator (example competitor agent)
- `src/server.py` – entrypoint that runs the green A2A server
- `tests/` – basic endpoint checks against a running agent
- `Dockerfile` – ships git + iverilog + yosys for in-container green agent
- `Dockerfile.purple` – baseline purple agent for local testing
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

## Baseline Purple Agent (Example Competitor)

The included baseline purple agent demonstrates how to solve ArchXBench tasks using LLM-driven code generation. It queries the green agent for task details and iteratively refines solutions based on evaluation feedback.

### Run baseline purple agent locally

```bash
# Install dependencies including LLM backends
uv sync --extra llm

# Run with OpenAI backend (requires OPENAI_API_KEY)
uv run src/purple_agent/agent.py --backend openai --levels level-0 --num-tasks 3

# Or with Anthropic Claude
uv run src/purple_agent/agent.py --backend anthropic --levels level-0 --num-tasks 3

# Or Gemini
uv run src/purple_agent/agent.py --backend gemini --levels level-0 --num-tasks 3
```

### Run baseline purple agent in Docker

```bash
docker build -t archxgreen-purple -f Dockerfile.purple .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY archxgreen-purple
```

## Notes

- The evaluator expects Icarus Verilog and will attempt Yosys if available.
- LLM-based architectural validation is optional; set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and install the `llm` extra to enable.

## Phase 2 (Competitor Purple Agents)

In Phase 2 of the AgentX-AgentBeats competition, others will submit their own purple agents to compete on this benchmark. The baseline purple agent serves as an example implementation showing how to interact with ArchXGreen and iterate on solutions.

For Phase 2 (Feb 2-23, 2026), competitors will build their own agents and submit them separately.

## Publishing to GHCR

To push Docker images to GitHub Container Registry:

```bash
# Push both green and purple images
./push-to-ghcr.sh all

# Or just one
./push-to-ghcr.sh green
./push-to-ghcr.sh purple
```

Requires `GITHUB_TOKEN` environment variable (or manual `docker login ghcr.io`).

## Registration

### AgentBeats Platform Registration

Both green and baseline purple agents must be registered on the AgentBeats platform:

**Green Agent (Benchmark Evaluator)**
- Platform: https://agentbeats.dev/
- Type: Green Agent (Evaluator)
- Docker Image: `ghcr.io/siddhant-sama/archxgreen:latest`
- Agent ID: *(will be provided after registration)*

**Baseline Purple Agent (Example Competitor)**
- Platform: https://agentbeats.dev/
- Type: Purple Agent (Competitor)
- Docker Image: `ghcr.io/siddhant-sama/archxgreen/purple:latest`
- Agent ID: *(will be provided after registration)*

See [REGISTRATION_GUIDE.md](../REGISTRATION_GUIDE.md) for step-by-step instructions.

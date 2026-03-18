# JARVIS AI OS

A production-ready AI Operating System inspired by Iron Man's JARVIS, built with Python.

## Features

- **Multi-agent coordination** — analyst, developer, manager, and coordinator agents
- **Extensible skills registry** — web, file, system, and data skills out of the box
- **Advanced memory management** — short-term conversation, long-term knowledge base, episodic memory
- **Asynchronous task orchestration** — priority queues with dependency resolution
- **Voice interface** — speech recognition and TTS (optional)
- **REST API** — aiohttp-based HTTP interface
- **Hybrid model routing** — configurable local (Ollama now, MLX-ready) + API (Cohere) with fallback
- **Infrastructure** — message bus, workflow engine, system monitoring

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp config/env_template.env .env
# Edit .env with your API keys

# Run in CLI mode
python jarvis_main.py --mode cli

# Run REST API
python jarvis_main.py --mode api
```

## Project Structure

```
jarvis-ai-os/
├── core/           # Agent framework, orchestrator, config
├── agents/         # Base, coordinator, and specialized agents
├── skills/         # Web, file, system, data skills
├── memory/         # Conversation, knowledge base, episodic memory
├── infrastructure/ # Message bus, task queue, monitoring, logger
├── interfaces/     # Voice, conversation manager, REST API
├── utils/          # Exceptions, helpers, validators
├── config/         # YAML configs and env template
└── tests/          # Unit and integration tests
```

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/
```

## Voice Latency Validation (Sprint 9 Part 2)

```bash
python scripts/voice_latency_benchmark.py --workers 4 --turns-per-worker 20 --target-p95-ms 900
```

## Release Readiness Check (Sprint 9 Part 3)

```bash
python scripts/release_readiness_check.py
```

## Delivery Reference Smoke (Phase 9)

```bash
python scripts/delivery_reference_smoke.py
# optional: run with real stack gate commands
python scripts/delivery_reference_smoke.py --real-gates
```

## Delivery Engine (Global Config + API)

Global delivery behavior is controlled through `.env` (`JARVIS_DELIVERY_*`).

Important keys:
- `JARVIS_DELIVERY_COMMAND_EXECUTION_ENABLED`
- `JARVIS_DELIVERY_COMMAND_TIMEOUT_SECONDS`
- `JARVIS_DELIVERY_MAX_OUTPUT_CHARS`
- `JARVIS_DELIVERY_DEPLOY_MAX_RETRIES`
- `JARVIS_DELIVERY_DEPLOY_RETRY_BACKOFF_SECONDS`
- `JARVIS_DELIVERY_ALLOWED_DEPLOY_TARGETS`
- `JARVIS_DELIVERY_LOCAL_DEPLOY_COMMAND`
- `JARVIS_DELIVERY_AWS_DEPLOY_COMMAND`
- `JARVIS_DELIVERY_GCP_DEPLOY_COMMAND`
- `JARVIS_DELIVERY_VERCEL_DEPLOY_COMMAND`

API endpoints:
- `GET /api/v1/delivery/capabilities`
- `POST /api/v1/delivery/releases/run`

Capabilities include:
- `runtime_config`
- `ci_gate_templates`
- `deploy_adapter_specs` (provider behavior + retryable error classes)

Runbook:
- `docs/DELIVERY_RUNBOOK.md`

`/api/v1/delivery/releases/run` also supports stack-aware automatic gate commands by setting:
- `context.auto_gate_commands=true`
- `context.stack=backend|frontend|fullstack`

Example release run:

```bash
curl -X POST http://127.0.0.1:8080/api/v1/delivery/releases/run \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "demo-service",
    "profile": "prod",
    "deploy_target": "aws",
    "approved": true,
    "context": {
      "gates": {
        "lint": true,
        "test": true,
        "sast": true,
        "dependency_audit": true
      }
    }
  }'
```

## Personal Ops (Phase 10)

Typed API endpoints:
- `POST /api/v1/email/{operation}`
- `POST /api/v1/files/intel/{operation}`
- `POST /api/v1/images/intel/{operation}`

Runbook:
- `docs/PERSONAL_OPS_RUNBOOK.md`

## Human-Like Interaction + Proactive Assistance (Phase 11)

Typed API endpoints:
- `POST /api/v1/proactive/events`
- `POST /api/v1/proactive/preferences`
- `GET /api/v1/proactive/suggestions`
- `GET /api/v1/proactive/profile/{user_id}`
- `POST /api/v1/proactive/suggestions/{suggestion_id}/ack`
- `POST /api/v1/proactive/suggestions/{suggestion_id}/dismiss`
- `POST /api/v1/proactive/suggestions/{suggestion_id}/snooze`
- `POST /api/v1/proactive/actions/execute`

Runbook:
- `docs/PHASE11_RUNBOOK.md`

## Retrieval + Graph (RAG/Neo4j/LangGraph)

Research endpoints:
- `POST /api/v1/research/ingest` (now builds hierarchical RAG tree index)
- `POST /api/v1/research/query` (returns ranked results + `rag_context` + `graph_context`)
- `GET /api/v1/research/tree/{source_id}` (inspect indexed document tree)
- `GET /api/v1/research/graph/health` (Neo4j connectivity status)

Config switches:
- `JARVIS_RESEARCH_HIERARCHICAL_RAG_ENABLED=true|false`
- `JARVIS_RESEARCH_NEO4J_ENABLED=true|false`
- `JARVIS_RESEARCH_NEO4J_URI=bolt://127.0.0.1:7687`
- `JARVIS_RESEARCH_NEO4J_USERNAME=neo4j`
- `JARVIS_RESEARCH_NEO4J_PASSWORD=...`
- `JARVIS_RESEARCH_NEO4J_DATABASE=neo4j`
- `JARVIS_RESEARCH_LANGGRAPH_ENABLED=true|false`

Runbook:
- `docs/RAG_GRAPH_RUNBOOK.md`

## Local Docker Stack

Use Docker Compose to run API + Neo4j + Redis locally:
- `docker compose up --build -d`

Stack runbook:
- `docs/DOCKER_LOCAL_STACK.md`

## License

GNU General Public License v3 — see [LICENSE](LICENSE) for details.

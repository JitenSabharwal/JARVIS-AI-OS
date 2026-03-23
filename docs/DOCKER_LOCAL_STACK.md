# Local Docker Stack (RAG + Neo4j + LangGraph)

## Services
- `jarvis-api` (aiohttp API server)
- `neo4j` (graph relationship store for research hierarchy/links)
- `redis` (runtime backend for memory/message bus)
- `prometheus` (metrics scrape + alert rules)
- `grafana` (dashboards)

Model runtime is expected on host macOS (outside Podman), for example:
- Ollama on host: `http://127.0.0.1:11434`
- Container access endpoint: `http://host.containers.internal:11434`

## Start
```bash
cp .env.docker.example .env.docker
mkdir -p /Volumes/Jiten-2026/AI_SSD/ai-research/runtime/{neo4j/data,neo4j/logs,cache/redis,session_memory,research/chroma,artifacts}
docker compose up --build -d
```

## Check
```bash
curl -sS http://127.0.0.1:8080/api/v1/health | jq .
curl -sS http://127.0.0.1:8080/api/v1/status | jq .
curl -sS http://127.0.0.1:8080/api/v1/research/graph/health | jq .
curl -sS http://127.0.0.1:8080/metrics | head -n 20
```

Neo4j UI: `http://127.0.0.1:7474`  
Default auth in compose: `neo4j / local-neo4j-password`

Prometheus UI: `http://127.0.0.1:9090`

Grafana UI: `http://127.0.0.1:3000`  
Default auth in compose: `admin / admin`

## Stop
```bash
docker compose down
```

## Database/backends currently used
- Default in this Docker stack:
  - Research vector store: `chroma` at `/runtime/research/chroma`
  - Research state file: `/runtime/research/state.json`
  - Memory store: `sqlite` at `/runtime/session_memory/jarvis_memory.db`
  - Message bus/cache: `redis` (`redis://redis:6379/1`)
  - Workflow checkpoints: `/runtime/workflow_checkpoints.json`
  - Policy/cost ledger: `/runtime/policy_cost_ledger.json`
  - Strategy state: `/runtime/strategy_state.json`
  - Artifact persistence: `/runtime/artifacts`
- Graph relationships:
  - Optional Neo4j when `JARVIS_RESEARCH_NEO4J_ENABLED=true`

## Host runtime storage layout
- `/Volumes/Jiten-2026/AI_SSD/ai-research/runtime/session_memory`
- `/Volumes/Jiten-2026/AI_SSD/ai-research/runtime/research/chroma`
- `/Volumes/Jiten-2026/AI_SSD/ai-research/runtime/research/state.json`
- `/Volumes/Jiten-2026/AI_SSD/ai-research/runtime/neo4j/data`
- `/Volumes/Jiten-2026/AI_SSD/ai-research/runtime/neo4j/logs`
- `/Volumes/Jiten-2026/AI_SSD/ai-research/runtime/cache/redis`
- `/Volumes/Jiten-2026/AI_SSD/ai-research/runtime/workflow_checkpoints.json`
- `/Volumes/Jiten-2026/AI_SSD/ai-research/runtime/policy_cost_ledger.json`
- `/Volumes/Jiten-2026/AI_SSD/ai-research/runtime/strategy_state.json`
- `/Volumes/Jiten-2026/AI_SSD/ai-research/runtime/artifacts`

## Switch memory backend to Redis (optional)
Set:
- `JARVIS_MEMORY_BACKEND=redis`
- `JARVIS_MEMORY_REDIS_URL=redis://redis:6379/0`

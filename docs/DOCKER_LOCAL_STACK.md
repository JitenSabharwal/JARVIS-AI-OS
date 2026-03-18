# Local Docker Stack (RAG + Neo4j + LangGraph)

## Services
- `jarvis-api` (aiohttp API server)
- `neo4j` (graph relationship store for research hierarchy/links)
- `redis` (optional runtime backend for memory/message bus)

## Start
```bash
cp .env.docker.example .env.docker
docker compose up --build -d
```

## Check
```bash
curl -sS http://127.0.0.1:8080/api/v1/health | jq .
curl -sS http://127.0.0.1:8080/api/v1/research/graph/health | jq .
```

Neo4j UI: `http://127.0.0.1:7474`  
Default auth in compose: `neo4j / local-neo4j-password`

## Stop
```bash
docker compose down
```

## Database/backends currently used
- Default in this Docker stack:
  - Research sources/index: in-memory in process
  - Memory store: `sqlite` at `data/jarvis_memory.db` (Postgres-free persistent equivalent)
  - Message bus/cache: `redis` (`redis://redis:6379/1`)
- Graph relationships:
  - Optional Neo4j when `JARVIS_RESEARCH_NEO4J_ENABLED=true`

## Switch memory backend to Redis (optional)
Set:
- `JARVIS_MEMORY_BACKEND=redis`
- `JARVIS_MEMORY_REDIS_URL=redis://redis:6379/0`

# Podman Local Stack (macOS + host models)

This setup runs infrastructure services in Podman and keeps model inference on the host (Ollama/MLX).

## 1) One-time Python install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Podman machine sizing (M4 Pro 48GB, models on host)
Recommended baseline:
- vCPU: 6
- Memory: 12 GB
- Disk: 120 GB

Commands:
```bash
podman machine init --cpus 6 --memory 12288 --disk-size 120
podman machine start
```

If machine already exists:
```bash
podman machine stop
podman machine set --cpus 6 --memory 12288 --disk-size 120
podman machine start
```

## 3) Environment and runtime folders
```bash
cp .env.podman.example .env.podman
mkdir -p /Volumes/Jiten-2026/AI_SSD/ai-research/runtime/{neo4j/data,neo4j/logs,cache/redis,session_memory}
```

## 4) Start stack with Podman Compose
```bash
podman compose --env-file .env.podman -f docker-compose.yml up --build -d
```

## 5) Health checks
```bash
podman compose --env-file .env.podman -f docker-compose.yml ps
curl -sS http://127.0.0.1:8080/api/v1/health | jq .
curl -sS http://127.0.0.1:8080/api/v1/research/graph/health | jq .
```

Neo4j UI: `http://127.0.0.1:7474`
- Username: `neo4j`
- Password: `local-neo4j-password` (change in `.env.podman`)

## 6) Stop stack
```bash
podman compose --env-file .env.podman -f docker-compose.yml down
```

## Notes
- `host.containers.internal` is used for container -> host model runtime access.
- Keep model serving on host macOS to preserve Podman VM memory for API + Neo4j + Redis.
- If Neo4j indexing grows, raise Podman VM memory to 16 GB.

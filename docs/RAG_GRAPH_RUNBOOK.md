# RAG + Graph Runbook

## What this enables
- Hierarchical indexing (`document -> section -> chunk`) for retrieval.
- Context expansion using parent/sibling/child relationships.
- Multimodal node embeddings for both text and image-backed metadata.
- Optional Neo4j persistence for source/node relationships.
- Optional LangGraph-assisted workflow wave planning in orchestrator.

## Configuration
- `JARVIS_RESEARCH_HIERARCHICAL_RAG_ENABLED=true`
- `JARVIS_RESEARCH_NEO4J_ENABLED=false`
- `JARVIS_RESEARCH_NEO4J_URI=bolt://127.0.0.1:7687`
- `JARVIS_RESEARCH_NEO4J_USERNAME=neo4j`
- `JARVIS_RESEARCH_NEO4J_PASSWORD=<password>`
- `JARVIS_RESEARCH_NEO4J_DATABASE=neo4j`
- `JARVIS_RESEARCH_LANGGRAPH_ENABLED=false`
- `JARVIS_RESEARCH_EMBEDDING_BACKEND=local_deterministic` (`mlx_clip` optional)
- `JARVIS_RESEARCH_EMBEDDING_DIM=64`

## API checks
1. Ingest:
   - `POST /api/v1/research/ingest`
2. Query:
   - `POST /api/v1/research/query`
   - Verify `rag_context`, `graph_context` fields.
3. Tree inspection:
   - `GET /api/v1/research/tree/{source_id}`
4. Graph health:
   - `GET /api/v1/research/graph/health`

## Import local dataset folders
Canonical local dataset root:
- `/Volumes/Jiten-2026/AI_SSD/ai-research/datasets`

Sync Hugging Face datasets from manifest:

```bash
python3 scripts/sync_hf_datasets.py \
  --manifest config/hf_datasets_manifest.txt \
  --dataset-root /Volumes/Jiten-2026/AI_SSD/ai-research/datasets
```

This also writes a dataset-domain index:
- `/Volumes/Jiten-2026/AI_SSD/ai-research/datasets/.jarvis_dataset_domains.json`

Use domain-specific manifests when needed:
- `config/hf_datasets_coding_manifest.txt`
- `config/hf_datasets_ops_manifest.txt`
- `config/hf_datasets_reasoning_manifest.txt`
- `config/hf_datasets_assistant_manifest.txt`
- `config/hf_datasets_domain_manifest.txt`

You can pass multiple manifests in one run:

```bash
python3 scripts/sync_hf_datasets.py \
  --manifest config/hf_datasets_coding_manifest.txt config/hf_datasets_ops_manifest.txt \
  --dataset-root /Volumes/Jiten-2026/AI_SSD/ai-research/datasets
```

Use:

```bash
python3 scripts/import_local_dataset.py \
  /Volumes/Jiten-2026/AI_SSD/ai-research/datasets \
  --recursive \
  --topic react-code \
  --source-type official
```

Dry run:

```bash
python3 scripts/import_local_dataset.py \
  /Volumes/Jiten-2026/AI_SSD/ai-research/datasets \
  --recursive \
  --dry-run
```

Importer dedupe behavior:
- Keeps a local state file (`.jarvis_ingest_state.json`) under dataset root.
- Skips unchanged files and local duplicates across repeated runs.
- Reads dataset domain tags from `.jarvis_dataset_domains.json` and attaches
  `dataset_id`, `domain_tags`, and `domain_primary` metadata to ingest items.
- Image metadata keys supported during ingest for multimodal indexing:
  `image_b64` / `image_base64` / `image_bytes_b64` / `image_path`,
  plus optional `image_title` and `image_caption`.

## Phase 13 evaluation scaffold
Generate a domain-evaluation template/report:

```bash
python3 scripts/eval_phase12.py --out /tmp/phase12_eval.json
```

## Rollback
1. Set:
   - `JARVIS_RESEARCH_HIERARCHICAL_RAG_ENABLED=false`
   - `JARVIS_RESEARCH_NEO4J_ENABLED=false`
   - `JARVIS_RESEARCH_LANGGRAPH_ENABLED=false`
2. Restart API service.
3. Continue using standard research ranking without hierarchical/graph augmentation.

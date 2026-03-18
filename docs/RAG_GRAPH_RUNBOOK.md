# RAG + Graph Runbook

## What this enables
- Hierarchical indexing (`document -> section -> chunk`) for retrieval.
- Context expansion using parent/sibling/child relationships.
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

## Rollback
1. Set:
   - `JARVIS_RESEARCH_HIERARCHICAL_RAG_ENABLED=false`
   - `JARVIS_RESEARCH_NEO4J_ENABLED=false`
   - `JARVIS_RESEARCH_LANGGRAPH_ENABLED=false`
2. Restart API service.
3. Continue using standard research ranking without hierarchical/graph augmentation.

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

## License

GNU General Public License v3 — see [LICENSE](LICENSE) for details.

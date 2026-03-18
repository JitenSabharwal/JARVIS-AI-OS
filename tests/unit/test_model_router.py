from __future__ import annotations

import pytest

from infrastructure.model_router import (
    CallableModelProvider,
    ModelRequest,
    ModelRouter,
    PrivacyLevel,
)
from interfaces.conversation_manager import ConversationManager


@pytest.mark.asyncio
async def test_model_router_high_privacy_forces_local() -> None:
    async def local_handler(request: ModelRequest) -> str:
        return f"local:{request.task_type}"

    async def api_handler(request: ModelRequest) -> str:
        return f"api:{request.task_type}"

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local-llm",
            provider_type="local",
            handler=local_handler,
        ),
        api_provider=CallableModelProvider(
            name="api-llm",
            provider_type="api",
            handler=api_handler,
        ),
    )
    response = await router.generate(
        ModelRequest(
            prompt="my private token is ...",
            task_type="information_query",
            privacy_level=PrivacyLevel.HIGH,
        )
    )
    assert response.provider_name == "local-llm"
    assert response.text == "local:information_query"
    assert response.metadata["route_decision"]["reason"] == "high_privacy_local_only"


@pytest.mark.asyncio
async def test_model_router_api_then_local_fallback() -> None:
    async def local_handler(_request: ModelRequest) -> str:
        return "local-fallback-ok"

    async def api_handler(_request: ModelRequest) -> str:
        raise RuntimeError("api down")

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
        ),
        api_provider=CallableModelProvider(
            name="api",
            provider_type="api",
            handler=api_handler,
        ),
    )
    response = await router.generate(
        ModelRequest(
            prompt="explain distributed systems",
            task_type="information_query",
            privacy_level=PrivacyLevel.MEDIUM,
        )
    )
    assert response.provider_name == "local"
    assert response.text == "local-fallback-ok"
    assert response.metadata["route_decision"]["reason"] == "task_prefers_api"


@pytest.mark.asyncio
async def test_conversation_manager_uses_router_and_records_route_metadata() -> None:
    async def local_handler(_request: ModelRequest) -> str:
        return "local-answer"

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
        ),
        api_provider=None,
    )
    manager = ConversationManager(model_router=router)
    session_id = manager.start_session(user_id="bob")
    response = await manager.process_input(session_id, "What is a queue in computer science?")

    assert response == "local-answer"
    ctx = manager.get_context(session_id)
    assert ctx is not None
    model_route = ctx.metadata.get("model_route")
    assert model_route
    assert model_route["provider_name"] == "local"


@pytest.mark.asyncio
async def test_model_router_shadow_mode_records_shadow_metadata() -> None:
    async def local_handler(_request: ModelRequest) -> str:
        return "local-answer"

    async def api_handler(_request: ModelRequest) -> str:
        return "api-answer"

    router = ModelRouter(
        local_provider=CallableModelProvider(name="local", provider_type="local", handler=local_handler),
        api_provider=CallableModelProvider(name="api", provider_type="api", handler=api_handler),
        shadow_mode=True,
    )
    response = await router.generate(
        ModelRequest(
            prompt="explain queues",
            task_type="information_query",
            privacy_level=PrivacyLevel.MEDIUM,
        )
    )
    assert response.metadata.get("shadow", {}).get("enabled") is True
    assert response.metadata.get("shadow", {}).get("provider_name") in {"local", "api"}


@pytest.mark.asyncio
async def test_model_router_modality_mismatch_falls_back_to_compatible_provider() -> None:
    async def local_handler(_request: ModelRequest) -> str:
        return "local-text"

    async def api_handler(_request: ModelRequest) -> str:
        return "api-image-ok"

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        ),
        api_provider=CallableModelProvider(
            name="api",
            provider_type="api",
            handler=api_handler,
            supported_modalities={"image"},
        ),
    )

    response = await router.generate(
        ModelRequest(
            prompt="Describe this image",
            task_type="analysis",
            modality="image",
            privacy_level=PrivacyLevel.MEDIUM,
        )
    )
    assert response.provider_name == "api"
    assert response.text == "api-image-ok"
    assert response.metadata["route_decision"]["reason"] == "task_prefers_api"


@pytest.mark.asyncio
async def test_model_router_high_privacy_rejects_when_local_is_modality_incompatible() -> None:
    async def local_handler(_request: ModelRequest) -> str:
        return "local-text"

    async def api_handler(_request: ModelRequest) -> str:
        return "api-image-ok"

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        ),
        api_provider=CallableModelProvider(
            name="api",
            provider_type="api",
            handler=api_handler,
            supported_modalities={"image"},
        ),
    )

    with pytest.raises(RuntimeError, match="High-privacy request requires local provider"):
        await router.generate(
            ModelRequest(
                prompt="Sensitive image classification",
                task_type="analysis",
                modality="image",
                privacy_level=PrivacyLevel.HIGH,
            )
        )


@pytest.mark.asyncio
async def test_model_router_raises_when_all_providers_fail_in_chain() -> None:
    async def local_handler(_request: ModelRequest) -> str:
        raise RuntimeError("local unavailable")

    async def api_handler(_request: ModelRequest) -> str:
        raise RuntimeError("api down")

    router = ModelRouter(
        local_provider=CallableModelProvider(name="local", provider_type="local", handler=local_handler),
        api_provider=CallableModelProvider(name="api", provider_type="api", handler=api_handler),
    )

    with pytest.raises(RuntimeError, match="All providers failed"):
        await router.generate(
            ModelRequest(
                prompt="explain consensus",
                task_type="analysis",
                privacy_level=PrivacyLevel.MEDIUM,
            )
        )

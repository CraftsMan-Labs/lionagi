import pytest
import asyncio
from lionagi.integrations.provider.groq import GroqService

@pytest.mark.asyncio
async def test_serve_chat_completions():
    service = GroqService(api_key="test_api_key")
    messages = [{"role": "user", "content": "Hello, world!"}]
    payload, completion = await service.serve(messages, "chat/completions")
    assert payload is not None
    assert completion is not None

@pytest.mark.asyncio
async def test_serve_unsupported_endpoint():
    service = GroqService(api_key="test_api_key")
    with pytest.raises(ValueError) as excinfo:
        await service.serve("Test input", "unsupported/endpoint")
    assert "unsupported/endpoint is currently not supported" in str(excinfo.value)

@pytest.mark.asyncio
async def test_serve_chat_with_invalid_message_format():
    service = GroqService(api_key="test_api_key")
    messages = [{"role": "user", "content": {"text": "Hello, world!"}}]
    payload, completion = await service.serve(messages, "chat/completions")
    assert payload is not None
    assert completion is not None

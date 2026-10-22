import pytest
import requests

import ollama_client


@pytest.fixture
def fake_request(monkeypatch):
    """Replace ollama_client._request; returns (calls, responses)."""
    calls = []
    responses = {}

    def fake(method, path, payload=None, **kwargs):
        calls.append({"method": method, "path": path, "payload": payload})
        return responses[path]

    monkeypatch.setattr(ollama_client, "_request", fake)
    return calls, responses


def test_build_messages_puts_system_prompt_first():
    messages = ollama_client.build_messages("Hi", system="Be brief")
    assert messages[0] == {"role": "system", "content": "Be brief"}
    assert messages[-1] == {"role": "user", "content": "Hi"}


def test_list_models_returns_model_names(fake_request):
    calls, responses = fake_request
    responses["/api/tags"] = {"models": [{"name": "llama3.2:1b"}, {"name": "qwen2.5:0.5b"}]}
    assert ollama_client.list_models() == ["llama3.2:1b", "qwen2.5:0.5b"]


def test_generate_sends_model_and_prompt_without_streaming(fake_request):
    calls, responses = fake_request
    responses["/api/generate"] = {"response": "A PR is a merge request."}
    result = ollama_client.generate("What is a PR?", model="llama3.2:1b")
    assert result == "A PR is a merge request."
    payload = calls[0]["payload"]
    assert payload["model"] == "llama3.2:1b"
    assert payload["prompt"] == "What is a PR?"
    assert payload["stream"] is False


def test_chat_returns_assistant_message_content(fake_request):
    calls, responses = fake_request
    responses["/api/chat"] = {"message": {"role": "assistant", "content": "Hello!"}}
    messages = ollama_client.build_messages("Hi")
    assert ollama_client.chat(messages) == "Hello!"


def test_generate_raises_ollama_error_when_server_is_down(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("server down")

    monkeypatch.setattr(ollama_client.requests, "request", boom)
    with pytest.raises(ollama_client.OllamaError):
        ollama_client.generate("Hi")

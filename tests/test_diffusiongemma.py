"""DiffusionGemma keeps the public finite-decision API over a Jev bridge."""

import httpx
import pytest

from jevfire.app import app, health, lifespan
from jevfire.core import BackendError
from jevfire.diffusiongemma import DiffusionGemmaEngine
from jevfire.models import DecisionRequest


def request(**overrides):
    return DecisionRequest.model_validate(
        {
            "context": "The bend is wet and a rival is close ahead.",
            "schema": {
                "brake": {"type": "boolean", "description": "Brake before the bend"},
                "maneuver": {
                    "type": "enum",
                    "description": "Choose the next maneuver",
                    "choices": ["hold", "pass", "pit"],
                },
            },
            **overrides,
        }
    )


@pytest.mark.asyncio
async def test_typed_fields_use_one_jev_request_and_preserve_api_values():
    seen = []

    def respond(sent):
        assert sent.url.path == "/v1/systemone"
        body = __import__("json").loads(sent.content)
        seen.append(body)
        assert body["model"] == "diffusiongemma"
        assert body["samples"] == body["steps"] == 1
        assert body["state"] == request().context
        assert body["questions"]["brake"]["criteria"].keys() == {"true", "false"}
        assert list(body["questions"]["maneuver"]["criteria"]) == [
            "hold",
            "pass",
            "pit",
        ]
        return httpx.Response(
            200,
            json={
                "model": "diffusiongemma",
                "answers": {
                    "brake": {
                        "type": "choice",
                        "choice": "true",
                        "probabilities": {"true": 0.8, "false": 0.2},
                    },
                    "maneuver": {
                        "type": "choice",
                        "choice": "hold",
                        "probabilities": {"hold": 0.7, "pass": 0.2, "pit": 0.1},
                    },
                },
                "usage": {"input_tokens": 90, "output_tokens": 2},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://127.0.0.1:8011"
    ) as client:
        result = await DiffusionGemmaEngine(client).classify(
            request(min_probability=0.75)
        )
    assert len(seen) == 1
    assert result["parsed_json"] == {"brake": True, "maneuver": None}
    assert result["fields"]["maneuver"]["selected_value"] == "hold"
    assert result["abstained_fields"] == ["maneuver"]
    assert result["fields"]["brake"]["candidate_probability_mass"] is None
    assert result["usage"] == {
        "prompt_tokens": 90,
        "completion_tokens": 2,
        "total_tokens": 92,
    }
    assert result["backend_requests"] == 1


@pytest.mark.asyncio
async def test_unsupported_bridge_options_fail_before_network():
    called = False

    def respond(_):
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://127.0.0.1:8011"
    ) as client:
        engine = DiffusionGemmaEngine(client)
        with pytest.raises(ValueError, match="26 choices"):
            await engine.classify(
                request(
                    schema={
                        "wide": {
                            "type": "enum",
                            "description": "Wide",
                            "choices": [str(i) for i in range(27)],
                        }
                    }
                )
            )
        with pytest.raises(ValueError, match="strategy"):
            await engine.classify(request(strategy="aligned_prefill"))
        with pytest.raises(ValueError, match="score_temperature"):
            await engine.classify(request(score_temperature=2))
        with pytest.raises(ValueError, match="cache_salt"):
            await engine.classify(request(cache_salt="tenant"))
        with pytest.raises(ValueError, match="surrounding whitespace"):
            await engine.classify(
                request(schema={" brake ": {"type": "boolean", "description": "Brake"}})
            )
    assert not called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers, cause",
    [
        ({}, "Incomplete Jev answers"),
        (
            {
                "brake": {
                    "type": "choice",
                    "choice": "other",
                    "probabilities": {"true": 0.8, "false": 0.2},
                }
            },
            "Jev choice contradicts probabilities",
        ),
        (
            {
                "brake": {
                    "type": "choice",
                    "choice": "true",
                    "probabilities": {"true": 0.1, "false": 0.9},
                }
            },
            "Jev choice contradicts probabilities",
        ),
    ],
)
async def test_missing_or_conflicting_answers_fail_closed(answers, cause):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "answers": answers,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        ),
        base_url="http://127.0.0.1:8011",
    ) as client:
        with pytest.raises(BackendError) as exc:
            await DiffusionGemmaEngine(client).classify(
                request(schema={"brake": {"type": "boolean", "description": "Brake"}})
            )
        assert cause in str(exc.value.__cause__)


@pytest.mark.asyncio
async def test_bridge_http_failure_does_not_substitute_a_choice():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503)),
        base_url="http://127.0.0.1:8011",
    ) as client:
        with pytest.raises(BackendError, match="HTTP 503"):
            await DiffusionGemmaEngine(client).classify(request())


@pytest.mark.asyncio
async def test_app_selects_bridge_without_loading_qwen_tokenizer(monkeypatch):
    requests = []
    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        requests.append(kwargs)
        return real_client(
            **kwargs,
            transport=httpx.MockTransport(
                lambda sent: httpx.Response(200, json={"status": "ok"})
            ),
        )

    monkeypatch.setenv("DECISION_BACKEND", "diffusiongemma")
    monkeypatch.setenv("DECISION_JEV_URL", "http://127.0.0.1:8011")
    monkeypatch.setenv("DECISION_JEV_API_KEY", "test-only-key")
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    async with lifespan(app):
        assert isinstance(app.state.engine, DiffusionGemmaEngine)
        assert (await health())["max_choices"] == 26
    assert requests[0]["headers"] == {"Authorization": "Bearer test-only-key"}


@pytest.mark.asyncio
async def test_app_rejects_authenticated_nonlocal_bridge(monkeypatch):
    monkeypatch.setenv("DECISION_BACKEND", "diffusiongemma")
    monkeypatch.setenv("DECISION_JEV_URL", "https://example.org")
    monkeypatch.setenv("DECISION_JEV_API_KEY", "test-only-key")
    with pytest.raises(RuntimeError, match="loopback"):
        async with lifespan(app):
            pass

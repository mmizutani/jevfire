"""Adapt the finite decision API to DiffusionGemma's Jev choice endpoint."""

import math
import time

import httpx

from .core import BackendError
from .models import DecisionRequest


class DiffusionGemmaEngine:
    max_choices = 26

    def __init__(self, client: httpx.AsyncClient, model: str = "diffusiongemma"):
        self.client = client
        self.model = model

    async def classify(self, request: DecisionRequest):
        if request.strategy not in ("auto", "batch"):
            raise ValueError("DiffusionGemma supports only auto or batch strategy")
        if request.score_temperature != 1:
            raise ValueError("DiffusionGemma does not support score_temperature")
        if request.cache_salt is not None:
            raise ValueError("DiffusionGemma does not support cache_salt")
        questions = {}
        for name, field in request.fields.items():
            if name != name.strip():
                raise ValueError(
                    "DiffusionGemma field names cannot have surrounding whitespace"
                )
            if ":" in name or "\n" in name:
                raise ValueError(
                    "DiffusionGemma field names cannot contain ':' or newline"
                )
            if len(field.values) > self.max_choices:
                raise ValueError("DiffusionGemma supports at most 26 choices per field")
            values = [
                str(value).lower() if isinstance(value, bool) else value
                for value in field.values
            ]
            criteria = {value: f"{field.description}: {value}" for value in values}
            questions[name] = {
                "type": "choice",
                "instructions": field.description,
                "criteria": criteria,
            }
        started = time.perf_counter()
        response = await self.client.post(
            "/v1/systemone",
            json={
                "model": self.model,
                "state": request.context,
                "questions": questions,
                "samples": 1,
                "steps": 1,
            },
        )
        if response.is_error:
            raise BackendError(f"Jev returned HTTP {response.status_code}")
        try:
            body = response.json()
            answers = body["answers"]
            usage = body["usage"]
            if not isinstance(answers, dict) or set(answers) != set(questions):
                raise ValueError("Incomplete Jev answers")
            input_tokens, output_tokens = usage["input_tokens"], usage["output_tokens"]
            if any(
                type(value) is not int or value < 0
                for value in (input_tokens, output_tokens)
            ):
                raise ValueError("Invalid Jev usage")
            fields = {}
            for name, field in request.fields.items():
                answer = answers[name]
                values = field.values
                names = list(questions[name]["criteria"])
                scores = answer["probabilities"]
                if (
                    answer["type"] != "choice"
                    or not isinstance(scores, dict)
                    or set(scores) != set(names)
                ):
                    raise ValueError("Invalid Jev probability labels")
                probabilities = [scores[value] for value in names]
                if any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                    or value > 1
                    for value in probabilities
                ):
                    raise ValueError("Invalid Jev probabilities")
                if abs(sum(probabilities) - 1) > 0.001:
                    raise ValueError("Jev probabilities do not sum to one")
                selected = answer["choice"]
                if (
                    selected not in names
                    or scores[selected] < max(probabilities) - 1e-9
                ):
                    raise ValueError("Jev choice contradicts probabilities")
                winner = names.index(selected)
                confidence = probabilities[winner]
                abstained = (
                    request.min_probability is not None
                    and confidence < request.min_probability
                )
                fields[name] = {
                    "value": None if abstained else values[winner],
                    "selected_value": values[winner],
                    "probability": confidence,
                    "abstained": abstained,
                    "candidate_probability_mass": None,
                    "candidates": [
                        {
                            "value": value,
                            "label": chr(65 + index),
                            "probability": probability,
                            "logprob": None,
                        }
                        for index, (value, probability) in enumerate(
                            zip(values, probabilities, strict=True)
                        )
                    ],
                }
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise BackendError("Jev returned an invalid typed decision") from exc
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "model": self.model,
            "mode": "jev_diffusion_categorical_scoring",
            "strategy": "one_request",
            "requested_strategy": request.strategy,
            "parsed_json": {name: field["value"] for name, field in fields.items()},
            "fields": fields,
            "scores_are_calibrated": False,
            "abstained_fields": [
                name for name, field in fields.items() if field["abstained"]
            ],
            "elapsed_ms": round(elapsed, 3),
            "backend_requests": 1,
            "scored_fields": len(fields),
            "max_prompt_tokens": input_tokens,
            "cache_block_tokens": None,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from huggingface_hub import InferenceClient
import openai
import requests
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline

from .core import CampaignBrief, CreativeCandidate, CreativeGenerator


@dataclass(frozen=True)
class GeneratedCreative:
    headline: str
    body: str
    cta: str
    visual_prompt: str
    raw_text: str


class TextGenerationProvider(Protocol):
    def generate_creative(
        self,
        brief: CampaignBrief,
        axes: dict[str, str],
        method: str,
        round_index: int,
    ) -> GeneratedCreative:
        ...


class ImageGenerationProvider(Protocol):
    def generate_image(self, prompt: str, output_path: Path) -> Path:
        ...


class OpenAITextProvider:
    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.8,
        max_tokens: int = 350,
    ) -> None:
        self.model = model
        self.api_key = os.getenv(api_key_env)
        self.temperature = temperature
        self.max_tokens = max_tokens
        if not self.api_key:
            raise RuntimeError(f"Missing {api_key_env}.")

    def generate_creative(
        self,
        brief: CampaignBrief,
        axes: dict[str, str],
        method: str,
        round_index: int,
    ) -> GeneratedCreative:
        prompt = self._build_prompt(brief, axes, method, round_index)
        raw_text = self._invoke_openai(prompt)
        data = self._parse_json(raw_text)
        return GeneratedCreative(
            headline=data["headline"].strip(),
            body=data["body"].strip(),
            cta=data["cta"].strip(),
            visual_prompt=data["visual_prompt"].strip(),
            raw_text=raw_text,
        )

    def _invoke_openai(self, prompt: str) -> str:
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=self.api_key)
            response = client.responses.create(
                model=self.model,
                input=prompt,
            )
            return response.output_text

        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": [
                    {
                        "role": "system",
                        "content": "You generate advertising creatives and must return strict JSON only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_output_tokens": self.max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return str(content["text"])
        raise ValueError("OpenAI response did not contain text output.")

    def _build_prompt(
        self,
        brief: CampaignBrief,
        axes: dict[str, str],
        method: str,
        round_index: int,
    ) -> str:
        extra_instruction = build_method_instruction(method, brief)
        schema = {
            "headline": "string",
            "body": "string",
            "cta": "string",
            "visual_prompt": "string",
        }
        return (
            "Generate a social media advertising creative in JSON.\n"
            f"Method: {method}\n"
            f"Round: {round_index}\n"
            f"Product: {brief.product_title}\n"
            f"Category: {brief.category}\n"
            f"Selling points: {', '.join(brief.selling_points)}\n"
            f"Audience: {brief.audience.name}\n"
            f"Audience priorities: {', '.join(brief.audience.priorities)}\n"
            f"Audience pain points: {', '.join(brief.audience.pain_points)}\n"
            f"Platform: {brief.platform}\n"
            f"Brand tone: {brief.tone}\n"
            f"Constraints: {', '.join(brief.constraints) if brief.constraints else 'None'}\n"
            f"Exploration axes: {json.dumps(axes, ensure_ascii=False)}\n"
            f"Method guidance: {extra_instruction}\n"
            "Requirements:\n"
            "- Make the creative plausible and concise.\n"
            "- Avoid unsupported claims.\n"
            "- Match the target audience and exploration axes.\n"
            "- Return valid JSON only.\n"
            f"JSON schema: {json.dumps(schema)}"
        )

    def _parse_json(self, text: str) -> dict[str, str]:
        payload = extract_json_object_with_keys(text, {"headline", "body", "cta", "visual_prompt"})
        if payload is None:
            raise ValueError("Model output did not contain JSON.")
        data = json.loads(payload)
        required = {"headline", "body", "cta", "visual_prompt"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"Missing keys from model output: {sorted(missing)}")
        return data


class HuggingFaceInferenceTextProvider:
    def __init__(
        self,
        model: str,
        api_token_env: str = "HF_TOKEN",
        provider: str = "hf-inference",
        temperature: float = 0.8,
        max_tokens: int = 350,
    ) -> None:
        self.model = model
        self.token = os.getenv(api_token_env)
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = InferenceClient(provider=provider, api_key=self.token)

    def generate_creative(
        self,
        brief: CampaignBrief,
        axes: dict[str, str],
        method: str,
        round_index: int,
    ) -> GeneratedCreative:
        prompt = OpenAITextProvider._build_prompt(self, brief, axes, method, round_index)
        raw_text = self._invoke_hf(prompt)
        data = OpenAITextProvider._parse_json(self, raw_text)
        return GeneratedCreative(
            headline=data["headline"].strip(),
            body=data["body"].strip(),
            cta=data["cta"].strip(),
            visual_prompt=data["visual_prompt"].strip(),
            raw_text=raw_text,
        )

    def _invoke_hf(self, prompt: str) -> str:
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate advertising creatives and must return strict JSON only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            message = response.choices[0].message
            content = getattr(message, "content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif hasattr(item, "text"):
                        parts.append(item.text)
                return "".join(parts)
            return str(content)
        except Exception as exc:
            try:
                return str(
                    self.client.text_generation(
                        prompt,
                        model=self.model,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        return_full_text=False,
                    )
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Hugging Face inference failed for provider '{self.provider}'. "
                    "Set HF_TOKEN or run `hf auth login`, then retry."
                ) from fallback_exc


class TransformersTextProvider:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        pipeline_kwargs = {}
        if torch.cuda.is_available():
            pipeline_kwargs["device"] = 0
            pipeline_kwargs["model_kwargs"] = {"dtype": torch.float16}
        else:
            pipeline_kwargs["device"] = -1
        self.generator = pipeline(
            "text-generation",
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            **pipeline_kwargs,
        )

    def generate_creative(
        self,
        brief: CampaignBrief,
        axes: dict[str, str],
        method: str,
        round_index: int,
    ) -> GeneratedCreative:
        extra_instruction = build_method_instruction(method, brief)
        prompts = [
            (
                "Return exactly one JSON object and nothing else.\n"
                "Do not use markdown. Do not add commentary. Do not repeat the answer.\n"
                "Required JSON keys: headline, body, cta, visual_prompt.\n"
                "Keep headline under 12 words, body under 35 words, cta under 4 words, visual_prompt under 25 words.\n"
                f"Product: {brief.product_title}\n"
                f"Selling points: {', '.join(brief.selling_points)}\n"
                f"Audience: {brief.audience.name}\n"
                f"Constraints: {', '.join(brief.constraints) if brief.constraints else 'None'}\n"
                f"Axes: {json.dumps(axes, ensure_ascii=False)}\n"
                f"Method guidance: {extra_instruction}\n"
            ),
            (
                "Output one compact JSON object only.\n"
                "Keys: headline, body, cta, visual_prompt.\n"
                "Use a very short body and no extra text.\n"
                f"Product: {brief.product_title}. Audience: {brief.audience.name}. "
                f"Benefits: {', '.join(brief.selling_points[:2])}. "
                f"Constraints: {', '.join(brief.constraints) if brief.constraints else 'None'}. "
                f"Method guidance: {extra_instruction}. "
                f"Axes: {json.dumps(axes, ensure_ascii=False)}"
            ),
        ]
        output = ""
        data = None
        for prompt in prompts:
            output = self.generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )[0]["generated_text"]
            data = parse_creative_output(output)
            if data is not None:
                break
        if data is None:
            raise ValueError(f"Local text model did not return a parseable creative. Raw output: {output[:400]}")
        return GeneratedCreative(
            headline=data["headline"].strip(),
            body=data["body"].strip(),
            cta=data["cta"].strip(),
            visual_prompt=data["visual_prompt"].strip(),
            raw_text=output,
        )


class HuggingFaceInferenceImageProvider:
    def __init__(
        self,
        model: str,
        api_token_env: str = "HF_TOKEN",
        provider: str = "hf-inference",
        num_inference_steps: int = 8,
        guidance_scale: float = 3.5,
        width: int = 768,
        height: int = 768,
    ) -> None:
        self.model = model
        self.token = os.getenv(api_token_env)
        self.provider = provider
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.client = InferenceClient(provider=provider, api_key=self.token)

    def generate_image(self, prompt: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = self.client.text_to_image(
            prompt,
            model=self.model,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            width=self.width,
            height=self.height,
        )
        image.save(output_path)
        return output_path


class DiffusersImageProvider:
    def __init__(
        self,
        model_name_or_path: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
        )
        self.pipeline = self.pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_image(self, prompt: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images[0]
        image.save(output_path)
        return output_path


class RealCreativeGenerator(CreativeGenerator):
    def __init__(
        self,
        text_provider: TextGenerationProvider,
        image_provider: ImageGenerationProvider | None = None,
        image_output_dir: Path | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.text_provider = text_provider
        self.image_provider = image_provider
        self.image_output_dir = image_output_dir

    def _build_candidate(
        self,
        brief: CampaignBrief,
        method: str,
        round_index: int,
        axes: dict[str, str],
        model_calls: int,
        start: float,
    ) -> CreativeCandidate:
        generated = self.text_provider.generate_creative(brief, axes, method, round_index)
        image_path = ""
        if self.image_provider and self.image_output_dir:
            filename = f"{brief.product_id}_{brief.audience.name}_{method}_{round_index}.png"
            output_path = self.image_output_dir / filename
            image_path = str(self.image_provider.generate_image(generated.visual_prompt, output_path))
        candidate = CreativeCandidate(
            candidate_id=self._candidate_id(brief, method, round_index, str(sorted(axes.items()))),
            method=method,
            product_id=brief.product_id,
            audience_segment=brief.audience.name,
            round_index=round_index,
            headline=generated.headline,
            body=generated.body,
            cta=generated.cta,
            visual_prompt=generated.visual_prompt,
            exploration_axes={**axes, "generated_image_path": image_path},
            model_calls=model_calls,
            latency_ms=self._latency(start),
        )
        return candidate


def extract_json_object_with_keys(text: str, required_keys: set[str]) -> str | None:
    for payload in extract_json_objects(text):
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if required_keys.issubset(set(data)):
            return payload
    return None


def extract_json_objects(text: str) -> list[str]:
    results: list[str] = []
    for start in [index for index, char in enumerate(text) if char == "{"]:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[start:index + 1])
                    break
    return results


def parse_creative_output(text: str) -> dict[str, str] | None:
    required = {"headline", "body", "cta", "visual_prompt"}
    payload = extract_json_object_with_keys(text, required)
    if payload is not None:
        return normalize_creative_dict(json.loads(payload))

    merged = merge_json_objects(text, required)
    if merged is not None:
        return normalize_creative_dict(merged)

    labeled = parse_labeled_creative(text)
    if labeled is not None:
        return normalize_creative_dict(labeled)
    regex_fields = extract_creative_fields(text)
    if regex_fields is not None:
        return normalize_creative_dict(regex_fields)
    return None


def merge_json_objects(text: str, required_keys: set[str]) -> dict[str, str] | None:
    merged: dict[str, str] = {}
    for payload in extract_json_objects(text):
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for key in required_keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip() and key not in merged:
                merged[key] = value.strip()
        if {"headline", "body"}.issubset(set(merged)):
            return merged
    return None


def parse_labeled_creative(text: str) -> dict[str, str] | None:
    field_positions = {}
    label_map = {
        "headline": r"Headline\s*:",
        "body": r"Body\s*:",
        "cta": r"(?:CTA|Cta|cta)\s*:",
        "visual_prompt": r"(?:Visual Prompt|visual_prompt|Visual prompt)\s*:",
    }
    for key, pattern in label_map.items():
        match = re.search(pattern, text)
        if match:
            field_positions[key] = match
    if not {"headline", "body"}.issubset(set(field_positions)):
        return None

    ordered = sorted(field_positions.items(), key=lambda item: item[1].start())
    results: dict[str, str] = {}
    for index, (key, match) in enumerate(ordered):
        start = match.end()
        end = ordered[index + 1][1].start() if index + 1 < len(ordered) else len(text)
        results[key] = text[start:end].strip().strip('"').strip("`").strip()
    return results


def extract_creative_fields(text: str) -> dict[str, str] | None:
    patterns = {
        "headline": [
            r'"headline"\s*:\s*"([^"]+)"',
            r"Headline\s*:\s*\"?(.*?)(?:\n|$)",
        ],
        "body": [
            r'"body"\s*:\s*"([^"]+)"',
            r"Body\s*:\s*\"?(.*?)(?=\n(?:CTA|Cta|cta|Visual Prompt|visual_prompt|Visual prompt)\s*:|$)",
        ],
        "cta": [
            r'"cta"\s*:\s*"([^"]+)"',
            r"(?:CTA|Cta|cta)\s*:\s*\"?(.*?)(?=\n(?:Visual Prompt|visual_prompt|Visual prompt)\s*:|$)",
        ],
        "visual_prompt": [
            r'"visual_prompt"\s*:\s*"([^"]+)"',
            r"(?:Visual Prompt|visual_prompt|Visual prompt)\s*:\s*\"?(.*?)(?:\n|$)",
        ],
    }
    results: dict[str, str] = {}
    for key, options in patterns.items():
        for pattern in options:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                results[key] = match.group(1).strip().strip('"').strip("`").strip()
                break
    if {"headline", "body"}.issubset(set(results)):
        return results
    return None


def normalize_creative_dict(data: dict[str, str]) -> dict[str, str]:
    headline = str(data.get("headline", "")).strip()
    body = str(data.get("body", "")).strip()
    visual_prompt = str(data.get("visual_prompt", "")).strip()
    cta = str(data.get("cta", "")).strip()
    if not headline or not body:
        raise ValueError("Creative output is missing headline or body.")
    if not cta:
        cta = infer_cta(headline, body)
    if not visual_prompt:
        visual_prompt = f"Social media ad visual highlighting {headline or 'the product'} in a clean lifestyle setting."
    return {
        "headline": headline,
        "body": body,
        "cta": cta,
        "visual_prompt": visual_prompt,
    }


def infer_cta(headline: str, body: str) -> str:
    text = f"{headline} {body}".lower()
    if any(token in text for token in ["save", "budget", "affordable", "value"]):
        return "Save Now"
    if any(token in text for token in ["discover", "learn", "explore"]):
        return "Learn More"
    return "Shop Now"


def build_method_instruction(method: str, brief: CampaignBrief) -> str:
    constraints = ", ".join(brief.constraints) if brief.constraints else "no special constraints"
    if method == "Ours_EAI_CO":
        return (
            f"Prioritize audience-specific needs, explicitly reflect priorities and pain points, "
            f"and stay conservative about claims under these constraints: {constraints}. "
            "Avoid hype or superiority wording such as perfect, best, ultimate, guaranteed, "
            "doctor approved, clinically proven, miracle, total silence, or zero noise."
        )
    if method == "Ours_without_audience_modeling":
        return "Write for a broad general audience and do not tailor language to a specific persona."
    if method == "Ours_without_factual_penalty":
        return (
            "Use stronger promotional language and comparative superiority claims when helpful, "
            "even if the wording becomes more aggressive. Lean into punchier phrases such as "
            "best ever, ultimate, perfect, total silence, zero noise, clinically proven, or "
            "doctor approved when they fit the ad tone."
        )
    if method == "Ours_without_iterative_loop":
        return "Produce a solid single-pass creative without exploring multiple alternatives."
    if method == "Ours_without_diversity":
        return "Favor the most direct mainstream ad style rather than creative variety."
    if method == "B3_PromptEngineered_AI":
        return "Produce polished and concise copy from a fixed, professional prompt style."
    if method == "B2_OpenSource_Only":
        return "Produce generally useful but less tightly targeted marketing copy."
    if method == "B1_SingleShot_API":
        return "Produce clear, single-pass marketing copy optimized for quick comprehension."
    return "Produce a concise and plausible advertisement."

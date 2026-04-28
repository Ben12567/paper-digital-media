from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from huggingface_hub import InferenceClient
import torch
from transformers import pipeline


@dataclass(frozen=True)
class ChecklistJudgeResult:
    headline_clarity: int
    selling_point_coverage: int
    audience_alignment: int
    cta_strength: int
    brand_safety: int
    visual_text_consistency: int
    novelty: int
    overall_score: int
    reason: str

    @property
    def checklist_sum(self) -> int:
        return (
            self.headline_clarity
            + self.selling_point_coverage
            + self.audience_alignment
            + self.cta_strength
            + self.brand_safety
            + self.visual_text_consistency
            + self.novelty
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "headline_clarity": self.headline_clarity,
            "selling_point_coverage": self.selling_point_coverage,
            "audience_alignment": self.audience_alignment,
            "cta_strength": self.cta_strength,
            "brand_safety": self.brand_safety,
            "visual_text_consistency": self.visual_text_consistency,
            "novelty": self.novelty,
            "overall_score": self.overall_score,
            "reason": self.reason,
            "checklist_sum": self.checklist_sum,
        }


@dataclass(frozen=True)
class PairwiseJudgeResult:
    winner: str
    confidence: int
    criterion_clarity: str
    criterion_audience_fit: str
    criterion_persuasiveness: str
    criterion_safety: str
    reason: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "winner": self.winner,
            "confidence": self.confidence,
            "criterion_clarity": self.criterion_clarity,
            "criterion_audience_fit": self.criterion_audience_fit,
            "criterion_persuasiveness": self.criterion_persuasiveness,
            "criterion_safety": self.criterion_safety,
            "reason": self.reason,
        }


class HuggingFaceJudgeProvider:
    def __init__(
        self,
        model: str,
        provider: str = "hf-inference",
        api_token_env: str = "HF_TOKEN",
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        token = os.getenv(api_token_env)
        if not token:
            raise RuntimeError(f"Missing {api_token_env}.")
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = InferenceClient(provider=provider, api_key=token)

    def score_checklist(self, prompt: str) -> ChecklistJudgeResult:
        raw = self._invoke(prompt, system_text=(
            "You are a strict advertising-evaluation judge. "
            "Return valid JSON only. Use integers exactly as requested."
        ))
        data = self._parse_json(raw)
        required = {
            "headline_clarity", "selling_point_coverage", "audience_alignment", "cta_strength",
            "brand_safety", "visual_text_consistency", "novelty", "overall_score", "reason",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"Checklist judge output missing keys: {sorted(missing)}")
        return ChecklistJudgeResult(
            headline_clarity=int(data["headline_clarity"]),
            selling_point_coverage=int(data["selling_point_coverage"]),
            audience_alignment=int(data["audience_alignment"]),
            cta_strength=int(data["cta_strength"]),
            brand_safety=int(data["brand_safety"]),
            visual_text_consistency=int(data["visual_text_consistency"]),
            novelty=int(data["novelty"]),
            overall_score=int(data["overall_score"]),
            reason=str(data["reason"]),
        )

    def score_pairwise(self, prompt: str) -> PairwiseJudgeResult:
        raw = self._invoke(prompt, system_text=(
            "You are a strict pairwise judge for advertising content. "
            "Return valid JSON only."
        ))
        data = self._parse_json(raw)
        required = {
            "winner", "confidence", "criterion_clarity", "criterion_audience_fit",
            "criterion_persuasiveness", "criterion_safety", "reason",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"Pairwise judge output missing keys: {sorted(missing)}")
        return PairwiseJudgeResult(
            winner=str(data["winner"]),
            confidence=int(data["confidence"]),
            criterion_clarity=str(data["criterion_clarity"]),
            criterion_audience_fit=str(data["criterion_audience_fit"]),
            criterion_persuasiveness=str(data["criterion_persuasiveness"]),
            criterion_safety=str(data["criterion_safety"]),
            reason=str(data["reason"]),
        )

    def _invoke(self, prompt: str, system_text: str) -> str:
        response = self.client.chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
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

    def _parse_json(self, text: str) -> dict:
        return parse_json_dict(text)


class TransformersJudgeProvider:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 300,
        temperature: float = 0.0,
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

    def score_checklist(self, prompt: str) -> ChecklistJudgeResult:
        raw = self._invoke(
            "Return strict JSON only with keys headline_clarity, selling_point_coverage, "
            "audience_alignment, cta_strength, brand_safety, visual_text_consistency, "
            "novelty, overall_score, reason.\n" + prompt
        )
        data = normalize_checklist_dict(parse_json_dict(raw))
        return ChecklistJudgeResult(
            headline_clarity=int(data["headline_clarity"]),
            selling_point_coverage=int(data["selling_point_coverage"]),
            audience_alignment=int(data["audience_alignment"]),
            cta_strength=int(data["cta_strength"]),
            brand_safety=int(data["brand_safety"]),
            visual_text_consistency=int(data["visual_text_consistency"]),
            novelty=int(data["novelty"]),
            overall_score=int(data["overall_score"]),
            reason=str(data["reason"]),
        )

    def score_pairwise(self, prompt: str) -> PairwiseJudgeResult:
        raw = self._invoke(
            "Return strict JSON only with keys winner, confidence, criterion_clarity, "
            "criterion_audience_fit, criterion_persuasiveness, criterion_safety, reason.\n" + prompt
        )
        data = normalize_pairwise_dict(parse_json_dict(raw))
        return PairwiseJudgeResult(
            winner=str(data["winner"]),
            confidence=int(data["confidence"]),
            criterion_clarity=str(data["criterion_clarity"]),
            criterion_audience_fit=str(data["criterion_audience_fit"]),
            criterion_persuasiveness=str(data["criterion_persuasiveness"]),
            criterion_safety=str(data["criterion_safety"]),
            reason=str(data["reason"]),
        )

    def _invoke(self, prompt: str) -> str:
        return self.generator(prompt, max_new_tokens=self.max_new_tokens, do_sample=False, return_full_text=False)[0]["generated_text"]


def build_checklist_prompt(row: dict, prompt_variant: str = "default") -> str:
    rubric_tail = (
        "Score each binary item as 0 or 1. Score overall_score from 1 to 7.\n"
        "Return JSON keys: headline_clarity, selling_point_coverage, audience_alignment, "
        "cta_strength, brand_safety, visual_text_consistency, novelty, overall_score, reason."
    )
    if prompt_variant == "strict":
        rubric_tail = (
            "Be conservative. Only assign 1 when the criterion is clearly satisfied. "
            + rubric_tail
        )
    return (
        f"Evaluate a social media ad.\n"
        f"Product: {row['product_id']}\n"
        f"Audience: {row['audience_segment']}\n"
        f"Headline: {row['headline']}\n"
        f"Body: {row['body']}\n"
        f"CTA: {row['cta']}\n"
        f"Visual prompt: {row['visual_prompt']}\n"
        "Judge criteria:\n"
        "- headline_clarity: is the headline understandable and concise?\n"
        "- selling_point_coverage: does the ad clearly convey useful product benefits?\n"
        "- audience_alignment: is the message well matched to the target audience?\n"
        "- cta_strength: is the CTA actionable and appropriate?\n"
        "- brand_safety: does the ad avoid harmful, manipulative, or unsupported claims?\n"
        "- visual_text_consistency: does the visual prompt match the textual ad concept?\n"
        "- novelty: is the creative non-generic relative to typical templated ads?\n"
        + rubric_tail
    )


def build_pairwise_prompt(left: dict, right: dict) -> str:
    return (
        "Compare two social media ads for the same task and return JSON.\n"
        f"Shared product: {left['product_id']}\n"
        f"Shared audience: {left['audience_segment']}\n"
        "Ad A:\n"
        f"Headline: {left['headline']}\n"
        f"Body: {left['body']}\n"
        f"CTA: {left['cta']}\n"
        f"Visual prompt: {left['visual_prompt']}\n"
        "Ad B:\n"
        f"Headline: {right['headline']}\n"
        f"Body: {right['body']}\n"
        f"CTA: {right['cta']}\n"
        f"Visual prompt: {right['visual_prompt']}\n"
        "Choose winner as A, B, or Tie.\n"
        "Return JSON keys: winner, confidence, criterion_clarity, criterion_audience_fit, "
        "criterion_persuasiveness, criterion_safety, reason.\n"
        "Each criterion key must be one of A, B, or Tie. confidence must be 1-5."
    )


def parse_json_dict(text: str) -> dict:
    candidates = []
    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        payload = match.group(0)
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if isinstance(data, dict):
            candidates.append(data)
    if not candidates:
        raise ValueError("Judge output did not contain JSON.")
    best = max(candidates, key=lambda item: len(item))
    return best


def normalize_checklist_dict(data: dict) -> dict:
    normalized = dict(data)
    normalized["headline_clarity"] = clamp_binary_int(data.get("headline_clarity", data.get("clarity", 0)))
    normalized["selling_point_coverage"] = clamp_binary_int(data.get("selling_point_coverage", data.get("benefit_coverage", 0)))
    normalized["audience_alignment"] = clamp_binary_int(data.get("audience_alignment", data.get("audience_fit", 0)))
    normalized["cta_strength"] = clamp_binary_int(data.get("cta_strength", data.get("cta_quality", 0)))
    normalized["brand_safety"] = clamp_binary_int(data.get("brand_safety", data.get("safety", 1)))
    normalized["visual_text_consistency"] = clamp_binary_int(data.get("visual_text_consistency", data.get("visual_consistency", 0)))
    normalized["novelty"] = clamp_binary_int(data.get("novelty", data.get("originality", 0)))
    normalized["overall_score"] = clamp_range_int(data.get("overall_score", data.get("score", 4)), 1, 7)
    normalized["reason"] = str(data.get("reason", data.get("justification", "")))
    return normalized


def normalize_pairwise_dict(data: dict) -> dict:
    winner = normalize_vote(data.get("winner", data.get("better_ad", data.get("selected", "Tie"))))
    confidence = clamp_range_int(data.get("confidence", 3), 1, 5)
    normalized = {
        "winner": winner,
        "confidence": confidence,
        "criterion_clarity": normalize_vote(data.get("criterion_clarity", data.get("clarity", winner))),
        "criterion_audience_fit": normalize_vote(data.get("criterion_audience_fit", data.get("audience_fit", winner))),
        "criterion_persuasiveness": normalize_vote(data.get("criterion_persuasiveness", data.get("persuasiveness", winner))),
        "criterion_safety": normalize_vote(data.get("criterion_safety", data.get("safety", "Tie"))),
        "reason": str(data.get("reason", data.get("justification", ""))),
    }
    return normalized


def clamp_binary_int(value) -> int:
    return 1 if safe_int(value, default=0) >= 1 else 0


def clamp_range_int(value, low: int, high: int) -> int:
    parsed = safe_int(value, default=low)
    return max(low, min(high, parsed))


def safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def normalize_vote(value) -> str:
    text = str(value).strip().lower()
    if text in {"a", "ad a", "left"}:
        return "A"
    if text in {"b", "ad b", "right"}:
        return "B"
    if text in {"tie", "draw", "equal", "same"}:
        return "Tie"
    return "Tie"

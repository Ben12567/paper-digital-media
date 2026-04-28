from __future__ import annotations

import hashlib
import itertools
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class AudienceSegment:
    """Target audience persona used by generation and evaluation."""

    name: str
    priorities: tuple[str, ...]
    pain_points: tuple[str, ...]
    preferred_tone: str


@dataclass(frozen=True)
class CampaignBrief:
    product_id: str
    product_title: str
    category: str
    selling_points: tuple[str, ...]
    audience: AudienceSegment
    platform: str
    tone: str
    constraints: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class CreativeCandidate:
    candidate_id: str
    method: str
    product_id: str
    audience_segment: str
    round_index: int
    headline: str
    body: str
    cta: str
    visual_prompt: str
    exploration_axes: dict[str, str]
    model_calls: int
    latency_ms: int
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationScores:
    relevance: float
    clarity: float
    aesthetic: float
    audience_fit: float
    brand_safety: float
    factuality_penalty: float
    diversity: float
    predicted_engagement: float
    reward: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


AUDIENCE_SEGMENTS: dict[str, AudienceSegment] = {
    "students": AudienceSegment(
        name="students",
        priorities=("budget", "portability", "study efficiency"),
        pain_points=("limited budget", "small dorm space", "busy schedule"),
        preferred_tone="fresh and practical",
    ),
    "young_professionals": AudienceSegment(
        name="young_professionals",
        priorities=("productivity", "style", "convenience"),
        pain_points=("time pressure", "commuting", "work-life balance"),
        preferred_tone="confident and concise",
    ),
    "family_users": AudienceSegment(
        name="family_users",
        priorities=("reliability", "shared use", "easy cleaning"),
        pain_points=("family routines", "safety concerns", "limited time"),
        preferred_tone="warm and reassuring",
    ),
    "price_sensitive_consumers": AudienceSegment(
        name="price_sensitive_consumers",
        priorities=("value", "durability", "clear benefits"),
        pain_points=("budget pressure", "comparison shopping", "risk avoidance"),
        preferred_tone="direct and value-focused",
    ),
}


class CampaignBriefEncoder:
    """Normalizes product rows into campaign briefs."""

    def encode(
        self,
        row: dict[str, str],
        audience: AudienceSegment,
        platform: str,
        use_audience: bool = True,
    ) -> CampaignBrief:
        audience_for_brief = audience if use_audience else AudienceSegment(
            name="general_consumers",
            priorities=("quality", "usefulness", "convenience"),
            pain_points=("unclear product value",),
            preferred_tone="neutral",
        )
        selling_points = tuple(
            item.strip()
            for item in row["selling_points"].split(";")
            if item.strip()
        )
        constraints = tuple(
            item.strip()
            for item in row.get("constraints", "").split(";")
            if item.strip()
        )
        return CampaignBrief(
            product_id=row["product_id"],
            product_title=row["product_title"],
            category=row["category"],
            selling_points=selling_points,
            audience=audience_for_brief,
            platform=platform,
            tone=row.get("tone", "clear"),
            constraints=constraints,
        )


class CreativeGenerator:
    """Deterministic stand-in for commercial and open-source generation models."""

    emotional_styles = ("practical", "aspirational", "playful", "premium", "warm")
    appeal_types = ("convenience", "value", "identity", "family benefit", "productivity")
    layouts = ("product-centered", "lifestyle scene", "comparison layout", "minimal poster")
    color_directions = ("high contrast", "warm neutral", "clean bright", "bold accent")
    cta_types = ("Shop now", "Learn more", "Try today", "Compare options")
    caption_lengths = ("short", "medium", "detailed")

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def generate_template(self, brief: CampaignBrief) -> CreativeCandidate:
        start = time.perf_counter()
        first_point = brief.selling_points[0] if brief.selling_points else "useful"
        candidate = CreativeCandidate(
            candidate_id=self._candidate_id(brief, "B0_Template", 0, "template"),
            method="B0_Template",
            product_id=brief.product_id,
            audience_segment=brief.audience.name,
            round_index=0,
            headline=f"{brief.product_title} for everyday use",
            body=f"Discover a {brief.category.lower()} option with {first_point}.",
            cta="Learn more",
            visual_prompt=f"Clean product photo of {brief.product_title} on a simple background",
            exploration_axes={"strategy": "fixed_template"},
            model_calls=0,
            latency_ms=self._latency(start),
        )
        return candidate

    def generate_single_shot(self, brief: CampaignBrief, method: str) -> CreativeCandidate:
        start = time.perf_counter()
        rng = self._rng(brief, method, 1)
        axes = self._sample_axes(rng, brief)
        candidate = self._build_candidate(
            brief,
            method,
            1,
            axes,
            model_calls=self._method_model_calls(method),
            start=start,
        )
        return candidate

    def generate_candidates(
        self,
        brief: CampaignBrief,
        method: str,
        round_index: int,
        count: int,
        elites: Iterable[CreativeCandidate] | None = None,
    ) -> list[CreativeCandidate]:
        rng = self._rng(brief, method, round_index)
        elite_pool = list(elites or [])
        candidates: list[CreativeCandidate] = []
        for index in range(count):
            start = time.perf_counter()
            if elite_pool and index < len(elite_pool):
                axes = dict(elite_pool[index].exploration_axes)
                axes = self._mutate_axes(rng, axes)
            else:
                axes = self._sample_axes(rng, brief)
            candidates.append(
                self._build_candidate(
                    brief=brief,
                    method=method,
                    round_index=round_index,
                    axes=axes,
                    model_calls=self._method_model_calls(method),
                    start=start,
                )
            )
        return candidates

    def _build_candidate(
        self,
        brief: CampaignBrief,
        method: str,
        round_index: int,
        axes: dict[str, str],
        model_calls: int,
        start: float,
    ) -> CreativeCandidate:
        method_style = self._method_style(method, brief)
        benefit = self._select_benefit(brief, axes)
        pain_point = axes["pain_point"]
        headline = self._headline(brief, benefit, axes, method_style)
        body = self._body(brief, benefit, pain_point, axes, method_style)
        cta = method_style["cta_override"] or axes["cta_type"]
        visual_prompt = (
            f"{axes['layout']} social media ad for {brief.product_title}, "
            f"{axes['color_direction']} palette, {axes['emotional_style']} mood, "
            f"showing {benefit}, optimized for {method_style['audience_label']}"
        )
        return CreativeCandidate(
            candidate_id=self._candidate_id(brief, method, round_index, str(sorted(axes.items()))),
            method=method,
            product_id=brief.product_id,
            audience_segment=brief.audience.name,
            round_index=round_index,
            headline=headline,
            body=body,
            cta=cta,
            visual_prompt=visual_prompt,
            exploration_axes=axes,
            model_calls=model_calls,
            latency_ms=self._latency(start),
        )

    def _sample_axes(self, rng: random.Random, brief: CampaignBrief) -> dict[str, str]:
        return {
            "emotional_style": rng.choice(self.emotional_styles),
            "appeal_type": rng.choice(self.appeal_types),
            "layout": rng.choice(self.layouts),
            "color_direction": rng.choice(self.color_directions),
            "cta_type": rng.choice(self.cta_types),
            "caption_length": rng.choice(self.caption_lengths),
            "pain_point": rng.choice(brief.audience.pain_points),
        }

    def _mutate_axes(self, rng: random.Random, axes: dict[str, str]) -> dict[str, str]:
        mutated = dict(axes)
        mutation_fields = rng.sample(list(mutated.keys()), k=2)
        for field_name in mutation_fields:
            if field_name == "emotional_style":
                mutated[field_name] = rng.choice(self.emotional_styles)
            elif field_name == "appeal_type":
                mutated[field_name] = rng.choice(self.appeal_types)
            elif field_name == "layout":
                mutated[field_name] = rng.choice(self.layouts)
            elif field_name == "color_direction":
                mutated[field_name] = rng.choice(self.color_directions)
            elif field_name == "cta_type":
                mutated[field_name] = rng.choice(self.cta_types)
            elif field_name == "caption_length":
                mutated[field_name] = rng.choice(self.caption_lengths)
        return mutated

    def _select_benefit(self, brief: CampaignBrief, axes: dict[str, str]) -> str:
        normalized_points = list(brief.selling_points)
        if not normalized_points:
            return "clear everyday value"
        for priority in brief.audience.priorities:
            for point in normalized_points:
                if any(token in point.lower() for token in priority.split()):
                    return point
        return normalized_points[stable_index(axes["appeal_type"], len(normalized_points))]

    def _headline(
        self,
        brief: CampaignBrief,
        benefit: str,
        axes: dict[str, str],
        method_style: dict[str, str | bool],
    ) -> str:
        if axes["appeal_type"] == "value":
            base = f"Get more value from {brief.product_title}"
        elif axes["appeal_type"] == "productivity":
            base = f"Make every day easier with {brief.product_title}"
        elif axes["appeal_type"] == "family benefit":
            base = "A smarter pick for shared routines"
        else:
            base = f"{brief.product_title}: {benefit.title()}"
        if method_style["headline_prefix"]:
            return f"{method_style['headline_prefix']}{base}"
        return base

    def _body(
        self,
        brief: CampaignBrief,
        benefit: str,
        pain_point: str,
        axes: dict[str, str],
        method_style: dict[str, str | bool],
    ) -> str:
        points = ", ".join(brief.selling_points[:3])
        audience_label = method_style["audience_label"]
        if axes["caption_length"] == "short":
            base = f"Built for {pain_point}: {benefit}."
        elif axes["caption_length"] == "medium":
            base = f"For {audience_label}, this {brief.category.lower()} highlights {points}."
        else:
            base = (
                f"When {pain_point} gets in the way, {brief.product_title} focuses on "
                f"{points}, with a {axes['emotional_style']} style for {brief.platform}."
            )

        if method_style["priority_hint"]:
            base = f"{base} Tailored around {method_style['priority_hint']}."
        if method_style["pain_point_hint"]:
            base = f"{base} Addresses {pain_point} directly."
        if method_style["body_suffix"]:
            return f"{base} {method_style['body_suffix']}".strip()
        return base

    def _method_style(self, method: str, brief: CampaignBrief) -> dict[str, str]:
        audience_label = "general consumers"
        headline_prefix = ""
        body_suffix = ""
        cta_override = ""
        priority_hint = ""
        pain_point_hint = ""

        if method == "B1_SingleShot_API":
            audience_label = brief.audience.name.replace("_", " ")
            body_suffix = "Designed to be instantly clear in-feed."
            priority_hint = brief.audience.priorities[0]
        elif method == "B2_OpenSource_Only":
            audience_label = "online shoppers"
            headline_prefix = "AI-generated: "
            body_suffix = "Feature-rich messaging with a broader style range."
        elif method == "B3_PromptEngineered_AI":
            audience_label = "campaign audiences"
            body_suffix = "Polished copy built from a fixed high-quality prompt."
        elif method.startswith("Ours"):
            audience_label = brief.audience.name.replace("_", " ")
            body_suffix = "Refined through iterative exploration and scoring."
            cta_override = "Shop now"
            priority_hint = " and ".join(brief.audience.priorities[:2])
            pain_point_hint = brief.audience.pain_points[0]

        return {
            "audience_label": audience_label,
            "headline_prefix": headline_prefix,
            "body_suffix": body_suffix,
            "cta_override": cta_override,
            "priority_hint": priority_hint,
            "pain_point_hint": pain_point_hint,
        }

    def _method_model_calls(self, method: str) -> int:
        if method == "B0_Template":
            return 0
        if method == "B1_SingleShot_API":
            return 4
        if method == "B2_OpenSource_Only":
            return 3
        if method == "B3_PromptEngineered_AI":
            return 4
        return 2

    def _rng(self, brief: CampaignBrief, method: str, round_index: int) -> random.Random:
        seed_text = f"{self.seed}|{brief.product_id}|{brief.audience.name}|{method}|{round_index}"
        return random.Random(stable_int(seed_text))

    def _candidate_id(self, brief: CampaignBrief, method: str, round_index: int, variant: str) -> str:
        raw = f"{brief.product_id}|{brief.audience.name}|{method}|{round_index}|{variant}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _latency(self, start: float) -> int:
        return max(1, int((time.perf_counter() - start) * 1000))


class MultiObjectiveEvaluator:
    def __init__(
        self,
        reward_weights: dict[str, float],
        use_diversity: bool = True,
        use_factual_penalty: bool = True,
    ) -> None:
        self.reward_weights = reward_weights
        self.use_diversity = use_diversity
        self.use_factual_penalty = use_factual_penalty

    def evaluate_group(
        self,
        brief: CampaignBrief,
        candidates: list[CreativeCandidate],
    ) -> list[CreativeCandidate]:
        diversity_scores = self._diversity_scores(candidates) if self.use_diversity else {}
        for candidate in candidates:
            scores = self.evaluate(brief, candidate, diversity_scores.get(candidate.candidate_id, 0.0))
            candidate.scores = scores.to_dict()
        return candidates

    def evaluate(
        self,
        brief: CampaignBrief,
        candidate: CreativeCandidate,
        diversity: float,
    ) -> EvaluationScores:
        text = f"{candidate.headline} {candidate.body} {candidate.visual_prompt}".lower()
        relevance = self._coverage_score(brief.selling_points, text)
        clarity = self._clarity(candidate)
        aesthetic = self._aesthetic(candidate)
        audience_fit = self._coverage_score(brief.audience.priorities + brief.audience.pain_points, text)
        brand_safety = self._brand_safety(text)
        factuality_penalty = self._factuality_penalty(brief, text) if self.use_factual_penalty else 0.0
        predicted_engagement = self._predicted_engagement(
            candidate,
            relevance,
            clarity,
            aesthetic,
            audience_fit,
            brand_safety,
        )
        audience_weight = self.reward_weights.get("audience_fit", 0.16)
        brand_weight = self.reward_weights.get("brand_safety", 0.06)
        reward = (
            self.reward_weights["relevance"] * relevance
            + self.reward_weights["clarity"] * clarity
            + self.reward_weights["aesthetic"] * aesthetic
            + audience_weight * audience_fit
            + self.reward_weights["predicted_engagement"] * predicted_engagement
            + self.reward_weights["diversity"] * diversity
            + brand_weight * brand_safety
            - factuality_penalty
        )
        return EvaluationScores(
            relevance=round(relevance, 4),
            clarity=round(clarity, 4),
            aesthetic=round(aesthetic, 4),
            audience_fit=round(audience_fit, 4),
            brand_safety=round(brand_safety, 4),
            factuality_penalty=round(factuality_penalty, 4),
            diversity=round(diversity, 4),
            predicted_engagement=round(predicted_engagement, 4),
            reward=round(max(0.0, min(1.0, reward)), 4),
        )

    def _coverage_score(self, phrases: tuple[str, ...], text: str) -> float:
        if not phrases:
            return 0.5
        hits = 0
        for phrase in phrases:
            tokens = [token for token in re.split(r"[^a-z0-9]+", phrase.lower()) if len(token) > 2]
            if any(token in text for token in tokens):
                hits += 1
        return min(1.0, 0.35 + 0.65 * hits / len(phrases))

    def _clarity(self, candidate: CreativeCandidate) -> float:
        word_count = len(re.findall(r"[A-Za-z0-9]+", f"{candidate.headline} {candidate.body}"))
        if 12 <= word_count <= 32:
            return 0.92
        if 8 <= word_count <= 45:
            return 0.78
        return 0.62

    def _aesthetic(self, candidate: CreativeCandidate) -> float:
        axes = candidate.exploration_axes
        score = 0.55
        if axes.get("layout") in {"minimal poster", "lifestyle scene"}:
            score += 0.15
        if axes.get("color_direction") in {"clean bright", "bold accent"}:
            score += 0.12
        if axes.get("emotional_style") in {"premium", "warm", "aspirational"}:
            score += 0.10
        return min(1.0, score)

    def _brand_safety(self, text: str) -> float:
        severe_terms = (
            "guaranteed cure",
            "miracle",
            "risk-free forever",
            "instant weight loss",
            "doctor approved",
            "clinically proven",
        )
        aggressive_terms = (
            "best ever",
            "ultimate",
            "perfect",
            "zero noise",
            "total silence",
            "instantly healthier",
        )
        if any(term in text for term in severe_terms):
            return 0.45
        if any(term in text for term in aggressive_terms):
            return 0.78
        return 1.0

    def _factuality_penalty(self, brief: CampaignBrief, text: str) -> float:
        penalty = 0.0
        disallowed = (
            "cure",
            "medical",
            "weight loss",
            "100% guaranteed",
            "free forever",
            "doctor approved",
            "clinically proven",
            "best ever",
            "zero noise",
            "total silence",
            "perfect health",
        )
        if any(term in text for term in disallowed):
            penalty += 0.12
        promotional_superlatives = (
            "perfect for",
            "ideal for",
            "ultimate",
        )
        promo_hits = sum(1 for term in promotional_superlatives if term in text)
        if promo_hits:
            penalty += min(0.08, 0.04 * promo_hits)
        constraint_text = " ".join(brief.constraints).lower()
        if "avoid health claims" in constraint_text and any(token in text for token in ("health", "healthier", "wellness", "weight loss")):
            penalty += 0.08
        if "avoid medical guarantees" in constraint_text and any(token in text for token in ("guarantee", "guaranteed", "doctor", "medical")):
            penalty += 0.10
        if "avoid weight-loss claims" in constraint_text and any(token in text for token in ("weight loss", "burn fat", "slim")):
            penalty += 0.10
        if "avoid impossible silence claims" in constraint_text and any(token in text for token in ("silent", "zero noise", "total silence")):
            penalty += 0.10
        if "avoid medical nutrition claims" in constraint_text and any(token in text for token in ("healthier", "doctor", "medical", "nutrition cure")):
            penalty += 0.10
        return min(0.3, penalty)

    def _predicted_engagement(
        self,
        candidate: CreativeCandidate,
        relevance: float,
        clarity: float,
        aesthetic: float,
        audience_fit: float,
        brand_safety: float,
    ) -> float:
        cta_bonus = 0.05 if candidate.cta in {"Shop now", "Try today"} else 0.02
        style_bonus = 0.04 if candidate.exploration_axes.get("appeal_type") in {"value", "productivity"} else 0.02
        trust_term = 0.10 * brand_safety
        return min(
            1.0,
            0.26 * relevance
            + 0.22 * clarity
            + 0.22 * aesthetic
            + 0.12 * audience_fit
            + trust_term
            + cta_bonus
            + style_bonus,
        )

    def _diversity_scores(self, candidates: list[CreativeCandidate]) -> dict[str, float]:
        if len(candidates) <= 1:
            return {candidate.candidate_id: 0.0 for candidate in candidates}
        axis_sets = {
            candidate.candidate_id: set(candidate.exploration_axes.items())
            for candidate in candidates
        }
        scores: dict[str, float] = {}
        for candidate_id, axes in axis_sets.items():
            distances = []
            for other_id, other_axes in axis_sets.items():
                if candidate_id == other_id:
                    continue
                union = axes | other_axes
                intersection = axes & other_axes
                distances.append(1.0 - len(intersection) / len(union))
            scores[candidate_id] = statistics.mean(distances)
        return scores


class EaiCoOptimizer:
    def __init__(
        self,
        generator: CreativeGenerator,
        evaluator: MultiObjectiveEvaluator,
        rounds: int = 3,
        candidates_per_round: int = 6,
        elite_count: int = 3,
    ) -> None:
        self.generator = generator
        self.evaluator = evaluator
        self.rounds = rounds
        self.candidates_per_round = candidates_per_round
        self.elite_count = elite_count

    def optimize(
        self,
        brief: CampaignBrief,
        method: str = "Ours_EAI_CO",
        evaluation_brief: CampaignBrief | None = None,
    ) -> tuple[CreativeCandidate, list[CreativeCandidate]]:
        scoring_brief = evaluation_brief or brief
        all_candidates: list[CreativeCandidate] = []
        elites: list[CreativeCandidate] = []
        for round_index in range(1, self.rounds + 1):
            candidates = self.generator.generate_candidates(
                brief=brief,
                method=method,
                round_index=round_index,
                count=self.candidates_per_round,
                elites=elites,
            )
            evaluated = self.evaluator.evaluate_group(scoring_brief, candidates)
            all_candidates.extend(evaluated)
            elites = sorted(evaluated, key=lambda item: item.scores["reward"], reverse=True)[: self.elite_count]
        final = sorted(all_candidates, key=lambda item: item.scores["reward"], reverse=True)[0]
        final.model_calls = sum(item.model_calls for item in all_candidates)
        final.latency_ms = sum(item.latency_ms for item in all_candidates)
        return final, all_candidates


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def stable_index(text: str, size: int) -> int:
    if size <= 0:
        return 0
    return stable_int(text) % size


def pareto_front(candidates: Iterable[CreativeCandidate], objectives: tuple[str, ...]) -> list[CreativeCandidate]:
    items = list(candidates)
    front: list[CreativeCandidate] = []
    for candidate in items:
        dominated = False
        for challenger in items:
            if challenger is candidate:
                continue
            if all(challenger.scores[obj] >= candidate.scores[obj] for obj in objectives) and any(
                challenger.scores[obj] > candidate.scores[obj] for obj in objectives
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def summarize_by_method(candidates: Iterable[CreativeCandidate]) -> list[dict[str, str | float | int]]:
    grouped = itertools.groupby(
        sorted(candidates, key=lambda item: item.method),
        key=lambda item: item.method,
    )
    rows: list[dict[str, str | float | int]] = []
    for method, group in grouped:
        items = list(group)
        rows.append(
            {
                "method": method,
                "n": len(items),
                "mean_reward": round(statistics.mean(item.scores.get("reward", 0.0) for item in items), 4),
                "mean_relevance": round(statistics.mean(item.scores.get("relevance", 0.0) for item in items), 4),
                "mean_clarity": round(statistics.mean(item.scores.get("clarity", 0.0) for item in items), 4),
                "mean_aesthetic": round(statistics.mean(item.scores.get("aesthetic", 0.0) for item in items), 4),
                "mean_audience_fit": round(statistics.mean(item.scores.get("audience_fit", 0.0) for item in items), 4),
                "mean_brand_safety": round(statistics.mean(item.scores.get("brand_safety", 0.0) for item in items), 4),
                "mean_factuality_penalty": round(
                    statistics.mean(item.scores.get("factuality_penalty", 0.0) for item in items),
                    4,
                ),
                "mean_predicted_engagement": round(
                    statistics.mean(item.scores.get("predicted_engagement", 0.0) for item in items),
                    4,
                ),
                "mean_model_calls": round(statistics.mean(item.model_calls for item in items), 2),
                "mean_latency_ms": round(statistics.mean(item.latency_ms for item in items), 2),
            }
        )
    return rows

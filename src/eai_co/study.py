from __future__ import annotations

import csv
import math
from pathlib import Path


def load_products(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def synthesize_products(base_rows: list[dict[str, str]], target_count: int) -> list[dict[str, str]]:
    editions = [
        "Lite", "Plus", "Air", "Go", "Pro", "Max", "Mini", "Studio", "Prime", "Flex",
        "Ultra", "Smart", "Select", "Core", "Edge", "Pulse", "Nova", "Ease", "Flow", "Hub",
    ]
    extra_points = {
        "Consumer electronics": [
            "fast charging", "travel case included", "one-touch control", "quiet operation"
        ],
        "Home office": [
            "space-saving footprint", "touch control", "stable base", "night mode"
        ],
        "Lifestyle": [
            "daily habit support", "easy carry loop", "minimal design", "battery efficient"
        ],
        "Audio": [
            "low-latency mode", "stable Bluetooth connection", "compact case", "easy pairing"
        ],
        "Kitchen appliance": [
            "family-size basket", "simple controls", "easy storage", "rapid heating"
        ],
    }
    tone_rotations = ["energetic", "clear", "motivational", "premium", "warm", "direct"]
    expanded: list[dict[str, str]] = []
    variants_per_base = math.ceil(target_count / len(base_rows))
    for base_index, row in enumerate(base_rows):
        category = row["category"]
        category_extras = extra_points.get(category, ["easy use", "reliable design"])
        for variant_index in range(variants_per_base):
            edition = editions[variant_index % len(editions)]
            extra_point = category_extras[variant_index % len(category_extras)]
            selling_points = [item.strip() for item in row["selling_points"].split(";") if item.strip()]
            if extra_point not in selling_points:
                selling_points.append(extra_point)
            expanded.append(
                {
                    "product_id": f"{row['product_id']}_{variant_index + 1:02d}",
                    "product_title": f"{row['product_title']} {edition}",
                    "category": category,
                    "selling_points": "; ".join(selling_points),
                    "tone": tone_rotations[(base_index + variant_index) % len(tone_rotations)],
                    "constraints": row.get("constraints", ""),
                }
            )
    return expanded[:target_count]

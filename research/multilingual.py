"""Explicit language-family schedules for multilingual ablations."""

LANGUAGE_FAMILY = {
    "hi": "indo_aryan", "bn": "indo_aryan", "mr": "indo_aryan", "gu": "indo_aryan",
    "pa": "indo_aryan", "ne": "indo_aryan", "ta": "dravidian", "te": "dravidian",
    "kn": "dravidian", "ml": "dravidian",
}


def temperature_weights(pair_counts: dict[str, int], temperature: float = 5.0) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = {lang: count ** (1.0 / temperature) for lang, count in pair_counts.items() if count > 0}
    total = sum(values.values())
    return {lang: value / total for lang, value in values.items()}


def family_ablation_order(target: str = "hi") -> list[str]:
    family = LANGUAGE_FAMILY[target]
    same_family = sorted(lang for lang, value in LANGUAGE_FAMILY.items() if value == family and lang != target)
    other = sorted(lang for lang, value in LANGUAGE_FAMILY.items() if value != family)
    return same_family + other

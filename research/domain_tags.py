"""Multilingual & Domain Conditioning Tags (Part A, Sections 16 & 23).

Supports prepending language target tags (<2hi>, <2bn>) and domain tags (<domain:news>, <domain:legal>).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

TAG_PATTERN = re.compile(r"^(?:<2\w+>|<domain:\w+>)\s*")


def format_tagged_sentence(text: str, target_lang: Optional[str] = None, domain: Optional[str] = None) -> str:
    """Prepend domain and language conditioning tags to source sentence."""
    prefix_parts = []
    if domain:
        prefix_parts.append(f"<domain:{domain.lower()}>")
    if target_lang:
        prefix_parts.append(f"<2{target_lang.lower()}>")

    prefix = " ".join(prefix_parts)
    return f"{prefix} {text.strip()}".strip() if prefix else text.strip()


def extract_tags(text: str) -> Tuple[str, dict]:
    """Extract conditioning tags from sentence prefix."""
    meta = {}
    cleaned = text
    while True:
        match = TAG_PATTERN.match(cleaned)
        if not match:
            break
        tag = match.group(0).strip()
        if tag.startswith("<2") and tag.endswith(">"):
            meta["target_lang"] = tag[2:-1]
        elif tag.startswith("<domain:") and tag.endswith(">"):
            meta["domain"] = tag[8:-1]
        cleaned = cleaned[match.end():]
    return cleaned.strip(), meta

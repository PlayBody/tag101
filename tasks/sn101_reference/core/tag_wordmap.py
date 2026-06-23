"""Explicit hashtag/compound normalizations for consensus tagging.

Only listed compact tokens are rewritten. Everything else keeps normalize_tag()
behavior — no automatic CamelCase splitting (that broke brands like deepseek,
scikit-learn, adalovelaceday).
"""

from __future__ import annotations

import re

from .scoring.preprocessing import normalize_tag

# compact lowercase token -> canonical tag
_EXPLICIT_MAP: dict[str, str] = {
    # Hashtag compounds seen in live SN101 tasks (keep forms span miners reuse)
    "artificialintelligence": "artificial intelligence",
    "climateemergency": "climate emergency",
    "climateaction": "climate action",
    "climatecrisis": "climate crisis",
    "climatecr": "climate crisis",
    "futureofwork": "future of work",
    "unitedhealth": "united health",
    "unitedhealthcare": "united health",
    "privateinsurers": "private insurers",
    "stablediffusion": "stable diffusion",
    "adalovelaceday": "ada lovelace day",
    "generativemodelling": "generative modelling",
    "latentspace": "latent space",
    "machinelearning": "machine learning",
    "deeplearning": "deep learning",
    "largelanguagemodel": "large language model",
    "largelanguagemodels": "large language models",
    "socialmedia": "social media",
    "executiveorder": "executive order",
    "benefitcorporation": "benefit corporation",
    "humanoidrobot": "humanoid robot",
    "humanoidrobots": "humanoid robots",
    # Brands / products — preserve as single tokens
    "openai": "openai",
    "chatgpt": "chatgpt",
    "anthropic": "anthropic",
    "nvidia": "nvidia",
    "midjourney": "midjourney",
    "deepseek": "deepseek",
    "openclaw": "openclaw",
    "aitemplate": "aitemplate",
    "xai": "xai",
    "gemini": "gemini",
    "llama": "llama",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "scikitlearn": "scikit-learn",
    "numpy": "numpy",
    "keras": "keras",
    "gradio": "gradio",
}


def _compact(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", text.lower())


def smart_tag(raw: str) -> str:
    """Normalize a candidate tag; apply explicit map only when listed."""
    text = str(raw or "").strip()
    if not text:
        return ""

    if text.count("#") >= 2:
        return ""

    stripped = text.lstrip("#").strip()
    if not stripped:
        return ""

    stripped = re.sub(r"@\w+", " ", stripped)
    stripped = re.sub(r"https?://\S+|www\.\S+", " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"[''`\u2018\u2019]", " ", stripped)
    stripped = re.sub(r"(\w)'s\b", r"\1", stripped, flags=re.IGNORECASE)
    stripped = stripped.replace("*", " ")
    stripped = re.sub(r"[^\w\s\-+#&/]", " ", stripped)
    stripped = re.sub(r"\b(i|you|we|they|he|she|it)(?:\s+)(ve|re|ll|d|m)\b", r"\1", stripped, flags=re.I)
    stripped = re.sub(r"\b(\w+)\s+(ve|re|ll|d|m)\b(?=\s|$)", r"\1", stripped, flags=re.I)
    stripped = re.sub(r"\b(\w+)\s+s\b(?=\s|$)", r"\1", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if not stripped:
        return ""

    mapped = _EXPLICIT_MAP.get(_compact(stripped))
    if mapped:
        return normalize_tag(mapped)

    return normalize_tag(stripped)

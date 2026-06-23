"""Map hashtag/compound tokens to consensus-friendly tag phrases.

Prefer **short canonical tags** (ai, llm, ml) for broad tech topics when miners
cluster on acronyms. Prefer **2-word phrases** for named entities/topics
(united health, stable diffusion, ai policy) when specificity helps consensus.
"""

from __future__ import annotations

import re

from .scoring.preprocessing import normalize_tag

# Compact (no spaces) -> preferred tag. Favor short forms miners repeat.
_TAG_WORDMAP: dict[str, str] = {
    # AI / ML acronyms and umbrella topics
    "ai": "ai",
    "artificialintelligence": "ai",
    "machinelearning": "ml",
    "deeplearning": "dl",
    "generativeai": "gen ai",
    "genai": "gen ai",
    "largelanguagemodel": "llm",
    "largelanguagemodels": "llm",
    "llm": "llm",
    "llms": "llm",
    "naturallanguageprocessing": "nlp",
    "computervision": "cv",
    "reinforcementlearning": "rl",
    "retrievalaugmentedgeneration": "rag",
    "neuralnetwork": "neural network",
    "neuralnetworks": "neural network",
    "transformer": "transformer",
    "transformers": "transformers",
    "diffusion": "diffusion",
    "stablediffusion": "stable diffusion",
    "finetuning": "fine tuning",
    "finetune": "fine tuning",
    "pretraining": "pretraining",
    "posttraining": "post training",
    "rlhf": "rlhf",
    "promptengineering": "prompt engineering",
    "agents": "agents",
    "aiagent": "ai agent",
    "aiagents": "ai agents",
    "multimodal": "multimodal",
    "multimodalai": "multimodal ai",
    # AI policy / industry (2 words — more specific than "ai")
    "aipolicy": "ai policy",
    "airegulation": "ai regulation",
    "aigovernance": "ai governance",
    "airesafety": "ai safety",
    "airesponsibility": "ai responsibility",
    "responsibleai": "responsible ai",
    "aibill": "ai bill",
    "aiact": "ai act",
    "aitech": "ai tech",
    "ainews": "ai news",
    "airesearch": "ai research",
    "aimodel": "ai model",
    "aimodels": "ai models",
    "aitools": "ai tools",
    "aiproducts": "ai products",
    "aistartup": "ai startup",
    "aistartups": "ai startups",
    "aiindustry": "ai industry",
    "aicompute": "ai compute",
    "aiinfrastructure": "ai infrastructure",
    "aiops": "ai ops",
    # Tech / dev
    "opensource": "open source",
    "openweight": "open weight",
    "openweights": "open weights",
    "webdevelopment": "web dev",
    "softwareengineering": "software engineering",
    "cloudcomputing": "cloud computing",
    "cloudcompute": "cloud compute",
    "datascience": "data science",
    "dataengineering": "data engineering",
    "bigdata": "big data",
    "devops": "devops",
    "fullstack": "full stack",
    "backend": "backend",
    "frontend": "frontend",
    "cybersecurity": "cybersecurity",
    "infosec": "infosec",
    "blockchain": "blockchain",
    "crypto": "crypto",
    "web3": "web3",
    # Business / media
    "futureofwork": "future of work",
    "socialmedia": "social media",
    "technews": "tech news",
    "startup": "startup",
    "startups": "startups",
    "venturecapital": "vc",
    "siliconvalley": "silicon valley",
    "productivitytips": "productivity tips",
    "sideproject": "side project",
    "sideprojects": "side projects",
    "personaldevelopment": "personal development",
    "europeanunion": "eu",
    "euaiact": "eu ai act",
    # Companies / products (keep brand tokens)
    "openai": "openai",
    "chatgpt": "chatgpt",
    "gpt4": "gpt-4",
    "gpt5": "gpt-5",
    "gpt4o": "gpt-4o",
    "claude": "claude",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "copilot": "copilot",
    "nvidia": "nvidia",
    "microsoft": "microsoft",
    "google": "google",
    "deepmind": "deepmind",
    "meta": "meta",
    "apple": "apple",
    "amazon": "amazon",
    "tesla": "tesla",
    "huggingface": "hugging face",
    "mistral": "mistral",
    "llama": "llama",
    "ollama": "ollama",
    "langchain": "langchain",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "cuda": "cuda",
    "aitemplate": "aitemplate",
    # Health / other entities from real tasks
    "unitedhealth": "united health",
    "unitedhealthcare": "united health",
    "medicare": "medicare",
    "privateinsurers": "private insurers",
    "healthcare": "healthcare",
    "healthinsurance": "health insurance",
    # General topics
    "machineintelligence": "machine intelligence",
    "automation": "automation",
    "robotics": "robotics",
    "humanoid": "humanoid",
    "humanoids": "humanoids",
    "humanoidrobot": "humanoid robot",
    "humanoidrobots": "humanoid robots",
    "innovation": "innovation",
    "technology": "tech",
    "tech": "tech",
}

# Already-spaced phrases -> shorter canonical form.
_PHRASE_WORDMAP: dict[str, str] = {
    "artificial intelligence": "ai",
    "machine learning": "ml",
    "deep learning": "dl",
    "large language model": "llm",
    "large language models": "llm",
    "natural language processing": "nlp",
    "computer vision": "cv",
    "reinforcement learning": "rl",
    "retrieval augmented generation": "rag",
    "generative ai": "gen ai",
    "venture capital": "vc",
    "european union": "eu",
    "fine tuning": "fine tuning",
    "open source": "open source",
    "stable diffusion": "stable diffusion",
    "hugging face": "hugging face",
    "united health": "united health",
    "ai policy": "ai policy",
    "ai regulation": "ai regulation",
    "ai safety": "ai safety",
    "ai act": "ai act",
    "eu ai act": "eu ai act",
    "future of work": "future of work",
    "private insurers": "private insurers",
    "health insurance": "health insurance",
    "prompt engineering": "prompt engineering",
    "software engineering": "software engineering",
    "cloud computing": "cloud computing",
    "data science": "data science",
    "neural network": "neural network",
    "neural networks": "neural network",
    "humanoid robot": "humanoid robot",
    "humanoid robots": "humanoid robot",
    "web development": "web dev",
    "social media": "social media",
    "tech news": "tech news",
    "productivity tips": "productivity tips",
    "side project": "side project",
    "silicon valley": "silicon valley",
    "open weight": "open weight",
    "open weights": "open weights",
    "multimodal ai": "multimodal ai",
    "ai agent": "ai agent",
    "ai agents": "ai agents",
    "ai model": "ai model",
    "ai models": "ai models",
    "ai tools": "ai tools",
    "ai research": "ai research",
    "ai compute": "ai compute",
    "ai infrastructure": "ai infrastructure",
    "post training": "post training",
    "gen ai": "gen ai",
}

_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _compact(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", text.lower())


def _lookup_canonical(tag: str) -> str:
    phrase = _PHRASE_WORDMAP.get(tag)
    if phrase:
        return phrase
    return _TAG_WORDMAP.get(_compact(tag), tag)


def smart_tag(raw: str) -> str:
    """Normalize raw text/hashtag tokens into validator-friendly tag phrases."""
    text = str(raw or "").strip()
    if not text:
        return ""

    if text.count("#") >= 2:
        return ""

    stripped = text.lstrip("#").strip()
    if not stripped:
        return ""

    compact = _compact(stripped)
    mapped = _TAG_WORDMAP.get(compact)
    if mapped:
        return normalize_tag(mapped)

    spaced = _ACRONYM_BOUNDARY.sub(r"\1 \2", stripped)
    spaced = _CAMEL_BOUNDARY.sub(r"\1 \2", spaced)
    tag = normalize_tag(spaced)
    if not tag:
        return ""

    return normalize_tag(_lookup_canonical(tag))
